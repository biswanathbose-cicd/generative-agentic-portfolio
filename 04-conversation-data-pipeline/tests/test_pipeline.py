import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import generate_raw, monitoring, stages  # noqa: E402
from pipeline.pii import contains_pii, redact  # noqa: E402
from pipeline.schema import validate_event  # noqa: E402

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def good_event(**kw):
    ev = {"event_id": "e1", "session_id": "s1", "turn": 1, "timestamp": "2026-10-01T10:00:00+00:00",
          "user_text": "where is my order WM-10003", "intent": "order_status", "outcome": "answered",
          "tools": ["get_order_status"], "response": "ok", "latency_ms": 12.5, "guardrail_flags": [],
          "feedback": None}
    ev.update(kw)
    return ev


class SchemaTests(unittest.TestCase):
    def test_valid_event(self):
        self.assertEqual(validate_event(good_event(), NOW), [])

    def test_each_defect_is_caught(self):
        cases = {
            "missing:intent": {"intent": None},
            "bad_timestamp": {"timestamp": "yesterday"},
            "future_timestamp": {"timestamp": (NOW + timedelta(days=3)).isoformat()},
            "stale_timestamp": {"timestamp": "2019-01-01T00:00:00+00:00"},
            "type:latency_ms": {"latency_ms": "12ms"},
            "range:latency_ms": {"latency_ms": -3},
            "enum:intent": {"intent": "smalltalk_v2"},
            "range:turn": {"turn": 0},
            "type:turn": {"turn": "1"},
            "type:tools": {"tools": "get_order_status"},
            "enum:feedback": {"feedback": "meh"},
        }
        for code, patch in cases.items():
            self.assertIn(code, validate_event(good_event(**patch), NOW), code)

    def test_booleans_are_not_numbers(self):
        self.assertIn("type:latency_ms", validate_event(good_event(latency_ms=True), NOW))


class PIITests(unittest.TestCase):
    def test_redacts_all_types_including_parenthesised_phone(self):
        text = "card 4242 4242 4242 4242, a@b.com, (555) 123-4567, 555-867-5309, ssn 123-45-6789"
        out, counts = redact(text)
        self.assertFalse(contains_pii(out), out)
        self.assertEqual(set(counts), {"credit_card", "email", "phone", "ssn"})

    def test_no_false_positives_on_order_ids_and_prices(self):
        for t in ("where is WM-10003", "show me earbuds under $40", "tracking TRK3269380278 arrives 2026-10-05"):
            self.assertEqual(redact(t)[0], t)


class StageTests(unittest.TestCase):
    def test_invalid_json_and_non_object_lines_are_quarantined(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text(json.dumps(good_event()) + "\n" + '{"event_id": "x", "sess\n' + "[1,2]\n\n")
            good, bad, n = stages.ingest_validate(p, NOW)
        self.assertEqual((len(good), len(bad), n), (1, 2, 3))
        self.assertEqual({e for q in bad for e in q["errors"]}, {"invalid_json", "not_an_object"})

    def test_dedupe_keeps_earliest_copy(self):
        a = good_event(timestamp="2026-10-01T10:00:00+00:00", response="first")
        b = good_event(timestamp="2026-10-01T10:00:05+00:00", response="second")
        kept, dropped = stages.dedupe([b, a])
        self.assertEqual((len(kept), dropped, kept[0]["response"]), (1, 1, "first"))

    def test_enrich_builds_session_table(self):
        evs = [good_event(event_id=f"e{i}", turn=i, timestamp=f"2026-10-01T10:00:{i * 10:02d}+00:00",
                          outcome="handoff" if i == 2 else "answered") for i in (1, 2, 3)]
        df, sessions = stages.enrich(evs)
        self.assertEqual(len(sessions), 1)
        row = sessions.iloc[0]
        self.assertEqual((row.n_turns, row.n_events, bool(row.escalated)), (3, 3, True))
        self.assertAlmostEqual(row.duration_s, 20.0)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        raw = Path(cls.tmp.name) / "raw.jsonl"
        cls.truth = generate_raw.generate(raw, n_sessions_per_day=40, seed=5)
        good, cls.bad, cls.n_lines = stages.ingest_validate(raw, generate_raw.NOW)
        deduped, cls.n_dupes = stages.dedupe(good)
        clean, _, cls.pii_events = stages.redact_events(deduped)
        cls.df, cls.sessions = stages.enrich(clean)
        stages.publish(cls.df, cls.sessions, cls.bad, Path(cls.tmp.name) / "out")
        cls.curated = stages.read_curated(Path(cls.tmp.name) / "out")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_injected_defect_is_removed(self):
        self.assertFalse(set(self.truth["defects"]) & set(self.curated.event_id))

    def test_no_clean_event_is_lost_or_quarantined(self):
        self.assertEqual(set(self.truth["clean_ids"]), set(self.curated.event_id))

    def test_duplicates_dropped_exactly(self):
        self.assertEqual(self.n_dupes, self.truth["duplicates"])
        self.assertTrue(self.curated.event_id.is_unique)

    def test_truncated_json_quarantined(self):
        self.assertEqual(sum(q["errors"] == ["invalid_json"] for q in self.bad), self.truth["invalid_json"])

    def test_no_pii_left_and_counts_match(self):
        self.assertFalse(self.curated.user_text.fillna("").map(contains_pii).any())
        self.assertEqual(self.pii_events, len(self.truth["pii_events"]))

    def test_sessions_are_not_collapsed(self):
        self.assertGreater(len(self.sessions), 1000)

    def test_partitions_written_per_day(self):
        parts = list(Path(self.tmp.name, "out", "events").glob("date=*/part-0.csv"))
        # 30 days of traffic starting at noon spans 31 calendar dates
        self.assertEqual(len(parts), self.df.date.nunique())
        self.assertGreaterEqual(len(parts), generate_raw.DAYS)


class MonitoringTests(unittest.TestCase):
    def test_psi_zero_for_identical_and_large_for_shifted(self):
        a = pd.Series(["x"] * 50 + ["y"] * 50)
        self.assertAlmostEqual(monitoring.psi(a, a), 0.0, places=6)
        self.assertGreater(monitoring.psi(a, pd.Series(["x"] * 10 + ["y"] * 90)), 0.2)

    def test_drift_flagged_on_generated_incident_and_not_without_it(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "r.jsonl"
            generate_raw.generate(raw, n_sessions_per_day=60, seed=9)
            good, _, _ = stages.ingest_validate(raw, generate_raw.NOW)
            df, _ = stages.enrich(stages.redact_events(stages.dedupe(good)[0])[0])
        drift = monitoring.detect_drift(df)
        self.assertTrue(any(a["signal"] == "handoff_rate_increase" for a in drift["alerts"]))
        calm = df[df.date < sorted(df.date.unique())[-7]]  # drop the drifted final week
        self.assertEqual(monitoring.detect_drift(calm, baseline_days=10, recent_days=7)["alerts"], [])


if __name__ == "__main__":
    unittest.main()
