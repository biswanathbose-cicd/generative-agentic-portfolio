import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit import dataset, paths, stats  # noqa: E402,F401
from evalkit.judge import LLMJudge, RuleJudge  # noqa: E402
from evalkit.metrics import run_eval, summarize  # noqa: E402
from evalkit.retrieval_eval import run as run_retrieval  # noqa: E402
from support_agent import V1_BASELINE, V2_IMPROVED  # noqa: E402
from support_agent.llm import product_vocabulary  # noqa: E402
from support_agent.data import load_catalog  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_deterministic_and_sized(self):
        a, b = dataset.build_cases(2026, "dev"), dataset.build_cases(2026, "dev")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 285)
        self.assertEqual(len({c["id"] for c in a}), 285)

    def test_heldout_phrasings_do_not_overlap_dev(self):
        dev = {t for c in dataset.build_cases(2026, "dev") for t in c["turns"]}
        held = {t for c in dataset.build_cases(4051, "heldout") for t in c["turns"]}
        # order ids are random, so compare with ids stripped
        import re
        strip = lambda s: re.sub(r"WM-\d{5}", "<ID>", s)  # noqa: E731
        dev_t = {strip(t) for t in dev if not any(ch.isdigit() for ch in strip(t))}
        held_t = {strip(t) for t in held if not any(ch.isdigit() for ch in strip(t))}
        self.assertFalse(dev_t & held_t, dev_t & held_t)

    def test_expectation_fields(self):
        for c in dataset.build_cases(2026, "dev"):
            for k in ("intent", "outcomes", "required_tools", "forbidden_tools"):
                self.assertIn(k, c["expected"])

    def test_dev_out_of_scope_templates_avoid_product_vocabulary(self):
        import re
        vocab = product_vocabulary(load_catalog())
        for t in dataset.TEMPLATES["dev"]["out_of_scope"]:
            toks = set(re.findall(r"[a-z][a-z'-]*", t.lower()))
            overlap = toks & vocab
            # known, documented polysemy: 'watch' (verb vs smart watch) is one of the dev failures
            self.assertTrue(overlap <= {"watch"}, (t, overlap))


class StatsTests(unittest.TestCase):
    def test_sample_size_matches_textbook(self):
        n = stats.sample_size_two_proportions(0.5, 0.6)
        self.assertTrue(380 <= n <= 395, n)

    def test_ztest(self):
        r = stats.two_proportion_ztest(500, 1000, 600, 1000)
        self.assertAlmostEqual(r["diff"], 0.1)
        self.assertLess(r["p_value"], 1e-4)
        self.assertLess(r["ci"][0], 0.1)
        self.assertGreater(r["ci"][1], 0.1)

    def test_mcnemar(self):
        a = np.array([1] * 5 + [0] * 15, bool)
        b = np.array([1] * 5 + [1] * 15, bool)
        r = stats.mcnemar_exact(a, b)
        self.assertEqual((r["only_a_correct"], r["only_b_correct"]), (0, 15))
        self.assertLess(r["p_value"], 0.001)

    def test_kappa(self):
        x = np.array([1, 0, 1, 1, 0], bool)
        self.assertEqual(stats.cohen_kappa(x, x), 1.0)
        self.assertLess(stats.cohen_kappa(x, ~x), 0)

    def test_cuped_reduces_variance_by_about_rho_squared(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=20000)
        y = 0.6 * x + 0.8 * rng.normal(size=20000)
        self.assertAlmostEqual(stats.cuped(y, x)["variance_reduction"], 0.36, delta=0.03)

    def test_peeking_inflates_type_one_error(self):
        single = stats.simulate_ab(0.7, 0.7, 1000, n_sims=1500, seed=1)["reject_rate"]
        peek = stats.peeking_false_positive_rate(0.7, 1000, looks=5, n_sims=600, seed=1)
        self.assertLess(single, 0.08)
        self.assertGreater(peek, single + 0.03)

    def test_power_close_to_target(self):
        n = stats.sample_size_two_proportions(0.7, 0.72)
        self.assertAlmostEqual(stats.simulate_ab(0.7, 0.72, n, n_sims=3000, seed=2)["reject_rate"], 0.8, delta=0.04)


class JudgeTests(unittest.TestCase):
    CASE = {"expected": {"intent": "product_search", "pii_strings": []}}

    def test_rule_judge(self):
        j = RuleJudge()
        self.assertTrue(j.judge(self.CASE, "Here are some options: X ($5.00).", "")["pass"])
        self.assertFalse(j.judge(self.CASE, "I only help with orders.", "")["pass"])

    def test_llm_judge_parsing(self):
        ok = LLMJudge(lambda s, u: 'noise {"pass": true, "reason": "fine"} trailing')
        self.assertTrue(ok.judge(self.CASE, "r", "q")["pass"])
        bad = LLMJudge(lambda s, u: "no json here")
        self.assertFalse(bad.judge(self.CASE, "r", "q")["pass"])


class PipelineTests(unittest.TestCase):
    def test_v2_beats_v1_on_a_small_slice(self):
        cases = dataset.build_cases(2026, "dev")[::6]
        s1 = summarize(run_eval(V1_BASELINE, cases))["task_success"]
        s2 = summarize(run_eval(V2_IMPROVED, cases))["task_success"]
        self.assertGreater(s2, s1 + 0.3)

    def test_hybrid_retrieval_helps_on_typos(self):
        r = run_retrieval(n=60)
        self.assertGreater(r["typo_queries"]["hybrid"]["recall@5"], r["typo_queries"]["word_tfidf"]["recall@5"])


if __name__ == "__main__":
    unittest.main()
