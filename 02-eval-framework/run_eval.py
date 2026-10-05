"""Run the full evaluation + experimentation study and write ``results/``.

    python run_eval.py

Outputs: results/summary.json, results/report.md, results/case_results.csv, results/*.png
Every number in the report is produced by this script; nothing is hand-edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evalkit import dataset, paths  # noqa: F401
from evalkit import retrieval_eval, stats
from evalkit.error_analysis import cluster_failures, failure_table
from evalkit.judge import RuleJudge
from evalkit.metrics import by_category, run_eval, summarize
from support_agent import V1_BASELINE, V2_1_FIXED, V2_IMPROVED

RESULTS = Path(__file__).resolve().parent / "results"
VERSIONS = [V1_BASELINE, V2_IMPROVED, V2_1_FIXED]
COLORS = {"v1_baseline": "#2a78d6", "v2_improved": "#eb6834", "v2.1_fuzzy_fix": "#1baf7a"}  # categorical slots 1-3
INK, MUTED = "#0b0b0b", "#52514e"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor("#fcfcfb")


def fig_success_by_version(summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor="#fcfcfb")
    splits, w = ["dev", "heldout"], 0.26
    for i, v in enumerate(VERSIONS):
        vals = [summary[s][v.name]["task_success"] * 100 for s in splits]
        xs = np.arange(len(splits)) + (i - 1) * (w + 0.02)
        bars = ax.bar(xs, vals, w, color=COLORS[v.name], edgecolor="#fcfcfb", linewidth=2, label=v.name)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(["dev phrasings\n(used while building)", "held-out phrasings\n(never tuned on)"])
    ax.set_ylim(0, 125)
    ax.set_ylabel("task success (%)", color=MUTED)
    ax.set_title("Task success: dev vs held-out set (285 cases each)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_category(df_dev: pd.DataFrame, df_held: pd.DataFrame, path: Path) -> None:
    d = by_category(df_dev).success_rate * 100
    h = by_category(df_held).success_rate * 100
    order = h.sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor="#fcfcfb")
    y = np.arange(len(order))
    ax.barh(y + 0.19, d[order], 0.34, color="#2a78d6", edgecolor="#fcfcfb", linewidth=2, label="dev")
    ax.barh(y - 0.19, h[order], 0.34, color="#eb6834", edgecolor="#fcfcfb", linewidth=2, label="held-out")
    for yi, c in zip(y, order):
        ax.text(h[c] + 1.5, yi - 0.19, f"{h[c]:.0f}%", va="center", fontsize=8, color=INK)
        ax.text(d[c] + 1.5, yi + 0.19, f"{d[c]:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("task success (%)", color=MUTED)
    ax.set_title(f"{V2_1_FIXED.name}: success by case category", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = {s: dataset.load(s) for s in ("dev", "heldout")}
    cases_by_id = {c["id"]: c for s in data.values() for c in s}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary: dict = {}
    judge = RuleJudge()
    judge_rows = []
    for split, cases in data.items():
        summary[split] = {}
        for cfg in VERSIONS:
            df = run_eval(cfg, cases)
            df["split"] = split
            df["judge_pass"] = [judge.judge(cases_by_id[r.id], r.response, r.final_user_text)["pass"]
                                for r in df.itertuples()]
            frames[(split, cfg.name)] = df
            summary[split][cfg.name] = summarize(df)
            ok, jp = df.success.to_numpy(), df.judge_pass.to_numpy()
            judge_rows.append({
                "split": split, "version": cfg.name, "task_success": ok.mean(), "judge_pass": jp.mean(),
                "kappa(judge, task_success)": stats.cohen_kappa(jp, ok),
                "judge_passes_among_task_failures": float(jp[~ok].mean()) if (~ok).any() else float("nan"),
            })
    pd.concat(frames.values()).drop(columns=["response"]).to_csv(RESULTS / "case_results.csv", index=False)

    # ---- paired comparisons (same cases, so McNemar, not an unpaired z-test)
    paired = []
    for split in data:
        for a, b in ((V1_BASELINE, V2_IMPROVED), (V2_IMPROVED, V2_1_FIXED)):
            sa, sb = frames[(split, a.name)].success.to_numpy(), frames[(split, b.name)].success.to_numpy()
            mc = stats.mcnemar_exact(sa, sb)
            lo, hi = stats.paired_bootstrap_ci(sa, sb)
            paired.append({"split": split, "comparison": f"{a.name} -> {b.name}", "diff_pp": (sb.mean() - sa.mean()) * 100,
                           "ci95_lo_pp": lo * 100, "ci95_hi_pp": hi * 100, "only_b_correct": mc["only_b_correct"],
                           "only_a_correct": mc["only_a_correct"], "mcnemar_p": mc["p_value"]})

    # ---- retrieval
    retr = retrieval_eval.run()

    # ---- error analysis
    dev_v2 = frames[("dev", V2_IMPROVED.name)]
    held_v21 = frames[("heldout", V2_1_FIXED.name)]
    dev_fail_tbl, dev_clusters = failure_table(dev_v2), cluster_failures(dev_v2, k=3)
    held_fail_tbl, held_clusters = failure_table(held_v21), cluster_failures(held_v21, k=8)

    # ---- experimentation: planning + simulation (assumed parameters, clearly labelled)
    p0, mde = 0.70, 0.02
    n_plan = stats.sample_size_two_proportions(p0, p0 + mde)
    n_plan_5 = stats.sample_size_two_proportions(p0, p0 + 0.05)
    sim_power = stats.simulate_ab(p0, p0 + mde, n_plan, n_sims=4000, seed=1)["reject_rate"]
    sim_aa = stats.simulate_ab(p0, p0, n_plan, n_sims=4000, seed=2)["reject_rate"]
    peek = stats.peeking_false_positive_rate(p0, 2000, looks=5, n_sims=2000, seed=3)
    one_look = stats.simulate_ab(p0, p0, 2000, n_sims=4000, seed=4)["reject_rate"]
    rng = np.random.default_rng(7)
    n_u, rho = 6000, 0.6
    x = rng.normal(size=n_u)
    y_ctrl = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=n_u)
    cup = stats.cuped(y_ctrl, x)
    exp_sim = {
        "assumed_baseline_success": p0, "mde_abs": mde, "n_per_arm_for_2pp": n_plan, "n_per_arm_for_5pp": n_plan_5,
        "simulated_power_at_planned_n": sim_power, "simulated_aa_false_positive_rate": sim_aa,
        "aa_false_positive_rate_single_look_n2000": one_look, "aa_false_positive_rate_5_peeks_n2000": peek,
        "cuped_assumed_rho": rho, "cuped_variance_reduction": cup["variance_reduction"],
    }

    summary["paired"] = paired
    summary["judge"] = judge_rows
    summary["retrieval"] = retr
    summary["experimentation_simulation"] = exp_sim
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    fig_success_by_version(summary, RESULTS / "success_dev_vs_heldout.png")
    fig_category(frames[("dev", V2_1_FIXED.name)], held_v21, RESULTS / "success_by_category.png")

    # ---- report.md
    keys = ["task_success", "intent_accuracy", "tool_call_accuracy", "injection_refusal_rate",
            "injection_unsafe_action_rate", "pii_leak_rate", "policy_violation_rate_ineligible_returns",
            "product_top1_category_accuracy", "llm_calls_per_case"]
    rows = []
    for split in data:
        for cfg in VERSIONS:
            rows.append({"split": split, "version": cfg.name, **{k: summary[split][cfg.name][k] for k in keys}})
    main_tbl = pd.DataFrame(rows)
    retr_rows = [{"queries": q, "retriever": m, **vals} for q, d in retr.items() for m, vals in d.items()]
    report = f"""# Evaluation report (auto-generated by `run_eval.py`)

> All data is synthetic and template-generated; the 'LLM' is a deterministic rule-based stand-in. These numbers
> measure the **evaluation pipeline and the assistant's orchestration logic**, not the quality of any real language
> model. Treat the dev-vs-held-out gap as the main finding.

## 1. Headline metrics (285 cases per split)
{md_table(main_tbl)}

`product_top1_category_accuracy` is over *all* product cases, so a case the router never sends to search counts as a miss.
`policy_violation_rate_ineligible_returns` = share of ineligible-return cases where `initiate_return` was called.

![success](success_dev_vs_heldout.png)

## 2. Paired comparisons (same cases -> exact McNemar + paired bootstrap CI)
{md_table(pd.DataFrame(paired), "{:.2f}")}

## 3. Does a text-only judge catch behavioural failures?
`RuleJudge` reads only the reply text. `judge_passes_among_task_failures` is how often it still says "pass" when the
trace-based label says the task failed.
{md_table(pd.DataFrame(judge_rows))}

## 4. Retrieval (recall@5, MRR; 100 queries each)
{md_table(pd.DataFrame(retr_rows))}

## 5. Error analysis
### 5a. v2_improved failures on the dev set (the analysis that produced v2.1)
{md_table(dev_fail_tbl, "{:.0f}")}

Clusters of the failed user turns:
{md_table(dev_clusters)}

### 5b. v2.1 failures on the held-out set (not tuned on; this is the roadmap)
{md_table(held_fail_tbl, "{:.0f}")}

{md_table(held_clusters)}

![categories](success_by_category.png)

## 6. Experimentation planning (simulation with ASSUMED parameters)
Assumed online baseline success = {p0:.0%}; minimum detectable effect = +{mde*100:.0f}pp.

| quantity | value |
|---|---|
| n per arm to detect +2pp (80% power, alpha 0.05) | {n_plan:,} |
| n per arm to detect +5pp | {n_plan_5:,} |
| simulated power at the planned n | {sim_power:.3f} |
| simulated false-positive rate, A/A at planned n | {sim_aa:.3f} |
| A/A false-positive rate, single look (n=2000/arm) | {one_look:.3f} |
| A/A false-positive rate, stop-at-first-significant over 5 looks | {peek:.3f} |
| CUPED variance reduction (assumed pre/post correlation {rho}) | {cup['variance_reduction']:.3f} |
"""
    (RESULTS / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
