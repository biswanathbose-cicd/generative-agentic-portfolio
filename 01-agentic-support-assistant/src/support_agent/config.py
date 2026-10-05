"""Feature flags that define an assistant "version".

Two presets are shipped so the evaluation framework has something real to compare:

* ``V1_BASELINE``  - the naive first cut: exact keyword routing, no memory, no guardrails,
  returns are initiated without checking policy.
* ``V2_IMPROVED``  - adds fuzzy routing, multi-turn memory, input/output guardrails,
  frustration -> human hand-off, hybrid retrieval and a return-eligibility check.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AssistantConfig:
    name: str = "v2_improved"
    use_memory: bool = True
    input_guardrails: bool = True
    output_guardrails: bool = True
    check_return_eligibility: bool = True
    fuzzy_routing: bool = True
    detect_frustration: bool = True
    hybrid_retrieval: bool = True
    strict_fuzzy: bool = False  # v2.1: <=1 edit + corroboration for corrected order/return keywords
    max_steps: int = 6  # hard cap on tool calls per turn (loop / cost control)


V1_BASELINE = AssistantConfig(
    name="v1_baseline",
    use_memory=False,
    input_guardrails=False,
    output_guardrails=False,
    check_return_eligibility=False,
    fuzzy_routing=False,
    detect_frustration=False,
    hybrid_retrieval=False,
)

V2_IMPROVED = AssistantConfig()

# Fixes found by error analysis of v2 (see 02-eval-framework): fuzzy matching turned "trick" into
# "track" and "weather" into "water".
V2_1_FIXED = replace(V2_IMPROVED, name="v2.1_fuzzy_fix", strict_fuzzy=True)
