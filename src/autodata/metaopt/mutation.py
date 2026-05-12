"""Mutation operator: LLM proposes a diff to a HarnessSpec.

Mutations are structured edits to rule lists and a few numeric knobs —
the mutator never emits Python source. Out-of-range indices, wrong types,
and unknown keys are silently dropped so a malformed mutator response can
never crash the loop.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from autodata.harness import HarnessSpec, make_harness
from autodata.llm import LLMClient, LLMRequest
from autodata.utils import stable_id

_RULE_ROLES = ("challenger", "quality", "judge", "weak_solver", "strong_solver", "reflector")


class Mutator:
    SYSTEM = (
        "ROLE:META_MUTATOR. You optimize an agentic synthetic-data harness by editing its rule set.\n"
        "Given the current HarnessSpec (as JSON) and a failure summary from the previous evaluation, propose ONE\n"
        "small mutation that addresses the most common failure mode. Mutations are structured edits to rule lists\n"
        "and structural knobs — you do NOT emit code.\n"
        "Return STRICT JSON with these keys (every key optional except `rationale`):\n"
        "{\n"
        '  "rationale": "one or two sentences",\n'
        '  "challenger_rules_add": [strings],\n'
        '  "challenger_rules_remove_indices": [ints],\n'
        '  "quality_rules_add": [strings],\n'
        '  "quality_rules_remove_indices": [ints],\n'
        '  "judge_rules_add": [strings],\n'
        '  "judge_rules_remove_indices": [ints],\n'
        '  "weak_solver_rules_add": [strings],\n'
        '  "weak_solver_rules_remove_indices": [ints],\n'
        '  "strong_solver_rules_add": [strings],\n'
        '  "strong_solver_rules_remove_indices": [ints],\n'
        '  "reflector_rules_add": [strings],\n'
        '  "reflector_rules_remove_indices": [ints],\n'
        '  "rubric_max_weight": int 1..10,\n'
        '  "require_self_test": bool\n'
        "}\n"
        "Add NEW rules, not paraphrases of existing ones. Keep rule strings imperative and specific."
    )

    def __init__(self, llm: LLMClient, model_key: str):
        self.llm = llm
        self.model_key = model_key

    def propose(self, parent: HarnessSpec, failure_summary: dict[str, Any]) -> dict[str, Any]:
        user = json.dumps(
            {
                "current_harness": parent.model_dump(
                    exclude={"harness_id", "parent_id", "iteration", "train_score", "val_score"}
                ),
                "failure_summary": failure_summary,
            },
            indent=2,
        )
        request_id = stable_id("metaopt-mutator", parent.harness_id,
                               parent.iteration, json.dumps(failure_summary)[:200])
        request = LLMRequest(
            request_id=request_id,
            item_id="metaopt",
            round_n=parent.iteration,
            role="meta_mutator",
            model_key=self.model_key,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user},
            ],
            json_mode=True,
        )
        try:
            resp = self.llm.complete(request)
            return resp.parse_json()
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("mutator parse failure: {}", e)
            return {"rationale": "mutator parse error; no mutation applied"}
        except Exception as e:
            logger.warning("mutator call failed: {}", e)
            return {"rationale": f"mutator error: {e}; no mutation applied"}


def apply_mutation(parent: HarnessSpec, mutation: dict[str, Any], *, iteration: int) -> HarnessSpec:
    """Apply an add/remove diff and produce a child HarnessSpec."""
    fields = parent.model_dump()

    for role in _RULE_ROLES:
        key = f"{role}_rules"
        rules: list[str] = list(fields.get(key, []) or [])

        rm = mutation.get(f"{role}_rules_remove_indices") or []
        if isinstance(rm, list):
            rm_set = {int(i) for i in rm if isinstance(i, int) and 0 <= int(i) < len(rules)}
            rules = [r for i, r in enumerate(rules) if i not in rm_set]

        add = mutation.get(f"{role}_rules_add") or []
        if isinstance(add, list):
            for s in add:
                if isinstance(s, str) and s.strip():
                    rules.append(s.strip())

        fields[key] = rules

    rmw = mutation.get("rubric_max_weight")
    if isinstance(rmw, int) and 1 <= rmw <= 10:
        fields["rubric_max_weight"] = rmw

    rst = mutation.get("require_self_test")
    if isinstance(rst, bool):
        fields["require_self_test"] = rst

    rationale = str(mutation.get("rationale") or "").strip()

    # Strip non-mutable / inherited fields before re-constructing.
    for k in ("harness_id", "parent_id", "iteration", "rationale", "train_score", "val_score"):
        fields.pop(k, None)

    return make_harness(
        parent_id=parent.harness_id,
        iteration=iteration,
        rationale=rationale or "no rationale",
        **fields,
    )
