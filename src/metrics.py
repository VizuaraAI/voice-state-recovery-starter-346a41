"""The metric suite.

Each metric corresponds to one column of Table 1 in the paper draft and to one
distinct way recovery can fail. Keep that one-to-one mapping: if you add a
metric here, add a column there, and add a row to
experiments/log_template.md.

* ``jga_post``            did the agent end up with the right state?
* ``cir``                 did it actually apply the correction?
* ``sir``                 did it repeat something the user already heard?
* ``intent_preservation`` did it keep the user's goal (proxy here, LLM judge later)?
* ``task_success``        would the booking have gone through correctly?
* ``recovery_latency_ms`` how long until the user hears something useful?
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .agents import RecoveryResult
from .interruptions import InterruptionCase


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def joint_goal_accuracy(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    """Exact match over the full frame. Harsh on purpose, as in MultiWOZ."""
    if not gold:
        return 1.0 if not pred else 0.0
    keys = set(gold)
    return float(all(_norm(pred.get(k)) == _norm(gold[k]) for k in keys))


def slot_accuracy(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    if not gold:
        return 1.0
    hits = sum(_norm(pred.get(k)) == _norm(v) for k, v in gold.items())
    return hits / len(gold)


def correction_incorporated(case: InterruptionCase, result: RecoveryResult) -> float | None:
    """Only defined for cases that actually carry a correction."""
    injected = case.payload.get("slots") or {}
    if case.event_type not in {"self_correction", "topic_switch"} or not injected:
        return None
    return float(all(_norm(result.predicted_state.get(k)) == _norm(v)
                     for k, v in injected.items()))


def stale_information(case: InterruptionCase, result: RecoveryResult) -> float:
    """1.0 when the resumed turn restates something already heard or overridden."""
    if result.repeated_values:
        return 1.0
    overridden = case.payload.get("slots") or {}
    low = result.response.lower()
    for slot, new_value in overridden.items():
        old = case.dialogue.gold_state.get(slot)
        if old is not None and _norm(old) != _norm(new_value) and _norm(old) in low:
            return 1.0
    return 0.0


def intent_preserved(case: InterruptionCase, result: RecoveryResult) -> float:
    """Offline proxy for the LLM judge.

    Intent counts as preserved when the domain survives and no slot the user
    never mentioned was invented. Replace this with a real judge in Week 6; the
    proxy exists so the pipeline is runnable before you have API access.
    """
    gold = case.gold_post_state
    pred = result.predicted_state
    if gold.get("domain") and _norm(pred.get("domain")) != _norm(gold["domain"]):
        return 0.0
    invented = set(pred) - set(gold)
    return float(len(invented) == 0)


def task_success(case: InterruptionCase, result: RecoveryResult) -> float:
    """Every gold slot right AND nothing stale said back to the user."""
    frame_ok = joint_goal_accuracy(result.predicted_state, case.gold_post_state)
    return float(frame_ok == 1.0 and stale_information(case, result) == 0.0)


def _bootstrap_ci(values: list[float], n: int, seed: int) -> tuple[float, float]:
    if not values or n <= 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return (lo, hi)


def evaluate(cases: list[InterruptionCase], results: list[RecoveryResult],
             cfg: dict) -> dict[str, Any]:
    """Aggregate overall and per interruption type, with bootstrap intervals."""
    per_metric: dict[str, list[float]] = defaultdict(list)
    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for case, result in zip(cases, results):
        row = {
            "jga_post": joint_goal_accuracy(result.predicted_state,
                                            case.gold_post_state),
            "slot_acc": slot_accuracy(result.predicted_state, case.gold_post_state),
            "sir": stale_information(case, result),
            "intent_preservation": intent_preserved(case, result),
            "task_success": task_success(case, result),
            "recovery_latency_ms": result.latency_ms,
        }
        cir = correction_incorporated(case, result)
        if cir is not None:
            row["cir"] = cir
        for name, value in row.items():
            per_metric[name].append(value)
            per_type[case.event_type][name].append(value)

    n_boot = int(cfg["eval"].get("bootstrap_samples", 0))
    seed = int(cfg["run"]["seed"])

    def summarise(bucket: dict[str, list[float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, values in bucket.items():
            mean = sum(values) / len(values)
            scale = 1.0 if name == "recovery_latency_ms" else 100.0
            entry: dict[str, Any] = {"mean": round(mean * scale, 2),
                                     "n": len(values)}
            if n_boot and name != "recovery_latency_ms":
                lo, hi = _bootstrap_ci(values, n_boot, seed)
                entry["ci95"] = [round(lo * scale, 2), round(hi * scale, 2)]
            out[name] = entry
        return out

    return {
        "overall": summarise(per_metric),
        "by_event_type": {k: summarise(v) for k, v in sorted(per_type.items())},
        "n_cases": len(cases),
    }
