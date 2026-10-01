"""Evaluation entry point.

    python -m src.eval --config configs/default.yaml
    python -m src.eval --config configs/default.yaml --all-policies
    python -m src.eval --config configs/default.yaml --ablations

Writes a JSON report whose structure maps onto Table 1 and Figure 2 of the
paper draft, so that src/plot_results.py can regenerate the figure directly.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import yaml

from .agents import POLICIES, build_policy
from .data import load_dialogues
from .interruptions import build_suite
from .metrics import evaluate

ABLATIONS = {
    "full": {},
    "no_commitment_ledger": {"use_commitment_ledger": False},
    "no_typology": {"use_typology": False},
    "no_resumption_planner": {"use_resumption_planner": False},
    "no_state_edit_operator": {"use_state_edit_operator": False},
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_one(cfg: dict, policy_name: str) -> dict[str, Any]:
    dialogues = load_dialogues(cfg)
    cases = build_suite(dialogues, cfg)
    policy = build_policy(cfg, policy_name)
    results = [policy.recover(case) for case in cases]
    report = evaluate(cases, results, cfg)
    report["policy"] = policy_name
    return report


def _fmt(report: dict[str, Any]) -> str:
    o = report["overall"]
    def g(k: str) -> str:
        return f"{o[k]['mean']:5.1f}" if k in o else "  n/a"
    return (f"{report['policy']:<22} "
            f"JGA-R {g('jga_post')}  CIR {g('cir')}  SIR {g('sir')}  "
            f"Intent {g('intent_preservation')}  Success {g('task_success')}  "
            f"Lat {o['recovery_latency_ms']['mean']:6.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", default=None,
                    help="override agent.policy for a single run")
    ap.add_argument("--all-policies", action="store_true",
                    help="run every baseline plus ReState (fills Table 1)")
    ap.add_argument("--ablations", action="store_true",
                    help="run ReState with each component removed (Figure 2b)")
    ap.add_argument("--out", default=None, help="override eval.report_path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = args.out or cfg["eval"]["report_path"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    report: dict[str, Any] = {"run": cfg["run"], "data": cfg["data"]}

    if args.all_policies:
        report["policies"] = {}
        for name in POLICIES:
            r = run_one(cfg, name)
            report["policies"][name] = r
            print(_fmt(r))
    elif args.ablations:
        report["ablations"] = {}
        for name, overrides in ABLATIONS.items():
            variant = json.loads(json.dumps(cfg))
            variant["restate"].update(overrides)
            r = run_one(variant, "restate")
            r["policy"] = f"restate/{name}"
            report["ablations"][name] = r
            print(_fmt(r))
    else:
        r = run_one(cfg, args.policy)
        report["policies"] = {r["policy"]: r}
        print(_fmt(r))

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
