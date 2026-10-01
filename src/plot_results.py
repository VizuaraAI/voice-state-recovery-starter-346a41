"""Regenerate the paper's results figure from a real report.

    python -m src.eval --config configs/default.yaml --all-policies --out runs/main.json
    python -m src.eval --config configs/default.yaml --ablations    --out runs/abl.json
    python -m src.plot_results --main runs/main.json --ablations runs/abl.json \
        --out ../paper/figures/fig2_results.png

Once this produces a figure from your own numbers, delete the illustrative
placeholder figure from the paper directory and stop describing the values as
illustrative in the caption.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

ORDER = ["naive_resume", "restart_turn", "full_context_llm", "cascaded_dst", "restate"]
PRETTY = {"naive_resume": "Naive resume", "restart_turn": "Restart turn",
          "full_context_llm": "Full-context LLM", "cascaded_dst": "Cascaded DST",
          "restate": "ReState (ours)"}
COLORS = ["#b9bec7", "#8fb4d9", "#5fa8a0", "#e0ad5c", "#23456e"]
EVENTS = ["backchannel", "barge_in_stop", "self_correction", "topic_switch",
          "post_pause_resume"]
EVENT_LABELS = ["Backchannel", "Barge-in\nstop", "Self-\ncorrection",
                "Topic\nswitch", "Post-pause\nresume"]
ABL_ORDER = ["full", "no_commitment_ledger", "no_typology",
             "no_resumption_planner", "no_state_edit_operator"]
ABL_LABELS = ["Full ReState", "− commitment ledger", "− interruption typology",
              "− resumption planner", "− state edit operator"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", required=True)
    ap.add_argument("--ablations", required=True)
    ap.add_argument("--out", default="fig2_results.png")
    args = ap.parse_args()

    with open(args.main, encoding="utf-8") as fh:
        main_report = json.load(fh)["policies"]
    with open(args.ablations, encoding="utf-8") as fh:
        abl_report = json.load(fh)["ablations"]

    jga = np.array([[main_report[p]["by_event_type"].get(e, {})
                     .get("jga_post", {}).get("mean", 0.0)
                     for e in EVENTS] for p in ORDER])
    err = np.array([[_halfwidth(main_report[p]["by_event_type"].get(e, {})
                                .get("jga_post", {}))
                     for e in EVENTS] for p in ORDER])
    abl = np.array([abl_report[k]["overall"]["task_success"]["mean"]
                    for k in ABL_ORDER])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.edgecolor": "#5c6372", "axes.linewidth": 0.7,
                         "savefig.dpi": 400})
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.1, 2.85),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.30})

    x, w = np.arange(len(EVENTS)), 0.155
    for i, p in enumerate(ORDER):
        off = (i - (len(ORDER) - 1) / 2) * w
        axa.bar(x + off, jga[i], width=w, color=COLORS[i], label=PRETTY[p],
                edgecolor="#17304f" if p == "restate" else "none",
                linewidth=0.7 if p == "restate" else 0, zorder=3)
        axa.errorbar(x + off, jga[i], yerr=err[i], fmt="none", ecolor="#3d4350",
                     elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=4)
    axa.set_xticks(x)
    axa.set_xticklabels(EVENT_LABELS, fontsize=7.0)
    axa.set_ylim(0, 100)
    axa.set_ylabel("Post-interruption Joint Goal Accuracy (%)", fontsize=7.4)
    axa.yaxis.grid(True, color="#d8dbe0", linewidth=0.6, zorder=0)
    axa.set_axisbelow(True)
    axa.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.set_title("(a) Recovery quality by interruption type", fontsize=8.0,
                  pad=22, loc="left")
    axa.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=5,
               frameon=False, fontsize=6.4, handlelength=1.0, handleheight=0.8,
               columnspacing=1.0, handletextpad=0.45)

    y = np.arange(len(abl))[::-1]
    axb.barh(y, abl, height=0.56, zorder=3, edgecolor="#17304f", linewidth=0.5,
             color=["#23456e"] + ["#e0ad5c"] * (len(abl) - 1))
    axb.axvline(abl[0], color="#23456e", linestyle=(0, (3, 2)), linewidth=0.8)
    for yi, v in zip(y, abl):
        axb.text(v + 1.0, yi, f"{v:.0f}", va="center", fontsize=7.0)
    axb.set_yticks(y)
    axb.set_yticklabels(ABL_LABELS, fontsize=7.0)
    axb.set_xlim(0, max(100.0, float(abl.max()) + 10))
    axb.set_xlabel("Task Success Rate (%)", fontsize=7.4)
    axb.xaxis.grid(True, color="#d8dbe0", linewidth=0.6, zorder=0)
    axb.set_axisbelow(True)
    axb.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.set_title("(b) Ablation: components removed", fontsize=8.0, pad=22,
                  loc="left")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.145)
    fig.savefig(args.out, facecolor="white")
    print(f"wrote {args.out}")


def _halfwidth(entry: dict) -> float:
    ci = entry.get("ci95")
    if not ci:
        return 0.0
    return max(0.0, (ci[1] - ci[0]) / 2.0)


if __name__ == "__main__":
    main()
