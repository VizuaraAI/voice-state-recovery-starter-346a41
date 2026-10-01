"""Render Figure 1, the ReState architecture diagram.

    python scripts/make_fig1.py ../paper/figures/fig1_method.png

Edit the boxes and routes below as your design changes. Keeping the figure in
code rather than in a drawing tool means it stays in sync with src/state.py.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 400})

INK = "#1f2430"
SUB = "#5c6372"
BLUE, BLUE_E = "#dce7f5", "#3a6ea5"
AMBER, AMBER_E = "#fbeedc", "#b2802f"
GREY, GREY_E = "#f4f5f7", "#8d939e"
TEAL = "#2e7d75"
RED = "#c0392b"

fig, ax = plt.subplots(figsize=(7.0, 4.35))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, fc=GREY, ec=GREY_E, lw=0.8, r=0.8, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def label(x, y, s, fs=6.8, color=INK, bold=False, style="normal", ha="center",
          z=5, family="DejaVu Sans", ls=1.2):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, zorder=z,
            weight="bold" if bold else "normal", style=style, family=family,
            linespacing=ls)


def pill(cx, y, w, s, fc="#ffffff", ec=GREY_E, fs=5.4):
    h = 2.6
    ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.3",
                                linewidth=0.55, edgecolor=ec, facecolor=fc, zorder=5))
    label(cx, y, s, fs=fs, z=6)


def arrow(p0, p1, color=INK, lw=0.9, z=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color, zorder=z,
                                 shrinkA=0.5, shrinkB=0.5))


def route(points, color, lw=1.1, dashed=True):
    """Orthogonal multi-segment connector with a single arrowhead at the end."""
    style = (0, (3.4, 2.1)) if dashed else "-"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.add_line(Line2D(xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1]], color=color,
                       linewidth=lw, linestyle=style, zorder=7,
                       solid_capstyle="butt"))
    arrow(points[-2], points[-1], color=color, lw=lw, z=8)


def band_title(cx, s):
    label(cx, 55.6, s, fs=7.2, bold=True, color="#343a47")


# ================================================================= band 1
band_title(9.0, "Streaming Perception")
label(9.0, 50.4, "User audio  (16 kHz stream)", fs=6.2, color=SUB)
t = np.linspace(0, 1, 300)
ax.plot(2.6 + 12.8 * t, 47.6 + 1.0 * np.sin(2 * np.pi * 9 * t) * np.exp(-((t - .5) ** 2) / .09),
        color=TEAL, lw=0.75, zorder=4)
arrow((9.0, 46.0), (9.0, 45.2), lw=0.8)

box(0.5, 39.2, 17.0, 6.0)
label(9.0, 41.9, "Incremental ASR", fs=7.0)
label(9.0, 40.3, "partial hypotheses", fs=5.5, color=SUB, style="italic")

box(0.5, 29.8, 17.0, 6.0)
label(9.0, 32.5, "Turn-taking predictor", fs=6.2)
label(9.0, 30.9, "overlap · barge-in · pause", fs=5.3, color=SUB, style="italic")

# ================================================================= band 2
band_title(27.2, "Event Detection")
box(20.0, 22.6, 14.5, 26.4, fc=AMBER, ec=AMBER_E, lw=1.0)
label(27.2, 46.6, "Interruption", fs=7.0, bold=True)
label(27.2, 44.8, "Typology", fs=7.0, bold=True)
label(27.2, 43.0, "Classifier", fs=7.0, bold=True)
for i, tag in enumerate(["backchannel", "barge-in / stop", "self-correction",
                         "topic switch", "post-pause resume"]):
    pill(27.2, 39.6 - i * 3.2, 13.0, tag, ec=AMBER_E, fs=5.5)

arrow((17.5, 42.2), (20.0, 40.6))
arrow((17.5, 32.8), (20.0, 34.4))

arrow((34.5, 38.0), (38.5, 38.0))
label(36.5, 39.6, r"$e_t$", fs=6.6, color="#343a47")

# ================================================================= band 3
band_title(53.7, "ReState Core")
box(38.5, 17.6, 30.5, 32.0, fc=BLUE, ec=BLUE_E, lw=1.5, r=1.1, z=2)

box(39.6, 42.4, 14.6, 5.8, fc="#ffffff", ec=BLUE_E)
label(46.9, 45.1, "Belief Frame", fs=6.8)
label(46.9, 43.5, "slots + confidence", fs=5.4, color=SUB, style="italic")

box(39.6, 34.7, 14.6, 5.8, fc="#ffffff", ec=BLUE_E, lw=1.2)
label(46.9, 37.4, "Commitment Ledger", fs=6.3)
label(46.9, 35.8, "what was SAID ALOUD", fs=5.4, color=SUB, style="italic")

box(39.6, 27.0, 14.6, 5.8, fc="#ffffff", ec=BLUE_E)
label(46.9, 29.7, "Pending Actions", fs=6.8)
label(46.9, 28.1, "tool calls not yet run", fs=5.3, color=SUB, style="italic")

box(55.6, 27.0, 12.2, 21.2, fc="#ffffff", ec=BLUE_E, lw=1.2)
label(61.7, 46.4, "State Edit", fs=7.0, bold=True)
label(61.7, 44.6, "Operator", fs=7.0, bold=True)
for i, op in enumerate(["RETAIN", "DISCARD", "REVISE", "RE-ASK"]):
    pill(61.7, 41.4 - i * 3.2, 10.0, op, fc=BLUE, ec=BLUE_E, fs=5.9)

for yy in (45.3, 37.6, 29.9):
    arrow((54.2, yy), (55.6, yy), lw=0.75, color=BLUE_E)
ax.add_patch(FancyArrowPatch((58.0, 27.0), (46.2, 26.0), arrowstyle="-|>",
                             mutation_scale=7, linewidth=1.0, color=BLUE_E,
                             connectionstyle="arc3,rad=-0.28", zorder=6))
label(53.7, 20.6, r"$s_t \rightarrow s_t'$   state rewritten in place", fs=5.9,
      color=BLUE_E, style="italic")

arrow((69.0, 38.0), (70.9, 38.0))

# ================================================================= band 4
band_title(77.6, "Resumption Planner")
box(70.9, 28.8, 13.4, 17.4, fc=AMBER, ec=AMBER_E, lw=1.0)
label(77.6, 44.3, "Resumption", fs=7.0, bold=True)
label(77.6, 42.5, "Planner", fs=7.0, bold=True)
for i, opt in enumerate(["continue remainder", "reground, continue", "restart turn"]):
    pill(77.6, 39.1 - i * 3.2, 12.3, opt, ec=AMBER_E, fs=5.3)
label(77.6, 27.1, "resume point + optional", fs=5.3, color=SUB, style="italic")
label(77.6, 25.7, "one-clause confirmation", fs=5.3, color=SUB, style="italic")

arrow((84.3, 38.6), (87.0, 42.0))

# ================================================================= band 5
band_title(93.5, "Response")
box(87.0, 40.2, 13.0, 6.0)
label(93.5, 43.9, "LLM response", fs=6.6)
label(93.5, 42.0, "generator", fs=6.6)

box(87.0, 31.2, 13.0, 6.0, ec=TEAL, lw=1.0)
label(93.5, 34.2, "Streaming TTS", fs=6.8)
arrow((93.5, 40.2), (93.5, 37.2))
arrow((93.5, 31.2), (93.5, 28.4), color=TEAL)
label(91.2, 21.6, "Agent audio", fs=6.2, color=TEAL)
ax.add_patch(Rectangle((91.6, 25.6), 1.4, 2.0, facecolor=TEAL, zorder=4))
ax.add_patch(Polygon([[93.0, 25.0], [93.0, 28.2], [94.8, 29.4], [94.8, 23.8]],
                     closed=True, facecolor=TEAL, zorder=4))

# ========================= key detail 1: halt playback (top routing lane)
# Routed through the free corridor between Event Detection and ReState Core so
# it crosses no box: a detected barge-in halts the speaker immediately.
route([(34.5, 44.0), (36.5, 44.0), (36.5, 51.6), (93.5, 51.6), (93.5, 46.2)], BLUE_E)
label(65.0, 53.2, "interrupt / halt playback", fs=6.3, color=BLUE_E, bold=True)

# ====================== key detail 2: playback cursor (bottom routing lane)
route([(97.8, 31.2), (97.8, 12.4), (36.8, 12.4), (36.8, 35.6), (39.6, 35.6)], RED)
label(68.0, 13.9, "playback cursor: only audio actually heard is committed",
      fs=6.2, color=RED, bold=True)
label(84.5, 10.6, "✂  truncate at barge-in", fs=6.0, color=RED)

# ============================================================ worked example
box(0.5, 0.6, 99.0, 8.4, fc="#fbfbfc", ec=GREY_E, r=0.7)
label(2.0, 7.9, "Worked example", fs=6.4, bold=True, ha="left")
left = [('A:  "Booking a table for four at seven…"', INK),
        ('U:  "no, make it six people"        ◀ correction', RED),
        ('A:  "Got it, six people at seven."', TEAL)]
for i, (s, c) in enumerate(left):
    label(2.0, 6.0 - i * 1.75, s, fs=5.9, color=c, ha="left",
          family="DejaVu Sans Mono")
right = [("ledger.spoken   = [“a table for four”]", INK),
         ("ledger.unspoken = [“at seven”]   → DISCARD", RED),
         ("belief.party_size: 4 → 6         → REVISE", BLUE_E)]
for i, (s, c) in enumerate(right):
    label(53.0, 6.0 - i * 1.75, s, fs=5.9, color=c, ha="left",
          family="DejaVu Sans Mono")
ax.add_line(Line2D([51.0, 51.0], [1.4, 7.0], color=GREY_E, lw=0.6, zorder=4))

fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "../paper/figures/fig1_method.png"

fig.savefig(out, facecolor="white")
print("wrote", out)
