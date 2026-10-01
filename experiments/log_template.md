# Experiment log

One row per run. The columns are the columns of Table 1 in `../paper/main.tex`,
so this file fills the paper in as you go. Copy the row template, never
overwrite an old row, and never log a run whose config you did not save.

Rules that will save you in week 10:

- Every row needs a **config path** and a **seed**. A number without both is
  not a result, it is a rumour.
- Log the runs that got worse. The ablation table is built out of them, and
  "we tried X and it hurt" is a sentence reviewers trust.
- If you change `src/interruptions.py`, draw a horizontal line here and start a
  new block. Changing the injection protocol invalidates every row above it.

Arrows show the preferred direction. JGA-R: post-interruption joint goal
accuracy. CIR: correction incorporation rate. SIR: stale information rate.

---

## Block 1: synthetic data, injection protocol v1

| Date | Config | Seed | Policy | Data | JGA-R ↑ | CIR ↑ | SIR ↓ | Intent ↑ | Success ↑ | Latency ms ↓ | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | `configs/default.yaml` | 13 | naive_resume | synthetic | | | | | | | baseline floor |
| YYYY-MM-DD | `configs/default.yaml` | 13 | restart_turn | synthetic | | | | | | | |
| YYYY-MM-DD | `configs/default.yaml` | 13 | full_context_llm | synthetic | | | | | | | |
| YYYY-MM-DD | `configs/default.yaml` | 13 | cascaded_dst | synthetic | | | | | | | |
| YYYY-MM-DD | `configs/default.yaml` | 13 | restate | synthetic | | | | | | | |

### Ablations (Figure 2b)

| Date | Config | Seed | Variant | Success ↑ | Delta vs full | What this tells me |
|---|---|---|---|---|---|---|
| YYYY-MM-DD | `configs/default.yaml` | 13 | full | | 0.0 | |
| YYYY-MM-DD | `configs/default.yaml` | 13 | − commitment ledger | | | is the ledger the contribution? |
| YYYY-MM-DD | `configs/default.yaml` | 13 | − interruption typology | | | does event typing matter? |
| YYYY-MM-DD | `configs/default.yaml` | 13 | − resumption planner | | | what does re-grounding cost? |
| YYYY-MM-DD | `configs/default.yaml` | 13 | − state edit operator | | | is explicit editing needed? |

### Per interruption type (Figure 2a)

Fill one of these per policy you care about.

| Policy | Backchannel | Barge-in stop | Self-correction | Topic switch | Post-pause resume |
|---|---|---|---|---|---|
| restate | | | | | |

---

## Row template

```
| 2026-10-14 | configs/exp-onset-sweep.yaml | 13 | restate | multiwoz | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | onset 0.2 only |
```

## Running notes

Keep a dated paragraph here whenever something surprises you. Surprises are
where the paper's discussion section comes from, and you will not remember them
in December.

**YYYY-MM-DD.** _What I expected, what happened, what I will try next._
