# voice-state-recovery-starter

Starter code for **ReState: Explicit Conversational State Recovery for
Streaming Voice Agents After Interruptions and Corrections**.

Kunal Saurabh, Vizuara Voice Agents Bootcamp.

**The research question.** When a real-time voice conversation is interrupted
or corrected, how should a voice agent recover and update its conversational
state to continue from the user's actual intent rather than simply resuming the
previous turn?

This repository is the smallest thing that can answer that question. It runs
end to end on a laptop in under a minute with no API key, no GPU and no
downloads, and every number it prints corresponds to a cell in Table 1 of
`../paper/main.tex`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The speech packages (`faster-whisper`, `librosa`, `soundfile`) are not needed
until Week 3. If install is slow, comment them out and come back to them.

## The first three things to run

**1. Reproduce the full baseline table (about 20 seconds).**

```bash
python -m src.eval --config configs/default.yaml --all-policies --out runs/main.json
```

You should see five rows, one per policy, with ReState ahead on joint goal
accuracy and correction incorporation. Those rows are Table 1.

**2. Run the component ablations.**

```bash
python -m src.eval --config configs/default.yaml --ablations --out runs/abl.json
```

This is Figure 2(b). Read the output carefully: **in the starter
implementation, removing the resumption planner currently *improves* task
success.** That is not a bug in the harness, it is a real weakness in the
starting policy, and fixing it is your first piece of research (see "Known
weaknesses" below).

**3. Regenerate the paper figure from your own numbers.**

```bash
python -m src.plot_results --main runs/main.json --ablations runs/abl.json \
  --out ../paper/figures/fig2_results.png
```

The moment this works, the figure in your paper is yours rather than an
illustrative placeholder, and you can delete that word from the caption.

Optionally, train the learned interruption classifier:

```bash
python -m src.train --config configs/default.yaml
# then set restate.typology_model: learned in your config
```

## Layout

```
configs/default.yaml      every knob; copy it per experiment, do not edit in place
src/state.py              BeliefFrame, CommitmentLedger, PendingActionQueue,
                          StateEditOperator, ResumptionPlanner  (Figure 1)
src/data.py               synthetic / MultiWOZ / SpokenWOZ loading
src/interruptions.py      the injection protocol: the measurement instrument
src/typology.py           rule-based and learned event classifiers
src/agents.py             the five policies compared in Table 1
src/metrics.py            JGA-R, CIR, SIR, intent, task success, latency
src/eval.py               evaluation entry point
src/train.py              trains the typology classifier
src/plot_results.py       regenerates Figure 2 from a report
scripts/                  one-line wrappers for the commands above
experiments/log_template.md   one row per run, columns match Table 1
```

## The idea in one paragraph

A voice agent that gets interrupted has a problem no transcript can express:
it generated a sentence, but the loudspeaker was cut off partway through, so
the agent's record of the conversation and the user's memory of it disagree.
ReState keeps three stores instead of one. The **belief frame** holds slot
values with provenance. The **commitment ledger** holds only the clauses that
were actually played out of the speaker before the barge-in. The **pending
action queue** holds tool calls tagged with the slots they depend on, so a
correction can invalidate exactly the right ones. An interruption is classified
into one of five types, an explicit edit operation (`RETAIN`, `DISCARD`,
`REVISE`, `RE-ASK`) rewrites state, and a resumption planner decides whether to
continue, re-ground, or restart.

## Known weaknesses (your starting research agenda)

These are real and deliberate. Finding them is cheaper than being told, but
being told is cheaper than losing a week.

1. **The re-grounding path leaks stale values.** When ReState re-grounds after
   a correction, it prepends a confirming clause but still replays the unspoken
   remainder, which can contain the *old* value. This is why the no-planner
   ablation scores better on task success. Fix: rewrite the remainder against
   the edited belief frame instead of replaying it verbatim.
2. **Synthetic dialogues make the typology classifier look perfect.** It scores
   1.00 because the templates are fixed. Move to MultiWOZ, then SpokenWOZ, and
   expect the number to drop hard.
3. **`intent_preserved` is a proxy, not a judge.** Replace it with a real
   LLM judge in Week 6 and report agreement with your own labels on a sample.
4. **Interruptions are injected, not recorded.** Record 30 real interruptions
   with yourself and a friend before you claim the results transfer.
5. **The latency numbers are a simulated budget**, read from
   `configs/default.yaml`, not measured. Replace them with wall-clock timings
   once there is real audio in the loop.

## Datasets

- **synthetic**: built into `src/data.py`. No download.
- **MultiWOZ 2.2**: `source: multiwoz`, downloads from the Hugging Face Hub on
  first use.
- **SpokenWOZ**: `source: spokenwoz`, needs a manual download. Run
  `scripts/download_data.sh` and follow what it prints.

No data is committed to this repository.

## Working with a coding agent

Read `CLAUDE.md` first, then point your agent at it. It contains the project
goal, the conventions, and a ranked list of next steps.
