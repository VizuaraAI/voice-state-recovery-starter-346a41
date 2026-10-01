# CLAUDE.md

Instructions for the coding agent working on this repository with Kunal.

## Project goal

Answer one question with evidence: **when a real-time voice conversation is
interrupted or corrected, how should the agent recover and update its
conversational state so it continues from the user's actual intent rather than
simply resuming the previous turn?**

The deliverable is a short paper (`../paper/main.tex`), not a product. Every
change should make a number in Table 1 more trustworthy or make a claim in the
paper more defensible. If a change does neither, it is probably not worth
making right now.

## What exists

- `src/state.py` is the contribution. `BeliefFrame`, `CommitmentLedger`,
  `PendingActionQueue`, `StateEditOperator`, `ResumptionPlanner`. The
  commitment ledger is the novel part: it records only the clauses that were
  actually played out of the speaker before the barge-in, so the agent reasons
  about what the user *heard* rather than what it *generated*.
- `src/interruptions.py` is the measurement instrument. It injects five event
  types (backchannel, barge-in stop, self-correction, topic switch, post-pause
  resume) at three onset fractions and derives the correct post-recovery state
  deterministically. Treat changes here as changes to the experiment, not to
  the code: anything you change invalidates every previously logged run.
- `src/agents.py` holds five policies behind one `recover(case)` method.
- `src/metrics.py` holds six metrics, one per column of Table 1.

## Conventions

- **One interface, many policies.** Every policy implements
  `recover(case) -> RecoveryResult`. Adding a baseline must never require
  editing `src/eval.py` or `src/metrics.py`. If it does, the abstraction is
  wrong; fix the abstraction.
- **Configs are data.** No magic numbers in `src/`. If you need a knob, add it
  to `configs/default.yaml` with a comment explaining what it trades off.
- **Never edit `configs/default.yaml` for an experiment.** Copy it to
  `configs/<experiment-name>.yaml` so every row of
  `experiments/log_template.md` points at an exact file.
- **Determinism.** Everything seeds off `run.seed`. Derived seeds are offsets
  of it (see `build_suite`, `src/train.py`). Do not introduce an unseeded RNG.
- **No em dashes in anything written for the paper.** Commas, colons,
  parentheses or semicolons instead.
- **The metrics table and the paper table are one artefact.** Adding a metric
  means adding a column to Table 1 in `../paper/main.tex` and a column to
  `experiments/log_template.md` in the same change.
- Comment the *why*, not the *what*. The reason a branch exists is research
  content; the fact that it is an `if` statement is not.

## Ranked next steps

Work top down. Each item says what to change and how you will know it worked.

1. **Fix the stale-remainder bug.** When the resumption planner re-grounds
   after a correction, `ReState.recover` still replays `ledger.unspoken`
   verbatim, which can contain the value the user just overrode. Rewrite the
   remainder against the edited belief frame. *Done when* the no-planner
   ablation no longer beats the full model on task success.
2. **Move off synthetic data.** Set `data.source: multiwoz` and make the whole
   suite run. Expect joint goal accuracy to fall a long way; that fall is the
   real baseline. *Done when* `--all-policies` completes on 200 MultiWOZ
   dialogues and the ordering of policies is unchanged.
3. **Make the typology classifier honest.** It scores 1.00 on synthetic
   templates because the templates are fixed. Train on MultiWOZ-derived
   injections, report per-class F1, and always report the oracle-event-type
   condition next to the predicted one so the reader can separate classifier
   error from policy error.
4. **Replace the intent proxy with a real judge.** `metrics.intent_preserved`
   is a stand-in. Implement an LLM judge (set `agent.backend` and
   `agent.model`), hand-label 50 cases yourself, and report agreement. Do not
   report judge scores without that agreement number.
5. **Add the onset sweep.** Vary `interruptions.onset_fractions` from 0.1 to
   0.9 and plot joint goal accuracy against onset. This is the single most
   direct test of the commitment-ledger hypothesis: if the curve is flat, the
   ledger is not doing what the paper claims.
6. **Put real audio in the loop.** Streaming TTS for the agent turn, truncation
   at the real playback cursor, `faster-whisper` for incremental recognition.
   Replace the simulated latency budget with measured wall-clock timings.
7. **Add SpokenWOZ.** Real disfluency and real cross-turn slots. Expect
   everything to get worse and the error analysis to get more interesting.
8. **Run the external benchmarks.** Full-Duplex-Bench and EchoChain, to show
   the gains are not artefacts of your own injection protocol.

## Things to push back on

If Kunal asks for any of these, say so before doing it:

- Adding a sixth interruption type before the five existing ones are solid.
- Switching to a bigger model to improve a number. The claim is about the
  recovery mechanism; a model swap confounds it.
- Deleting a baseline because it looks weak. Weak baselines that are honestly
  reported are what make the strong result credible.
- Reporting any number without a seed and a config path next to it.

## Paper mapping

| Code | Paper |
|---|---|
| `src/state.py` | Section 3.1, 3.3, Figure 1 |
| `src/typology.py` | Section 3.2 |
| `src/interruptions.py` | Section 3.4, injection protocol |
| `src/agents.py` | Section 3.5, baselines |
| `src/metrics.py`, `src/eval.py` | Section 4, Table 1 |
| `src/plot_results.py` | Figure 2 |
