"""The interruption injection protocol.

This is the measurement instrument of the whole project, so it deserves more
care than any other file. The idea: take a dialogue whose state is already
annotated, cut the agent's turn at a known point, splice in a user utterance of
a known type, and derive the correct post-recovery state deterministically.
Because the ground truth is constructed rather than labelled, every injected
example is scorable without human annotation.

The cost of that convenience is realism. Validate on real recorded
interruptions before you claim the conclusions transfer (see README, Week 7).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .data import Dialogue

EVENT_TYPES = [
    "backchannel",
    "barge_in_stop",
    "self_correction",
    "topic_switch",
    "post_pause_resume",
]

BACKCHANNELS = ["mhm", "right", "okay", "yeah", "uh huh"]
STOPS = ["wait", "hold on", "stop", "no no", "hang on"]
PAUSE_RESUMES = ["sorry, where were we", "okay, go on", "yes, carry on"]

# Plausible replacement values per slot, used to build self-corrections.
CORRECTIONS: dict[str, list[str]] = {
    "party_size": ["two", "three", "five", "six", "eight"],
    "time": ["five", "six", "seven", "eight thirty", "nine"],
    "area": ["centre", "north", "south", "east", "west"],
    "price_range": ["cheap", "moderate", "expensive"],
    "date": ["Monday", "Thursday", "Saturday"],
    "destination": ["the airport", "the hotel", "the station"],
    "departure": ["the station", "the museum", "my house"],
}

TOPIC_SWITCHES = [
    ("hotel", {"domain": "hotel", "price_range": "cheap"},
     "actually, forget that, I need a cheap hotel instead"),
    ("taxi", {"domain": "taxi", "destination": "the airport"},
     "actually, can you book me a taxi to the airport instead"),
    ("train", {"domain": "train", "destination": "London"},
     "actually, I need a train to London instead"),
]


@dataclass
class InterruptionCase:
    """One scorable test item."""

    case_id: str
    dialogue: Dialogue
    event_type: str
    onset_fraction: float
    user_utterance: str
    payload: dict[str, Any] = field(default_factory=dict)
    gold_post_state: dict[str, Any] = field(default_factory=dict)
    # Values the user heard the agent say before the cut. Resuming with any of
    # these again counts as stale information.
    heard_values: dict[str, Any] = field(default_factory=dict)
    # True when the correct behaviour is to leave state untouched.
    expect_no_change: bool = False


def _heard(dialogue: Dialogue, onset: float) -> dict[str, Any]:
    clauses = dialogue.agent_turn.clauses
    cut = max(0, min(len(clauses), round(onset * len(clauses))))
    spoken = " ".join(clauses[:cut]).lower()
    return {k: v for k, v in dialogue.agent_turn.mentions.items()
            if str(v).lower() in spoken}


def build_case(dialogue: Dialogue, event_type: str, onset: float,
               rng: random.Random, index: int) -> InterruptionCase | None:
    gold = dict(dialogue.gold_state)
    heard = _heard(dialogue, onset)
    case_id = f"{dialogue.dialogue_id}|{event_type}|{onset:.1f}"

    if event_type == "backchannel":
        return InterruptionCase(case_id, dialogue, event_type, onset,
                                rng.choice(BACKCHANNELS), {}, gold, heard,
                                expect_no_change=True)

    if event_type == "barge_in_stop":
        return InterruptionCase(case_id, dialogue, event_type, onset,
                                rng.choice(STOPS), {}, gold, heard,
                                expect_no_change=True)

    if event_type == "self_correction":
        candidates = [s for s in gold if s in CORRECTIONS and s != "domain"]
        if not candidates:
            return None
        slot = rng.choice(candidates)
        options = [v for v in CORRECTIONS[slot] if v != gold[slot]]
        if not options:
            return None
        new_value = rng.choice(options)
        post = dict(gold)
        post[slot] = new_value
        text = rng.choice([
            f"no, make it {new_value}",
            f"sorry, I meant {new_value}",
            f"actually {new_value}, not {gold[slot]}",
        ])
        return InterruptionCase(case_id, dialogue, event_type, onset, text,
                                {"slots": {slot: new_value}}, post, heard)

    if event_type == "topic_switch":
        _, new_slots, text = TOPIC_SWITCHES[index % len(TOPIC_SWITCHES)]
        keep = {k: v for k, v in gold.items() if k in {"name", "phone", "language"}}
        post = {**keep, **new_slots}
        return InterruptionCase(case_id, dialogue, event_type, onset, text,
                                {"slots": dict(new_slots)}, post, heard)

    if event_type == "post_pause_resume":
        return InterruptionCase(case_id, dialogue, event_type, onset,
                                rng.choice(PAUSE_RESUMES), {}, gold, heard,
                                expect_no_change=True)

    raise ValueError(f"unknown event type: {event_type!r}")


def build_suite(dialogues: list[Dialogue], cfg: dict) -> list[InterruptionCase]:
    icfg = cfg["interruptions"]
    rng = random.Random(cfg["run"]["seed"] + 1)
    per_type = int(icfg["per_type_count"])
    onsets = [float(f) for f in icfg["onset_fractions"]]

    cases: list[InterruptionCase] = []
    for event_type in icfg["types"]:
        made, i = 0, 0
        while made < per_type and i < per_type * 20:
            dialogue = dialogues[i % len(dialogues)]
            onset = onsets[i % len(onsets)]
            case = build_case(dialogue, event_type, onset, rng, i)
            i += 1
            if case is None:
                continue
            cases.append(case)
            made += 1
    rng.shuffle(cases)
    return cases
