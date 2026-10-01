"""Dialogue loading.

Three sources, in the order you should adopt them:

1. ``synthetic``  built in this file, no downloads. Use it on day one so the
   whole pipeline runs in seconds while you are still changing the interfaces.
2. ``multiwoz``   downloaded from the Hugging Face Hub. Clean belief-state
   annotation, which is what makes injected interruptions scorable.
3. ``spokenwoz``  real human-human speech with natural disfluency. Needs a
   manual download; see scripts/download_data.sh.

No data files are committed to this repository. Everything downloads on demand
into ``data/``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterator

from .state import Utterance

# Slots we track. Keep this list short while you iterate; a smaller schema makes
# joint goal accuracy readable and the failures easier to eyeball.
SCHEMA = [
    "domain", "party_size", "time", "date", "area", "price_range", "name",
    "phone", "destination", "departure",
]


@dataclass
class Dialogue:
    """One dialogue up to and including the turn that gets interrupted."""

    dialogue_id: str
    history: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    gold_state: dict[str, Any] = field(default_factory=dict)
    agent_turn: Utterance = field(default_factory=lambda: Utterance(""))
    domain: str = "restaurant"

    def transcript(self) -> str:
        return "\n".join(f"{spk}: {txt}" for spk, txt in self.history)


# --------------------------------------------------------------------------
# Source 1: synthetic
# --------------------------------------------------------------------------

_TEMPLATES = [
    {
        "domain": "restaurant",
        "user": "I need a table for {party_size} at {time} in the {area} of town.",
        "agent": "Booking a table for {party_size}, at {time}, in the {area}, "
                 "and I will confirm the reservation now",
        "slots": {"party_size": ["two", "four", "six", "eight"],
                  "time": ["six", "seven", "eight thirty"],
                  "area": ["centre", "north", "south"]},
    },
    {
        "domain": "hotel",
        "user": "Find me a {price_range} hotel in the {area} for {date}.",
        "agent": "Searching {price_range} hotels, in the {area}, "
                 "for {date}, and I have three options to read out",
        "slots": {"price_range": ["cheap", "moderate", "expensive"],
                  "area": ["centre", "west"],
                  "date": ["Tuesday", "Friday", "the weekend"]},
    },
    {
        "domain": "taxi",
        "user": "I want a taxi from {departure} to {destination} at {time}.",
        "agent": "Booking a taxi, from {departure}, to {destination}, "
                 "leaving at {time}, and the fare is fixed",
        "slots": {"departure": ["the station", "the museum"],
                  "destination": ["the airport", "the hotel"],
                  "time": ["nine", "noon", "four"]},
    },
]


def _synthetic(n: int, rng: random.Random) -> Iterator[Dialogue]:
    for i in range(n):
        tpl = _TEMPLATES[i % len(_TEMPLATES)]
        values = {k: rng.choice(v) for k, v in tpl["slots"].items()}
        user_text = tpl["user"].format(**values)
        agent_text = tpl["agent"].format(**values)
        gold = {"domain": tpl["domain"], **values}
        yield Dialogue(
            dialogue_id=f"syn-{i:04d}",
            history=[("U", user_text)],
            gold_state=gold,
            agent_turn=Utterance(agent_text, mentions=dict(values)),
            domain=tpl["domain"],
        )


# --------------------------------------------------------------------------
# Source 2: MultiWOZ via the Hugging Face Hub
# --------------------------------------------------------------------------

def _multiwoz(n: int, split: str) -> Iterator[Dialogue]:
    from datasets import load_dataset  # imported lazily: not needed for synthetic

    ds = load_dataset("multi_woz_v22", split=split, trust_remote_code=True)
    produced = 0
    for row in ds:
        turns = row["turns"]
        speakers, texts = turns["speaker"], turns["utterance"]
        # Cut after the first system turn that follows at least one user turn.
        cut = next((i for i in range(1, len(speakers)) if speakers[i] == 1), None)
        if cut is None:
            continue
        history = [("U" if s == 0 else "A", t)
                   for s, t in zip(speakers[:cut], texts[:cut])]
        gold = _extract_multiwoz_state(turns, cut)
        if not gold:
            continue
        agent_text = texts[cut]
        yield Dialogue(
            dialogue_id=row["dialogue_id"],
            history=history,
            gold_state=gold,
            agent_turn=Utterance(agent_text, mentions=_mentions(agent_text, gold)),
            domain=gold.get("domain", "unknown"),
        )
        produced += 1
        if produced >= n:
            return


def _extract_multiwoz_state(turns: dict, upto: int) -> dict[str, Any]:
    """Flatten MultiWOZ 2.2 frame annotation into our flat schema."""
    state: dict[str, Any] = {}
    frames = turns.get("frames")
    if frames is None:
        return state
    for i in range(min(upto, len(frames))):
        frame = frames[i]
        for service, st in zip(frame.get("service", []), frame.get("state", [])):
            slots = st.get("slot_values", {})
            for key, values in zip(slots.get("slot_name", []),
                                   slots.get("slot_value_list", [])):
                short = key.split("-")[-1].replace(" ", "_")
                if short in SCHEMA and values:
                    state[short] = values[0]
            if service:
                state.setdefault("domain", service)
    return state


def _mentions(text: str, gold: dict[str, Any]) -> dict[str, Any]:
    low = text.lower()
    return {k: v for k, v in gold.items() if str(v).lower() in low}


# --------------------------------------------------------------------------
# Source 3: SpokenWOZ (manual download)
# --------------------------------------------------------------------------

def _spokenwoz(n: int, root: str) -> Iterator[Dialogue]:
    import json
    import os

    path = os.path.join(root, "text_5700_train_dev", "data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SpokenWOZ not found at {path}. Run scripts/download_data.sh and "
            "follow the instructions it prints."
        )
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    for i, (did, dial) in enumerate(raw.items()):
        if i >= n:
            return
        logs = dial.get("log", [])
        cut = 1 if len(logs) > 1 else None
        if cut is None:
            continue
        gold = {}
        for dom, book in dial.get("goal", {}).items():
            if isinstance(book, dict) and book:
                gold["domain"] = dom
                break
        history = [("U", logs[0].get("text", ""))]
        agent_text = logs[1].get("text", "")
        yield Dialogue(did, history, gold,
                       Utterance(agent_text, mentions=_mentions(agent_text, gold)),
                       gold.get("domain", "unknown"))


# --------------------------------------------------------------------------

def load_dialogues(cfg: dict) -> list[Dialogue]:
    data = cfg["data"]
    source = data["source"]
    n = int(data["max_dialogues"])
    rng = random.Random(cfg["run"]["seed"])

    if source == "synthetic":
        return list(_synthetic(n, rng))
    if source == "multiwoz":
        return list(_multiwoz(n, data.get("split", "validation")))
    if source == "spokenwoz":
        return list(_spokenwoz(n, data.get("spokenwoz_dir", "data/spokenwoz")))
    raise ValueError(f"unknown data source: {source!r}")
