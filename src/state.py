"""The conversational state representation and the edit operations over it.

Three stores, mirroring Figure 1 of the paper draft:

* ``BeliefFrame``       what the agent believes the user wants (slot -> value).
* ``CommitmentLedger``  what the agent has actually said *aloud*, i.e. what the
                        user could plausibly have heard. This is the piece most
                        voice stacks get wrong: text was generated, but the
                        audio was cut off mid-sentence, so the agent and the
                        user now disagree about what was said.
* ``PendingActionQueue`` tool calls that were planned but not yet executed.

``StateEditOperator`` applies one of four operations (RETAIN, DISCARD, REVISE,
RE_ASK) to those stores given a detected interruption event.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EditOp(str, Enum):
    RETAIN = "RETAIN"
    DISCARD = "DISCARD"
    REVISE = "REVISE"
    RE_ASK = "RE_ASK"


class ResumePlan(str, Enum):
    CONTINUE = "continue_remainder"
    REGROUND = "reground_then_continue"
    RESTART = "restart_turn"


@dataclass
class SlotValue:
    """One slot with the provenance we need in order to revise it later."""

    value: Any
    turn_index: int
    confidence: float = 1.0
    # True once the agent has verbalised this value and the user heard it.
    acknowledged: bool = False

    def copy(self) -> "SlotValue":
        return SlotValue(self.value, self.turn_index, self.confidence, self.acknowledged)


@dataclass
class BeliefFrame:
    slots: dict[str, SlotValue] = field(default_factory=dict)

    def set(self, slot: str, value: Any, turn_index: int, confidence: float = 1.0) -> None:
        self.slots[slot] = SlotValue(value, turn_index, confidence)

    def get(self, slot: str) -> Any:
        sv = self.slots.get(slot)
        return None if sv is None else sv.value

    def drop(self, slot: str) -> None:
        self.slots.pop(slot, None)

    def as_dict(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.slots.items()}

    def copy(self) -> "BeliefFrame":
        return BeliefFrame({k: v.copy() for k, v in self.slots.items()})


@dataclass
class Utterance:
    """A unit of agent speech, small enough to be individually committed.

    ``clauses`` are the resumption granularity: the planner can resume from a
    clause boundary instead of restarting the whole turn.
    """

    text: str
    clauses: list[str] = field(default_factory=list)
    mentions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.clauses:
            self.clauses = [c.strip() for c in self.text.split(",") if c.strip()]


@dataclass
class CommitmentLedger:
    """Tracks the playback cursor and therefore what the user actually heard."""

    spoken: list[str] = field(default_factory=list)
    unspoken: list[str] = field(default_factory=list)
    heard_mentions: dict[str, Any] = field(default_factory=dict)
    truncated_at: float | None = None

    def commit_playback(self, utterance: Utterance, onset_fraction: float) -> None:
        """Split an utterance at the point the user barged in.

        ``onset_fraction`` is how much of the utterance had been played out of
        the speaker when the interruption landed. Everything before the cursor
        is committed (the user heard it); everything after is discarded, because
        the agent must not assume the user knows it.
        """
        n = len(utterance.clauses)
        cut = max(0, min(n, round(onset_fraction * n)))
        self.spoken.extend(utterance.clauses[:cut])
        self.unspoken.extend(utterance.clauses[cut:])
        self.truncated_at = onset_fraction
        heard_text = " ".join(utterance.clauses[:cut]).lower()
        for slot, value in utterance.mentions.items():
            if str(value).lower() in heard_text:
                self.heard_mentions[slot] = value

    def was_heard(self, slot: str) -> bool:
        return slot in self.heard_mentions

    def copy(self) -> "CommitmentLedger":
        return copy.deepcopy(self)


@dataclass
class PendingActionQueue:
    actions: list[dict[str, Any]] = field(default_factory=list)

    def push(self, name: str, **kwargs: Any) -> None:
        self.actions.append({"name": name, "args": kwargs})

    def invalidate(self, slots: set[str]) -> list[dict[str, Any]]:
        """Drop queued actions whose arguments depend on a slot that changed."""
        kept, dropped = [], []
        for action in self.actions:
            if slots & set(action["args"]):
                dropped.append(action)
            else:
                kept.append(action)
        self.actions = kept
        return dropped

    def copy(self) -> "PendingActionQueue":
        return copy.deepcopy(self)


@dataclass
class ConversationState:
    belief: BeliefFrame = field(default_factory=BeliefFrame)
    ledger: CommitmentLedger = field(default_factory=CommitmentLedger)
    queue: PendingActionQueue = field(default_factory=PendingActionQueue)
    turn_index: int = 0

    def copy(self) -> "ConversationState":
        return ConversationState(
            self.belief.copy(), self.ledger.copy(), self.queue.copy(), self.turn_index
        )


@dataclass
class EditDecision:
    op: EditOp
    slots: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    dropped_actions: list[dict[str, Any]] = field(default_factory=list)


class StateEditOperator:
    """Maps (event type, user payload) to an in-place rewrite of the state.

    The mapping below is the paper's core claim written out as code. Each branch
    is a hypothesis you can test and, later, replace with a learned policy.
    """

    def __init__(self, use_commitment_ledger: bool = True) -> None:
        self.use_commitment_ledger = use_commitment_ledger

    def apply(
        self,
        state: ConversationState,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> EditDecision:
        payload = payload or {}
        new_slots: dict[str, Any] = payload.get("slots", {}) or {}

        if event_type == "backchannel":
            # "mhm", "right", "okay": the user is not taking the floor. Nothing
            # about the task changed, so the state must be left exactly as is.
            return EditDecision(EditOp.RETAIN, reason="backchannel, no state change")

        if event_type == "barge_in_stop":
            # The user wants the agent to stop talking. Keep the belief frame,
            # but the unspoken remainder of the turn is no longer valid content
            # to resume verbatim.
            state.ledger.unspoken.clear()
            return EditDecision(EditOp.DISCARD, reason="user halted playback")

        if event_type == "self_correction":
            # The interesting case. Overwrite the corrected slots and invalidate
            # everything downstream of them.
            for slot, value in new_slots.items():
                state.belief.set(slot, value, state.turn_index, confidence=0.95)
            dropped = state.queue.invalidate(set(new_slots))
            return EditDecision(
                EditOp.REVISE,
                slots=new_slots,
                reason=f"corrected {sorted(new_slots)}",
                dropped_actions=dropped,
            )

        if event_type == "topic_switch":
            # A new request. Retire the old goal's slots but keep the ones the
            # user has confirmed and that are domain independent.
            keep = {"name", "phone", "language"}
            for slot in list(state.belief.slots):
                if slot not in keep and slot not in new_slots:
                    state.belief.drop(slot)
            for slot, value in new_slots.items():
                state.belief.set(slot, value, state.turn_index)
            dropped = state.queue.invalidate(set(state.belief.slots) | set(new_slots))
            return EditDecision(
                EditOp.DISCARD, slots=new_slots, reason="new goal", dropped_actions=dropped
            )

        if event_type == "post_pause_resume":
            # The user went quiet and came back. Low-confidence slots captured
            # right before the pause are the ones most likely to be wrong.
            stale = {
                slot
                for slot, sv in state.belief.slots.items()
                if sv.confidence < 0.6 and not sv.acknowledged
            }
            return EditDecision(
                EditOp.RE_ASK, slots={s: None for s in stale}, reason="resume after pause"
            )

        return EditDecision(EditOp.RETAIN, reason=f"unhandled event {event_type}")


class ResumptionPlanner:
    """Decides *where* to resume once the state has been repaired.

    The trade-off the paper measures: re-grounding costs the user time but
    prevents the agent from acting on a state the user never agreed to.
    """

    def __init__(self, reground_threshold: float = 0.8) -> None:
        self.reground_threshold = reground_threshold

    def plan(
        self, state: ConversationState, decision: EditDecision, event_type: str
    ) -> tuple[ResumePlan, str]:
        if decision.op is EditOp.RETAIN and event_type == "backchannel":
            return ResumePlan.CONTINUE, ""

        if decision.op is EditOp.REVISE:
            changed = sorted(decision.slots)
            # If the agent already said the old value aloud, the user is holding
            # a stale belief and a silent fix would be confusing.
            said_old = any(state.ledger.was_heard(s) for s in changed)
            if said_old or len(changed) > 1:
                pairs = ", ".join(f"{s} {decision.slots[s]}" for s in changed)
                return ResumePlan.REGROUND, f"Got it, {pairs}."
            return ResumePlan.CONTINUE, ""

        if decision.op is EditOp.RE_ASK:
            missing = sorted(decision.slots)
            if missing:
                return ResumePlan.REGROUND, f"Where were we, you wanted {missing[0]}?"
            return ResumePlan.CONTINUE, ""

        if event_type == "topic_switch":
            return ResumePlan.RESTART, ""

        return ResumePlan.CONTINUE, ""
