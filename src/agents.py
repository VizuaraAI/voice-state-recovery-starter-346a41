"""The policies compared in Table 1 of the paper.

Every policy implements the same method::

    recover(case) -> RecoveryResult

so that adding a baseline never means touching the evaluation loop. That
constraint is the whole reason this file is shaped the way it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .interruptions import CORRECTIONS, InterruptionCase
from .state import (
    ConversationState,
    EditOp,
    ResumePlan,
    ResumptionPlanner,
    StateEditOperator,
)
from .typology import get_classifier


@dataclass
class RecoveryResult:
    """What a policy produced, and enough trace to debug it."""

    predicted_state: dict[str, Any]
    response: str
    resume_plan: str = ResumePlan.CONTINUE.value
    event_type: str = ""
    edit_op: str = ""
    latency_ms: float = 0.0
    repeated_values: list[Any] = field(default_factory=list)


def _initial_state(case: InterruptionCase) -> ConversationState:
    """Seed state from the dialogue history, before the interruption lands."""
    state = ConversationState(turn_index=len(case.dialogue.history))
    for slot, value in case.dialogue.gold_state.items():
        state.belief.set(slot, value, 0, confidence=0.9)
    for slot in case.heard_values:
        if slot in state.belief.slots:
            state.belief.slots[slot].acknowledged = True
    state.queue.push("book", **{k: v for k, v in case.dialogue.gold_state.items()
                                if k != "domain"})
    return state


def _latency(cfg: dict, *components: str) -> float:
    budget = cfg["latency"]
    return float(sum(budget.get(c, 0) for c in components))


def _repeated(response: str, heard: dict[str, Any]) -> list[Any]:
    low = response.lower()
    return [v for v in heard.values() if str(v).lower() in low]


# ---------------------------------------------------------------- baselines

class NaiveResume:
    """Play the rest of the interrupted turn. What most stacks do by default."""

    name = "naive_resume"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def recover(self, case: InterruptionCase) -> RecoveryResult:
        state = _initial_state(case)
        state.ledger.commit_playback(case.dialogue.agent_turn, case.onset_fraction)
        # The remainder is replayed verbatim, so nothing in state is updated and
        # any value in the already-spoken prefix risks being said again.
        response = " ".join(state.ledger.unspoken)
        return RecoveryResult(
            predicted_state=state.belief.as_dict(),
            response=response,
            event_type="",
            edit_op=EditOp.RETAIN.value,
            latency_ms=_latency(self.cfg, "asr_partial_ms"),
            repeated_values=_repeated(response, case.heard_values),
        )


class RestartTurn:
    """Throw the turn away and say the whole thing again."""

    name = "restart_turn"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def recover(self, case: InterruptionCase) -> RecoveryResult:
        state = _initial_state(case)
        state.ledger.commit_playback(case.dialogue.agent_turn, case.onset_fraction)
        response = case.dialogue.agent_turn.text
        return RecoveryResult(
            predicted_state=state.belief.as_dict(),
            response=response,
            resume_plan=ResumePlan.RESTART.value,
            edit_op=EditOp.RETAIN.value,
            latency_ms=_latency(self.cfg, "asr_partial_ms", "turn_taking_ms"),
            repeated_values=_repeated(response, case.heard_values),
        )


class FullContextLLM:
    """Re-prompt over the whole transcript with no explicit state.

    With ``backend: offline`` this uses a transparent extractor so the pipeline
    runs with no API key. The point of the offline path is that it has the same
    *information* as a real LLM prompt (the full transcript, including text the
    user never heard) and therefore the same structural weakness.
    """

    name = "full_context_llm"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.backend = cfg["agent"].get("backend", "offline")

    def recover(self, case: InterruptionCase) -> RecoveryResult:
        state = _initial_state(case)
        # No commitment ledger: this policy sees generated text, not heard text.
        predicted = dict(state.belief.as_dict())
        predicted.update(self._extract(case.user_utterance))
        response = f"{case.dialogue.agent_turn.text}"
        return RecoveryResult(
            predicted_state=predicted,
            response=response,
            event_type="",
            edit_op=EditOp.REVISE.value,
            latency_ms=_latency(self.cfg, "asr_partial_ms", "turn_taking_ms") + 320.0,
            repeated_values=_repeated(response, case.heard_values),
        )

    def _extract(self, utterance: str) -> dict[str, Any]:
        """Surface-form slot extraction, the offline stand-in for a prompt."""
        found: dict[str, Any] = {}
        low = utterance.lower()
        for slot, values in CORRECTIONS.items():
            for value in values:
                if re.search(rf"\b{re.escape(value.lower())}\b", low):
                    found[slot] = value
                    break
        return found


class CascadedDST:
    """Zero-shot state tracking over the transcript, without a ledger."""

    name = "cascaded_dst"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.inner = FullContextLLM(cfg)

    def recover(self, case: InterruptionCase) -> RecoveryResult:
        result = self.inner.recover(case)
        state = _initial_state(case)
        # A tracker handles corrections, but it has no notion of a topic switch
        # retiring the frame, and no notion of what was actually heard.
        predicted = dict(state.belief.as_dict())
        predicted.update(self.inner._extract(case.user_utterance))
        response = " ".join(case.dialogue.agent_turn.clauses[-2:])
        return RecoveryResult(
            predicted_state=predicted,
            response=response,
            event_type="",
            edit_op=EditOp.REVISE.value,
            latency_ms=result.latency_ms + 60.0,
            repeated_values=_repeated(response, case.heard_values),
        )


# ------------------------------------------------------------------ ReState

class ReState:
    """Typed interruption, explicit state edit, planned resumption."""

    name = "restate"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        rcfg = cfg["restate"]
        self.use_ledger = bool(rcfg.get("use_commitment_ledger", True))
        self.use_typology = bool(rcfg.get("use_typology", True))
        self.use_planner = bool(rcfg.get("use_resumption_planner", True))
        self.use_editor = bool(rcfg.get("use_state_edit_operator", True))
        self.classifier = get_classifier(cfg) if self.use_typology else None
        self.editor = StateEditOperator(use_commitment_ledger=self.use_ledger)
        self.planner = ResumptionPlanner()
        self.extractor = FullContextLLM(cfg)

    def recover(self, case: InterruptionCase) -> RecoveryResult:
        state = _initial_state(case)

        if self.use_ledger:
            state.ledger.commit_playback(case.dialogue.agent_turn,
                                         case.onset_fraction)
        else:
            # Ablation: assume the user heard the entire generated turn.
            state.ledger.commit_playback(case.dialogue.agent_turn, 1.0)

        event = (self.classifier.predict(case.user_utterance)
                 if self.use_typology else "barge_in_stop")

        payload = {"slots": self.extractor._extract(case.user_utterance)}
        if event == "topic_switch" and case.payload.get("slots"):
            payload = {"slots": dict(case.payload["slots"])}

        if self.use_editor:
            decision = self.editor.apply(state, event, payload)
        else:
            # Ablation: re-extract the frame and overwrite blindly.
            for slot, value in payload["slots"].items():
                state.belief.set(slot, value, state.turn_index)
            decision = self.editor.apply(state, "backchannel", {})

        if self.use_planner:
            plan, prefix = self.planner.plan(state, decision, event)
        else:
            plan, prefix = ResumePlan.CONTINUE, ""

        if plan is ResumePlan.RESTART:
            body = case.dialogue.agent_turn.text
        elif self.use_ledger:
            body = " ".join(state.ledger.unspoken)
        else:
            body = case.dialogue.agent_turn.text
        response = f"{prefix} {body}".strip()

        components = ["asr_partial_ms", "turn_taking_ms", "state_edit_ms"]
        if self.use_typology:
            components.append("typology_ms")
        if self.use_planner:
            components.append("planner_ms")

        return RecoveryResult(
            predicted_state=state.belief.as_dict(),
            response=response,
            resume_plan=plan.value,
            event_type=event,
            edit_op=decision.op.value,
            latency_ms=_latency(self.cfg, *components),
            repeated_values=_repeated(response, case.heard_values),
        )


POLICIES = {
    NaiveResume.name: NaiveResume,
    RestartTurn.name: RestartTurn,
    FullContextLLM.name: FullContextLLM,
    CascadedDST.name: CascadedDST,
    ReState.name: ReState,
}


def build_policy(cfg: dict, name: str | None = None):
    name = name or cfg["agent"]["policy"]
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; options: {sorted(POLICIES)}")
    return POLICIES[name](cfg)
