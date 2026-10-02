from __future__ import annotations

import ast
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from sys import _getframe
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .event_data import TriggerData
    from .state import State
    from .transition import Transition


@dataclass(frozen=True)
class TransitionDefinition:
    """Stable source location and identifiers for a transition declaration."""

    module: str
    filename: str
    line: int
    index: int = 0
    event: str = ""
    source_state: str = ""
    target_states: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        targets = ",".join(self.target_states)
        return (
            f"{self.module}:{self.filename}:{self.line}:{self.index}:"
            f"{self.source_state}:{targets}:{self.event}"
        )


@dataclass
class GuardDecision:
    name: str
    status: str = "not_evaluated"
    error: str | None = None


@dataclass
class CandidateDecision:
    id: str
    definition: TransitionDefinition
    active_state: str
    source_state: str
    target_states: tuple[str, ...]
    guards: list[GuardDecision] = field(default_factory=list)
    status: str = "candidate"
    eliminated_by_rule: str | None = None
    eliminated_by_candidate_id: str | None = None
    conflicting_states: tuple[str, ...] = ()
    conflicting_domains: tuple[str, ...] = ()


@dataclass
class MicrostepDecision:
    index: int
    trigger_event: str | None
    trigger_origin: str
    candidates: list[CandidateDecision] = field(default_factory=list)
    selected_candidate_ids: list[str] = field(default_factory=list)
    executed: bool = False
    status: str = "selected"
    error: str | None = None


@dataclass
class MacrostepDecision:
    index: int
    external_event: str
    steps: list[MicrostepDecision] = field(default_factory=list)
    status: str = "in_progress"
    error: str | None = None


class DecisionTrace:
    """Records only structural state-machine decisions made by the engine."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.macrosteps: list[MacrostepDecision] = []
        self._current_macrostep: MacrostepDecision | None = None
        self._current_step: MicrostepDecision | None = None
        self._pending_step: MicrostepDecision | None = None
        self._candidate_sequence = 0
        self._candidate_transition_ids: dict[str, int] = {}
        self._candidate_ids_by_transition: dict[int, list[str]] = {}

    def __repr__(self):
        return repr(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "macrosteps": tuple(asdict(macrostep) for macrostep in self.macrosteps),
        }

    def __iter__(self):
        return iter(self.macrosteps)

    def __len__(self) -> int:
        return len(self.macrosteps)

    def begin_macrostep(self, trigger_data: "TriggerData") -> MacrostepDecision | None:
        if not self.enabled:
            return None
        event = str(trigger_data.event) if trigger_data.event is not None else None
        if event == "__initial__":
            return None
        if self._current_macrostep is not None:
            self.complete_macrostep()
        macrostep = MacrostepDecision(index=len(self.macrosteps) + 1, external_event=event)
        self.macrosteps.append(macrostep)
        self._current_macrostep = macrostep
        return macrostep

    def begin_selection(
        self, trigger_data: "TriggerData", trigger_origin: str
    ) -> MicrostepDecision | None:
        if not self.enabled:
            return None
        macrostep = self._current_macrostep
        step = MicrostepDecision(
            index=len(macrostep.steps) + 1 if macrostep is not None else 0,
            trigger_event=str(trigger_data.event) if trigger_data.event is not None else None,
            trigger_origin=trigger_origin,
        )
        self._current_step = step
        if macrostep is None:
            self._pending_step = step
        else:
            macrostep.steps.append(step)
        return step

    def add_candidate(
        self,
        transition: "Transition",
        active_state: "State",
        guards: list[GuardDecision],
    ) -> CandidateDecision | None:
        step = self._current_step
        if not self.enabled or step is None:
            return None
        self._candidate_sequence += 1
        definition = replace(
            transition.definition_location,
            event=transition.event,
            source_state=transition.source.id,
            target_states=tuple(state.id for state in transition.targets),
        )
        candidate = CandidateDecision(
            id=f"candidate-{self._candidate_sequence}",
            definition=definition,
            active_state=active_state.id,
            source_state=transition.source.id,
            target_states=tuple(state.id for state in transition.targets),
            guards=guards,
        )
        step.candidates.append(candidate)
        transition_id = id(transition)
        self._candidate_transition_ids[candidate.id] = transition_id
        self._candidate_ids_by_transition.setdefault(transition_id, []).append(candidate.id)
        return candidate

    def mark_guard_result(
        self, candidate: CandidateDecision | None, guard: GuardDecision, result: bool
    ) -> None:
        if candidate is None:
            return
        guard.status = "true" if result else "false"
        if not result:
            candidate.status = "eliminated"
            candidate.eliminated_by_rule = "guard_denied"

    def mark_guard_error(
        self, candidate: CandidateDecision | None, guard: GuardDecision, error: Exception
    ) -> None:
        if candidate is None:
            return
        guard.status = "error"
        guard.error = type(error).__name__
        candidate.status = "eliminated"
        candidate.eliminated_by_rule = "guard_error"

    def mark_candidate_enabled(self, candidate: CandidateDecision | None) -> None:
        if candidate is not None:
            candidate.status = "enabled"

    def shadow_candidate(
        self,
        loser: CandidateDecision | None,
        winner: CandidateDecision,
        rule: str,
    ) -> None:
        if loser is None:
            return
        loser.status = "eliminated"
        loser.eliminated_by_rule = rule
        loser.eliminated_by_candidate_id = winner.id

    def finish_selection(self, enabled_transitions: Any) -> None:
        step = self._current_step
        if not self.enabled or step is None:
            return
        if not step.candidates and step.trigger_origin != "external":
            self._discard_current_step(step)
            return
        enabled_ids = {id(transition) for transition in enabled_transitions}
        selected = []
        has_guard_error = False
        for candidate in step.candidates:
            transition_id = self._candidate_transition_ids.get(candidate.id)
            has_guard_error = has_guard_error or candidate.eliminated_by_rule == "guard_error"
            if candidate.status == "enabled" and transition_id in enabled_ids:
                candidate.status = "selected"
                selected.append(candidate.id)
        step.selected_candidate_ids = selected
        step.status = "executed" if selected else "error" if has_guard_error else "not_enabled"
        self._current_step = None

    def mark_exit_conflict(
        self,
        loser_transition: "Transition",
        winner_transition: "Transition",
        conflicting_states: Any,
        descendant_priority: bool,
        conflicting_domains: tuple[str, ...] = (),
    ) -> None:
        step = self._current_step
        if not self.enabled or step is None:
            return
        winner = self._candidate_for_transition(step, winner_transition)
        if winner is None:
            return
        for loser in self._candidates_for_transition(step, loser_transition):
            same_active_state = loser.active_state == winner.active_state
            loser.status = "eliminated"
            loser.eliminated_by_rule = (
                "descendant_source_priority"
                if descendant_priority and same_active_state
                else "exit_set_conflict"
            )
            loser.eliminated_by_candidate_id = winner.id
            loser.conflicting_states = tuple(info.state.id for info in conflicting_states)
            loser.conflicting_domains = conflicting_domains

    def mark_microstep(self, trigger_data: "TriggerData", transitions: Any) -> None:
        if not self.enabled:
            return
        step = self._step_for_transitions(transitions)
        if step is None:
            return
        step.executed = True
        step.status = "executed"
        step.trigger_event = str(trigger_data.event) if trigger_data.event is not None else None
        step.trigger_origin = getattr(trigger_data, "origin", None) or step.trigger_origin

    def mark_execution_error(self, error: Exception, *, fatal: bool) -> None:
        if not self.enabled:
            return
        if self._current_step is not None:
            self._current_step.error = type(error).__name__
            self._current_step.status = "error"
        if fatal and self._current_macrostep is not None:
            self._current_macrostep.status = "error"
            self._current_macrostep.error = type(error).__name__

    def mark_external_error(self, error: Exception) -> None:
        if self.enabled and self._current_macrostep is not None:
            self._current_macrostep.status = "error"
            self._current_macrostep.error = type(error).__name__

    def complete_macrostep(self) -> None:
        if not self.enabled or self._current_macrostep is None:
            return
        if self._current_macrostep.status == "in_progress":
            self._current_macrostep.status = "completed"
        self._current_macrostep = None
        self._current_step = None
        self._pending_step = None

    def _discard_current_step(self, step: MicrostepDecision) -> None:
        if self._current_macrostep is not None:
            self._current_macrostep.steps.remove(step)
        self._pending_step = None
        self._current_step = None

    def _candidate_for_transition(
        self, step: MicrostepDecision, transition: "Transition"
    ) -> CandidateDecision | None:
        candidates = self._candidates_for_transition(step, transition)
        return candidates[0] if candidates else None

    def _candidates_for_transition(
        self, step: MicrostepDecision, transition: "Transition"
    ) -> list[CandidateDecision]:
        transition_id = id(transition)
        candidates_by_id = {candidate.id: candidate for candidate in step.candidates}
        candidates = [
            candidates_by_id[candidate_id]
            for candidate_id in self._candidate_ids_by_transition.get(transition_id, [])
            if candidate_id in candidates_by_id
        ]
        return sorted(candidates, key=lambda candidate: candidate.status != "enabled")

    def _step_for_transitions(self, transitions: Any) -> MicrostepDecision | None:
        transition_ids = {id(transition) for transition in transitions}
        steps = []
        if self._current_macrostep is not None:
            steps.extend(self._current_macrostep.steps)
        if self._pending_step is not None and self._pending_step not in steps:
            steps.append(self._pending_step)
        for step in reversed(steps):
            if any(
                self._candidate_transition_ids.get(candidate.id) in transition_ids
                for candidate in step.candidates
                if candidate.status in {"enabled", "selected", "error"}
            ):
                return step
        return None


_DEFINITION_LINE_CACHE: dict[tuple[str, int], int] = {}


def _assignment_start_line(filename: str, line: int) -> int:
    if not filename or filename.startswith("<"):
        return line
    cache_key = (filename, line)
    if cache_key in _DEFINITION_LINE_CACHE:
        return _DEFINITION_LINE_CACHE[cache_key]

    try:
        with open(filename, encoding="utf-8") as source_file:
            tree = ast.parse(source_file.read(), filename=filename)
    except (OSError, SyntaxError, ValueError):
        return line

    result_line = line
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign | ast.Assign):
            continue
        end_line = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= line <= (end_line or node.lineno):
            result_line = node.lineno
            break

    _DEFINITION_LINE_CACHE[cache_key] = result_line
    return result_line


def capture_transition_definition(index: int = 0) -> TransitionDefinition:
    frame = _getframe(1)
    internal_files = {"state.py", "transition.py", "transition_list.py"}
    while (
        Path(frame.f_code.co_filename).name in internal_files
        or frame.f_code.co_name in {"<genexpr>", "<listcomp>"}
    ):
        parent = frame.f_back
        if parent is None:
            break
        frame = parent
    co_filename = frame.f_code.co_filename
    module = frame.f_globals.get("__name__", Path(co_filename).stem)
    return TransitionDefinition(
        module=str(module),
        filename=Path(co_filename).name,
        line=_assignment_start_line(co_filename, frame.f_lineno),
        index=index,
    )
