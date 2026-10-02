"""Decision tracing for transition selection.

When enabled, the engine records *why* each candidate transition was or was not
executed within a macrostep: candidate discovery, guard (condition) outcomes,
parent/child priority, and exit-set conflict resolution.

The tracer only observes decisions the engine really makes — guards are never
called extra times to produce explanations. Candidates whose guard was not
evaluated are marked as :attr:`GuardOutcome.NOT_EVALUATED`, never as failed.

Only stable identifiers (state ids, transition definition ids) are recorded —
never the model, event arguments, or other potentially sensitive payloads.

Tracing is disabled by default. Enable it on an instance and inspect the
collected macrosteps:

>>> from statemachine import State, StateChart
>>> class Machine(StateChart):
...     s1 = State(initial=True)
...     s2 = State(final=True)
...     go = s1.to(s2)
>>> sm = Machine()
>>> sm.decision_tracer.enabled = True
>>> sm.send("go")
>>> trace = sm.decision_tracer.macrosteps[-1]
>>> trace.event
'go'
>>> [(c.transition_id, c.fate.value) for c in trace.microsteps[0].candidates]
[('s1->s2:go#0', 'executed')]
"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from .state import State
    from .transition import Transition


class GuardOutcome(Enum):
    """Result of evaluating a candidate's guard (``cond``/``unless``)."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
    """The guard was never called — e.g. an earlier candidate won the scan."""


class CandidateFate(Enum):
    """What happened to a candidate transition within the microstep selection."""

    SELECTED = "selected"
    """Won its scan and survived conflict resolution, but the microstep has not
    executed yet (only observable while a macrostep is in progress)."""
    EXECUTED = "executed"
    REJECTED = "rejected"
    """The guard was evaluated and did not pass."""
    SHADOWED = "shadowed"
    """Never evaluated: an earlier candidate won the scan for the same atomic state."""
    PREEMPTED = "preempted"
    """Won its scan but lost an exit-set conflict with another enabled transition."""


@dataclass
class CandidateRecord:
    """A single candidate transition considered during transition selection."""

    transition_id: str
    """Stable definition id, see :func:`transition_definition_id`."""
    source_state: str
    """Id of the state the transition is declared on."""
    evaluated_for: str
    """Id of the atomic state whose scan considered this candidate."""
    guard: GuardOutcome
    fate: CandidateFate
    eliminated_by: "str | None" = None
    """Definition id of the transition that eliminated this candidate, if any."""
    rule: "str | None" = None
    """The rule that decided the fate: ``guard``, ``child_priority``,
    ``document_order``, or ``descendant_priority``."""
    exit_set_overlap: tuple = ()
    """Ids of the states in the exit-set intersection, for preempted candidates."""


@dataclass
class MicrostepTrace:
    """One selection round (and its execution) within a macrostep."""

    event: "str | None"
    """The event being processed — ``None`` for eventless selection rounds."""
    kind: str
    """``external``, ``internal``, or ``eventless``."""
    candidates: list = field(default_factory=list)
    """List of :class:`CandidateRecord`, in evaluation order."""
    executed: list = field(default_factory=list)
    """Definition ids of the transitions the microstep actually executed."""


@dataclass
class MacrostepTrace:
    """All decisions made while processing one external event.

    Includes the eventless and internal-event microsteps triggered by it, in
    causal order.
    """

    index: int
    event: "str | None"
    microsteps: list = field(default_factory=list)
    """List of :class:`MicrostepTrace`, in causal order."""


def transition_definition_id(transition: "Transition") -> str:
    """Build a stable identifier for a transition definition.

    The format is ``<source>-><targets>:<events>#<ordinal>``, where ``ordinal``
    is the transition's position in its source state's declaration order — so
    two transitions with the same source, target, and event still get distinct
    ids.

    >>> from statemachine import State, StateChart
    >>> class Machine(StateChart):
    ...     s1 = State(initial=True)
    ...     s2 = State(final=True)
    ...     go = s1.to(s2)
    >>> transition_definition_id(Machine.s1.transitions[0])
    's1->s2:go#0'
    """
    source = transition.source
    event = transition.event or "<eventless>"
    targets = ",".join(target.id for target in transition.targets) or "<none>"
    ordinal = list(source.transitions).index(transition)
    return f"{source.id}->{targets}:{event}#{ordinal}"


class DecisionTracer:
    """Collects :class:`MacrostepTrace` records while the engine processes events.

    All ``record_*``/``begin_*`` methods are no-ops when :attr:`enabled` is
    ``False`` (the default), so tracing never changes the engine's behavior.
    """

    def __init__(self):
        self.enabled = False
        self.macrosteps: list = []
        """List of completed and in-progress :class:`MacrostepTrace` records."""
        self._current_macrostep: "MacrostepTrace | None" = None
        self._pending_microstep: "MicrostepTrace | None" = None

    @property
    def current(self) -> "MacrostepTrace | None":
        """The macrostep currently being recorded.

        Points to the partial trace when an exception aborts a macrostep.
        """
        return self._current_macrostep

    def clear(self):
        """Drop all recorded traces."""
        self.macrosteps.clear()
        self._current_macrostep = None
        self._pending_microstep = None

    def begin_macrostep(self, index: int, event: "str | None"):
        """Start recording a new macrostep."""
        if not self.enabled:
            return
        current = self._current_macrostep
        if current is not None and current.event is None and not current.microsteps:
            # Replace an implicit macrostep that recorded nothing.
            self.macrosteps.remove(current)
        self._current_macrostep = MacrostepTrace(index=index, event=event)
        self.macrosteps.append(self._current_macrostep)
        self._pending_microstep = None

    def begin_microstep(self, event: "str | None", kind: str):
        """Start recording a selection round within the current macrostep."""
        if not self.enabled:
            return
        if self._current_macrostep is None:
            self.begin_macrostep(0, None)
        self._pending_microstep = MicrostepTrace(event=event, kind=kind)

    def _materialize(self) -> MicrostepTrace:
        """Attach the pending microstep to the current macrostep and return it.

        Selection rounds that produce no candidates and no execution are never
        materialized, keeping the trace free of empty polling rounds.
        """
        if self._pending_microstep is None:
            self.begin_microstep(None, "eventless")
        microstep = cast(MicrostepTrace, self._pending_microstep)
        macrostep = cast(MacrostepTrace, self._current_macrostep)
        if not macrostep.microsteps or macrostep.microsteps[-1] is not microstep:
            macrostep.microsteps.append(microstep)
        return microstep

    def record_guard(self, transition: "Transition", evaluated_for: "State", matched: bool):
        """Record the outcome of a guard the engine actually evaluated."""
        if not self.enabled:
            return
        microstep = self._materialize()
        microstep.candidates.append(
            CandidateRecord(
                transition_id=transition_definition_id(transition),
                source_state=transition.source.id,
                evaluated_for=evaluated_for.id,
                guard=GuardOutcome.PASSED if matched else GuardOutcome.FAILED,
                fate=CandidateFate.SELECTED if matched else CandidateFate.REJECTED,
                rule=None if matched else "guard",
            )
        )

    def record_shadowed(
        self, transition: "Transition", evaluated_for: "State", winner: "Transition"
    ):
        """Record a candidate that was never evaluated because ``winner`` matched first.

        Only event matching is checked by the caller — guards are never called
        for shadowed candidates, so they keep :attr:`GuardOutcome.NOT_EVALUATED`.
        """
        if not self.enabled:
            return
        if winner.source.is_descendant(transition.source):
            rule = "child_priority"
        else:
            rule = "document_order"
        microstep = self._materialize()
        microstep.candidates.append(
            CandidateRecord(
                transition_id=transition_definition_id(transition),
                source_state=transition.source.id,
                evaluated_for=evaluated_for.id,
                guard=GuardOutcome.NOT_EVALUATED,
                fate=CandidateFate.SHADOWED,
                eliminated_by=transition_definition_id(winner),
                rule=rule,
            )
        )

    def record_preemption(
        self,
        preempted: "Transition",
        winner: "Transition",
        rule: str,
        exit_set_overlap: tuple,
    ):
        """Record that ``preempted`` lost an exit-set conflict against ``winner``."""
        if not self.enabled:
            return
        microstep = self._materialize()
        preempted_id = transition_definition_id(preempted)
        winner_id = transition_definition_id(winner)
        for candidate in microstep.candidates:
            if (
                candidate.transition_id == preempted_id
                and candidate.fate is CandidateFate.SELECTED
            ):
                candidate.fate = CandidateFate.PREEMPTED
                candidate.eliminated_by = winner_id
                candidate.rule = rule
                candidate.exit_set_overlap = tuple(exit_set_overlap)

    def record_execution(self, transitions: "list[Transition]"):
        """Record that the given transitions were executed by the microstep."""
        if not self.enabled:
            return
        microstep = self._materialize()
        executed_ids = [transition_definition_id(t) for t in transitions]
        microstep.executed.extend(executed_ids)
        executed_ids_set = set(executed_ids)
        for candidate in microstep.candidates:
            if (
                candidate.fate is CandidateFate.SELECTED
                and candidate.transition_id in executed_ids_set
            ):
                candidate.fate = CandidateFate.EXECUTED
