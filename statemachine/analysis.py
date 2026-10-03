"""Structural (static) analysis of statechart configurations.

The analyzer in this module inspects the *structure* of a statechart — states,
transitions, hierarchy, parallel regions, and history pseudo-states — to detect
configuration-level problems that per-state checks cannot see:

- states that no configuration exploration can reach;
- non-final configurations with no successors (dead ends);
- cycles formed by eventless transitions (the machine never goes idle).

The analysis **never executes user code**: guards (``cond``/``unless``),
validators, and actions are not called. Unguarded transitions are treated as
certain edges, while guarded transitions have an *unknown* outcome and are
reported conservatively. Results are therefore split into two certainty levels:

- ``Certainty.DEFINITE``: derived using only unguarded transitions;
- ``Certainty.POSSIBLE``: depends on the unknown outcome of at least one
  guard, on transitions shadowed by guarded ones, or on runtime-only data
  such as recorded history values.

Because of this approximation, a "possible" finding is **not** a proof of a
business-level deadlock — the guards involved may always evaluate to ``True``
at runtime. Likewise, reachability only means the structure allows reaching a
configuration; it does not guarantee the required events are ever sent.
"""

import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from itertools import product
from typing import TYPE_CHECKING
from typing import cast

from .callbacks import CallbackGroup
from .exceptions import InvalidDefinition
from .graph import iterate_states
from .i18n import _
from .state import HistoryState
from .state import State

if TYPE_CHECKING:
    from .statemachine import StateChart
    from .transition import Transition

__all__ = [
    "AnalysisReport",
    "Certainty",
    "Finding",
    "FindingKind",
    "PathStep",
    "analyze_statechart",
]

_Config = frozenset[str]
"""A configuration: the set of ids of the active atomic states."""

_GUARD_GROUPS = frozenset({CallbackGroup.COND, CallbackGroup.VALIDATOR})
"""Callback groups whose outcome depends on user code evaluated at runtime."""


class Certainty(Enum):
    """How certain a result is, given that guard outcomes are unknown."""

    DEFINITE = "definite"
    """Derived using only unguarded transitions (no unknown conditions involved)."""

    POSSIBLE = "possible"
    """Depends on the unknown outcome of at least one guard condition."""


class FindingKind(Enum):
    """Kinds of structural problems reported by the analysis."""

    UNREACHABLE_STATE = "unreachable_state"
    DEAD_CONFIGURATION = "dead_configuration"
    EVENTLESS_CYCLE = "eventless_cycle"


@dataclass(frozen=True)
class PathStep:
    """One microstep of a witness path."""

    kind: str
    """One of ``"initial"``, ``"event"`` or ``"eventless"``."""

    event: "str | None"
    """The triggering event id, or ``None`` for eventless/initial steps."""

    source: str
    """Id of the source state of the transition (empty for the initial step)."""

    targets: tuple[str, ...]
    """Ids of the target states of the transition."""

    guarded: bool
    """Whether the transition has conditions (``cond``/``unless``) or validators."""


@dataclass(frozen=True)
class Finding:
    """A single structural problem (or potential problem)."""

    kind: FindingKind
    certainty: Certainty
    states: tuple[str, ...]
    """Ids of the states involved in the finding."""

    configuration: "tuple[str, ...] | None"
    """The configuration the finding refers to, when applicable."""

    path: "tuple[PathStep, ...] | None"
    """A witness path from the initial configuration, or ``None`` when not applicable.

    For ``Certainty.DEFINITE`` findings the path uses only unguarded
    transitions. For ``Certainty.POSSIBLE`` findings it may contain guarded
    steps (marked with ``guarded=True``).
    """

    path_is_shortest: bool
    """Whether ``path`` is a shortest witness. ``False`` when the search was
    incomplete (configuration budget exhausted) or no path applies."""

    message: str


@dataclass(frozen=True)
class AnalysisReport:
    """Result of :func:`analyze_statechart`."""

    complete: bool
    """``False`` when the configuration budget was exhausted; results are partial."""

    max_configurations: int
    configurations_explored: int
    definitely_reachable_states: frozenset[str]
    possibly_reachable_states: frozenset[str]
    definitely_reachable_configurations: "frozenset[_Config]"
    possibly_reachable_configurations: "frozenset[_Config]"
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        """``True`` when no finding was reported."""
        return not self.findings

    @property
    def unreachable_states(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind is FindingKind.UNREACHABLE_STATE)

    @property
    def dead_configurations(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind is FindingKind.DEAD_CONFIGURATION)

    @property
    def eventless_cycles(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind is FindingKind.EVENTLESS_CYCLE)


@dataclass
class _Record:
    """Best known way to reach a configuration."""

    certainty: Certainty
    path: tuple[PathStep, ...]
    shortest: int


@dataclass
class _Draft:
    """A finding before the final report is assembled."""

    kind: FindingKind
    certainty: Certainty
    states: tuple[str, ...]
    configuration: "tuple[str, ...] | None"
    record: "_Record | None"
    suffix: tuple[PathStep, ...]
    message: str


class _Analyzer:
    """Explores the configuration graph of a statechart without running user code."""

    def __init__(self, sm_class: "type[StateChart]", max_configurations: int):
        self.sm_class = sm_class
        self.max_configurations = max_configurations
        self._option_cap = max(16, max_configurations)
        self.states_by_id = {state.id: state for state in iterate_states(sm_class.states)}
        self.complete = True
        self.records: "dict[_Config, _Record]" = {}
        self._stable: "dict[_Config, Certainty]" = {}
        self._expanded: "dict[_Config, Certainty]" = {}
        self._state_certainty: dict[str, Certainty] = {}
        self._eventless_edges: "dict[tuple[_Config, _Config], tuple[PathStep, ...]]" = {}
        self._dead: "dict[_Config, _Draft]" = {}
        self._cycles: "dict[tuple[str, object], _Draft]" = {}
        self._possible_cycles: "list[_Draft]" = []
        self._queue: "list[tuple[int, int, _Config]]" = []
        self._counter = 0

    # -- Public driver ---------------------------------------------------------

    def analyze(self) -> AnalysisReport:
        self._discover_initial()
        while self._queue:
            _, _, config = heapq.heappop(self._queue)
            self._expand(config)
        self._detect_possible_cycles()
        drafts = [
            *self._unreachable_drafts(),
            *self._dead.values(),
            *self._cycles.values(),
            *self._possible_cycles,
        ]
        findings = tuple(
            sorted(
                (self._finalize(draft) for draft in drafts),
                key=lambda finding: (finding.kind.value, finding.states),
            )
        )
        return AnalysisReport(
            complete=self.complete,
            max_configurations=self.max_configurations,
            configurations_explored=len(self.records),
            definitely_reachable_states=frozenset(
                state_id
                for state_id, certainty in self._state_certainty.items()
                if certainty is Certainty.DEFINITE
            ),
            possibly_reachable_states=frozenset(self._state_certainty),
            definitely_reachable_configurations=frozenset(
                config
                for config, certainty in self._stable.items()
                if certainty is Certainty.DEFINITE
            ),
            possibly_reachable_configurations=frozenset(self._stable),
            findings=findings,
        )

    # -- Initial configuration -------------------------------------------------

    def _discover_initial(self) -> None:
        start_values = self.sm_class.start_configuration_values
        if start_values:
            starts = [self.sm_class.states_map[value] for value in start_values]
        else:
            initial = self.sm_class.initial_state
            assert initial is not None  # guaranteed by the abstract-class check
            starts = [initial]
        options: "list[_Config]" = [frozenset()]
        for state in starts:
            options = self._combine(options, self._descendant_options(state))
        certainty = Certainty.DEFINITE if len(options) == 1 else Certainty.POSSIBLE
        for config in options:
            step = PathStep(
                kind="initial",
                event=None,
                source="",
                targets=tuple(sorted(config)),
                guarded=False,
            )
            self._discover(config, certainty, (step,))

    # -- Static expansion of states --------------------------------------------

    def _descendant_options(self, state: State) -> "list[_Config]":
        """Possible atomic leaf sets activated when entering ``state`` by default."""
        if state.is_history:
            return self._history_options(cast(HistoryState, state))
        if state.is_atomic:
            return [frozenset((state.id,))]
        if state.parallel:
            options: "list[_Config]" = [frozenset()]
            for child in state.states:
                options = self._combine(options, self._descendant_options(child))
            return options
        initial = next((t for t in state.transitions if t.initial), None)
        if initial is None:  # pragma: no cover - the factory always creates one
            return [frozenset()]
        return self._targets_options(initial.targets, bound=state)

    def _targets_options(self, targets: Iterable[State], bound: State) -> "list[_Config]":
        targets = list(targets)
        options: "list[_Config]" = [frozenset()]
        for target in targets:
            options = self._combine(options, self._descendant_options(target))
        # Parallel-region completion considers the combined targets, mirroring
        # the engine's entry-set computation (siblings of a target are only
        # default-entered when no other target covers them).
        for target in targets:
            options = self._complete_ancestors(target, bound, options)
        return options

    def _complete_ancestors(
        self, state: State, bound: "State | None", options: "list[_Config]"
    ) -> "list[_Config]":
        """Default-enter parallel regions on the ancestor chain up to ``bound``."""
        for ancestor in state.ancestors(parent=bound):
            if not ancestor.parallel:
                continue
            completed: "list[_Config]" = []
            for option in options:
                missing = [
                    child
                    for child in ancestor.states
                    if not any(self._within(leaf_id, child) for leaf_id in option)
                ]
                if not missing:
                    completed.append(option)
                    continue
                choices = [self._descendant_options(child) for child in missing]
                for combo in product(*choices):
                    completed.append(option.union(*combo))
                    if len(completed) > self._option_cap:
                        self.complete = False
                        return self._dedupe(completed)
            options = self._cap(completed)
        return options

    def _history_options(self, history: HistoryState) -> "list[_Config]":
        """Over-approximate the configurations a history pseudo-state may restore."""
        parent = history.parent
        if parent is None:  # pragma: no cover - history states always have a parent
            return [frozenset()]
        options: "list[_Config]" = []
        for transition in history.transitions:
            options.extend(self._targets_options(transition.targets, bound=parent))
        if not history.transitions:
            # No default content: entering with no recorded value enters nothing.
            options.append(frozenset())
        if history.type.is_deep:
            options.extend(self._all_subconfigurations(parent))
        elif parent.parallel:
            options.extend(self._descendant_options(parent))
        else:
            for child in parent.states:
                options.extend(self._descendant_options(child))
        return self._cap(options)

    def _all_subconfigurations(self, state: State) -> "list[_Config]":
        """Every leaf set that may be active within ``state`` (over-approximation)."""
        if state.is_atomic:
            return [frozenset((state.id,))]
        if state.parallel:
            options: "list[_Config]" = [frozenset()]
            for child in state.states:
                options = self._combine(options, self._all_subconfigurations(child))
            return options
        options = []
        for child in state.states:
            options.extend(self._all_subconfigurations(child))
        return self._cap(options)

    def _combine(self, left: "list[_Config]", right: "list[_Config]") -> "list[_Config]":
        combined: "list[_Config]" = []
        for left_option in left:
            for right_option in right:
                combined.append(left_option | right_option)
                if len(combined) > self._option_cap:
                    self.complete = False
                    return self._dedupe(combined)
        return self._dedupe(combined)

    def _cap(self, options: "list[_Config]") -> "list[_Config]":
        options = self._dedupe(options)
        if len(options) > self._option_cap:
            self.complete = False
            return options[: self._option_cap]
        return options

    @staticmethod
    def _dedupe(options: "list[_Config]") -> "list[_Config]":
        return list(dict.fromkeys(options))

    def _within(self, leaf_id: str, state: State) -> bool:
        leaf = self.states_by_id[leaf_id]
        return leaf is state or leaf.is_descendant(state)

    # -- Static transition semantics -------------------------------------------

    @staticmethod
    def _find_lcca(states: "list[State]") -> "State | None":
        """Least common compound ancestor, mirroring the engine's semantics."""
        head, *tail = states
        ancestors = [ancestor for ancestor in head.ancestors() if ancestor.is_compound]
        for ancestor in ancestors:
            if all(state.is_descendant(ancestor) for state in tail):
                return ancestor
        return None

    def _transition_domain(self, transition: "Transition") -> "State | None":
        targets: list[State] = []
        for target in transition.targets:
            if target.is_history:
                parent = target.parent
                assert parent is not None  # history states always belong to a compound
                targets.append(parent)
            else:
                targets.append(target)
        if transition.internal and transition.source.is_compound and all(
            target.is_descendant(transition.source) for target in targets
        ):
            return transition.source
        if (
            transition.internal
            and transition.is_self
            and transition.target is not None
            and transition.target.is_atomic
        ):
            return transition.source
        return self._find_lcca([transition.source, *targets])

    def _exit_leaves(self, config: _Config, transitions: "list[Transition]") -> set[str]:
        exited: set[str] = set()
        for transition in transitions:
            if not transition.targets:
                continue
            domain = self._transition_domain(transition)
            for leaf_id in config:
                leaf = self.states_by_id[leaf_id]
                if domain is None or leaf.is_descendant(domain):
                    exited.add(leaf_id)
        return exited

    def _entry_options(self, transition: "Transition") -> "list[_Config]":
        if not transition.targets:
            return [frozenset()]
        return self._targets_options(transition.targets, self._transition_domain(transition))

    def _apply(self, config: _Config, transitions: "list[Transition]") -> "list[_Config]":
        """Possible successor configurations of a microstep (never runs callbacks)."""
        exited = self._exit_leaves(config, transitions)
        options: "list[_Config]" = [frozenset(config - exited)]
        for transition in transitions:
            options = self._combine(options, self._entry_options(transition))
        return options

    # -- Transition selection ---------------------------------------------------

    @staticmethod
    def _is_guarded(transition: "Transition") -> bool:
        """Whether firing depends on user code (``cond``/``unless``/validators)."""
        return any(spec.group in _GUARD_GROUPS for spec in transition._specs)

    @staticmethod
    def _matches(transition: "Transition", event: "str | None") -> bool:
        if event is None:
            return transition.is_eventless
        return transition.match(event)

    def _leaf_selection(self, leaf: State, event: "str | None"):
        """Mimic the runtime per-state selection without evaluating guards.

        Returns ``(definite, possible)``: ``definite`` is the transition that
        certainly fires for this leaf (its first match is unguarded), while
        ``possible`` holds guarded matches and unguarded matches shadowed by
        them, which fire only for some guard outcomes.
        """
        possible: "list[Transition]" = []
        for state in chain([leaf], leaf.ancestors()):
            for transition in state.transitions:
                if transition.initial or not self._matches(transition, event):
                    continue
                if self._is_guarded(transition):
                    possible.append(transition)
                    continue
                if possible:
                    possible.append(transition)
                    return None, possible
                return transition, possible
        return None, possible

    def _sorted_leaves(self, config: _Config) -> list[str]:
        return sorted(config, key=lambda state_id: self.states_by_id[state_id].document_order)

    def _select(self, config: _Config, event: "str | None"):
        definite: "list[tuple[str, Transition]]" = []
        possible: "list[tuple[str, Transition]]" = []
        for leaf_id in self._sorted_leaves(config):
            leaf = self.states_by_id[leaf_id]
            selected, maybe = self._leaf_selection(leaf, event)
            if selected is not None:
                definite.append((leaf_id, selected))
            possible.extend((leaf_id, transition) for transition in maybe)
        return definite, possible

    def _filter_conflicting(
        self, transitions: "list[Transition]", config: _Config
    ) -> "list[Transition]":
        """Port of the engine's conflict resolution (document order / descendancy)."""
        filtered: "list[Transition]" = []
        for transition in transitions:
            exit_set = self._exit_leaves(config, [transition])
            preempted = False
            kept: "list[Transition]" = []
            for other in filtered:
                if exit_set & self._exit_leaves(config, [other]):
                    if transition.source.is_descendant(other.source):
                        continue
                    preempted = True
                    break
                kept.append(other)
            if not preempted:
                kept.append(transition)
                filtered = kept
        return filtered

    def _event_descriptors(self, config: _Config) -> list[str]:
        descriptors: list[str] = []
        seen: set[str] = set()
        for leaf_id in self._sorted_leaves(config):
            leaf = self.states_by_id[leaf_id]
            for state in chain([leaf], leaf.ancestors()):
                for transition in state.transitions:
                    if transition.initial:
                        continue
                    for event in transition.events:
                        if event.id not in seen:
                            seen.add(event.id)
                            descriptors.append(event.id)
        return descriptors

    # -- Exploration -------------------------------------------------------------

    def _discover(self, config: _Config, certainty: Certainty, path: tuple[PathStep, ...]):
        record = self.records.get(config)
        if record is None:
            if len(self.records) >= self.max_configurations:
                self.complete = False
                return
            self.records[config] = _Record(certainty=certainty, path=path, shortest=len(path))
        else:
            record.shortest = min(record.shortest, len(path))
            if record.certainty is Certainty.DEFINITE or certainty is Certainty.POSSIBLE:
                return
            record.certainty = Certainty.DEFINITE
            record.path = path
        heapq.heappush(self._queue, (len(path), self._counter, config))
        self._counter += 1
        self._mark_reachable(config, certainty)

    def _mark_reachable(self, config: _Config, certainty: Certainty) -> None:
        for leaf_id in config:
            self._mark_state(leaf_id, certainty)
            for ancestor in self.states_by_id[leaf_id].ancestors():
                self._mark_state(ancestor.id, certainty)

    def _mark_state(self, state_id: str, certainty: Certainty) -> None:
        current = self._state_certainty.get(state_id)
        if current is None or certainty is Certainty.DEFINITE:
            self._state_certainty[state_id] = certainty

    def _set_stable(self, config: _Config, certainty: Certainty) -> None:
        current = self._stable.get(config)
        if current is None or certainty is Certainty.DEFINITE:
            self._stable[config] = certainty

    def _step(self, event: "str | None", transition: "Transition") -> PathStep:
        return PathStep(
            kind="eventless" if event is None else "event",
            event=event,
            source=transition.source.id,
            targets=tuple(target.id for target in transition.targets),
            guarded=self._is_guarded(transition),
        )

    def _expand(self, config: _Config) -> None:
        record = self.records[config]
        previous = self._expanded.get(config)
        if previous is not None and (
            previous is Certainty.DEFINITE or record.certainty is Certainty.POSSIBLE
        ):
            return
        self._expanded[config] = record.certainty

        certainty = record.certainty
        path = record.path
        chain_start = config
        current = config
        configs = [current]
        paths = [path]
        transitions_chain: "list[tuple[Transition, ...]]" = []
        index = {current: 0}

        # Follow the deterministic (unguarded) eventless closure.
        while True:
            self._mark_reachable(current, certainty)
            definite, possible = self._select(current, event=None)
            definite_ts = self._filter_conflicting([t for _, t in definite], current)
            definite_ids = {id(t) for t in definite_ts}
            for _, transition in possible:
                # A leaf never contributes to both ``definite`` and ``possible``
                # (see ``_leaf_selection``), so only conflict-removed definite
                # transitions need to be filtered out here.
                others = [t for _, t in definite if id(t) in definite_ids]
                self._emit_eventless_edge(
                    chain_start, current, [*others, transition], path[len(record.path) :], path
                )
            if not definite_ts:
                break
            steps = tuple(self._step(None, t) for t in definite_ts)
            successors = self._apply(current, definite_ts)
            if len(successors) != 1:
                # History-dependent branching: each outcome is only possible.
                for successor in successors:
                    self._discover(successor, Certainty.POSSIBLE, path + steps)
                    self._eventless_edges[(chain_start, successor)] = (
                        path[len(record.path) :] + steps
                    )
                return
            next_config = successors[0]
            path = path + steps
            if next_config in index:
                start = index[next_config]
                cycle = [*transitions_chain[start:], tuple(definite_ts)]
                self._report_definite_cycle(
                    configs[start], cycle, record, paths[start][len(record.path) :]
                )
                return
            if len(configs) >= self.max_configurations:
                # Defensive bound for pathological eventless chains.
                self.complete = False
                break
            transitions_chain.append(tuple(definite_ts))
            configs.append(next_config)
            paths.append(path)
            index[next_config] = len(configs) - 1
            current = next_config

        # `current` is stable: register it and explore evented successors.
        self._mark_reachable(current, certainty)
        self._discover(current, certainty, path)
        if current is not chain_start:
            self._eventless_edges[(chain_start, current)] = path[len(record.path) :]
        self._set_stable(current, certainty)
        if self._expanded.get(current) is not Certainty.DEFINITE:
            self._expanded[current] = certainty
        if not self._is_final_config(current):
            self._expand_events(current, certainty, path)
        self._check_dead(current, record, path[len(record.path) :])

    def _emit_eventless_edge(
        self,
        chain_start: _Config,
        current: _Config,
        transitions: "list[Transition]",
        chain_steps: tuple[PathStep, ...],
        path: tuple[PathStep, ...],
    ) -> None:
        filtered = self._filter_conflicting(transitions, current)
        steps = tuple(self._step(None, t) for t in filtered)
        for successor in self._apply(current, filtered):
            self._discover(successor, Certainty.POSSIBLE, path + steps)
            self._eventless_edges[(chain_start, successor)] = chain_steps + steps

    def _expand_events(
        self, config: _Config, certainty: Certainty, path: tuple[PathStep, ...]
    ) -> None:
        for event in self._event_descriptors(config):
            definite, possible = self._select(config, event)
            definite_ts = [t for _, t in definite]
            if definite_ts:
                self._emit_edge(config, definite_ts, event, True, certainty, path)
            for _, transition in possible:
                # Same-leaf definite selections cannot coexist with possible ones.
                others = [t for _, t in definite]
                self._emit_edge(config, [*others, transition], event, False, certainty, path)

    def _emit_edge(
        self,
        config: _Config,
        transitions: "list[Transition]",
        event: str,
        definite_edge: bool,
        certainty: Certainty,
        path: tuple[PathStep, ...],
    ) -> None:
        filtered = self._filter_conflicting(transitions, config)
        steps = tuple(self._step(event, t) for t in filtered)
        successors = self._apply(config, filtered)
        definite = definite_edge and len(successors) == 1
        for successor in successors:
            self._discover(
                successor, certainty if definite else Certainty.POSSIBLE, path + steps
            )

    # -- Final configurations ----------------------------------------------------

    def _is_active(self, state: State, config: _Config) -> bool:
        return any(
            leaf_id == state.id or self.states_by_id[leaf_id].is_descendant(state)
            for leaf_id in config
        )

    def _region_final(self, state: State, config: _Config) -> bool:
        if state.parallel:
            return all(self._region_final(child, config) for child in state.states)
        if state.is_compound:
            return any(
                child.final and self._is_active(child, config) for child in state.states
            )
        return False

    def _is_final_config(self, config: _Config) -> bool:
        """Whether the machine terminates in ``config`` (root final or all regions final)."""
        for top in self.sm_class.states:
            if top.final and self._is_active(top, config):
                return True
            if top.parallel and all(
                self._region_final(child, config) for child in top.states
            ):
                return True
        return False

    # -- Findings ------------------------------------------------------------------

    def _check_dead(
        self, config: _Config, record: _Record, suffix: tuple[PathStep, ...]
    ) -> None:
        if self._is_final_config(config):
            return
        has_unguarded = False
        has_guarded = False
        for leaf_id in config:
            leaf = self.states_by_id[leaf_id]
            for state in chain([leaf], leaf.ancestors()):
                for transition in state.transitions:
                    if transition.initial:
                        continue
                    if self._is_guarded(transition):
                        has_guarded = True
                    else:
                        has_unguarded = True
        if has_unguarded:
            return
        label = ", ".join(sorted(config))
        if has_guarded:
            certainty = Certainty.POSSIBLE
            message = (
                f"Configuration [{label}] is not final and only has guarded outgoing "
                "transitions: it is a dead end if every guard evaluates to False "
                "(guards were not evaluated)."
            )
        else:
            certainty = record.certainty
            message = (
                f"Configuration [{label}] is not final and has no outgoing transitions."
            )
        existing = self._dead.get(config)
        if existing is not None and existing.certainty is Certainty.DEFINITE:
            return
        self._dead[config] = _Draft(
            kind=FindingKind.DEAD_CONFIGURATION,
            certainty=certainty,
            states=tuple(sorted(config)),
            configuration=tuple(sorted(config)),
            record=record,
            suffix=suffix,
            message=message,
        )

    def _report_definite_cycle(
        self,
        start_config: _Config,
        cycle: "list[tuple[Transition, ...]]",
        record: _Record,
        chain_suffix: tuple[PathStep, ...],
    ) -> None:
        key = ("transitions", tuple(sorted(id(t) for steps in cycle for t in steps)))
        cycle_steps = tuple(self._step(None, t) for steps in cycle for t in steps)
        states = tuple(
            sorted(
                {step.source for step in cycle_steps}
                | {target for step in cycle_steps for target in step.targets}
            )
        )
        existing = self._cycles.get(key)
        if existing is not None and existing.certainty is Certainty.DEFINITE:
            return
        self._cycles[key] = _Draft(
            kind=FindingKind.EVENTLESS_CYCLE,
            certainty=record.certainty,
            states=states,
            configuration=tuple(sorted(start_config)),
            record=record,
            suffix=chain_suffix + cycle_steps,
            message=(
                f"Eventless transitions form a loop through states {list(states)}: "
                "the machine never reaches a stable configuration "
                "(run-to-completion does not terminate)."
            ),
        )

    def _detect_possible_cycles(self) -> None:
        graph: "dict[_Config, list[tuple[_Config, tuple[PathStep, ...]]]]" = {}
        for (src, dst), steps in self._eventless_edges.items():
            if src in self.records and dst in self.records:
                graph.setdefault(src, []).append((dst, steps))
        color: "dict[_Config, int]" = {}
        for root in graph:
            if color.get(root, 0) != 0:
                continue
            color[root] = 1
            stack = [(root, iter(graph.get(root, [])))]
            configs = [root]
            steps_stack: "list[tuple[PathStep, ...]]" = []
            while stack:
                node, edges = stack[-1]
                advanced = False
                for dst, steps in edges:
                    dst_color = color.get(dst, 0)
                    if dst_color == 1:
                        start = configs.index(dst)
                        self._add_possible_cycle(
                            configs[start:], [*steps_stack[start:], steps]
                        )
                    elif dst_color == 0:
                        color[dst] = 1
                        stack.append((dst, iter(graph.get(dst, []))))
                        configs.append(dst)
                        steps_stack.append(steps)
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    configs.pop()
                    if steps_stack:
                        steps_stack.pop()
                    color[node] = 2

    def _add_possible_cycle(
        self, cycle_configs: "list[_Config]", cycle_steps: "list[tuple[PathStep, ...]]"
    ) -> None:
        start = cycle_configs[0]
        record = self.records[start]
        states = tuple(
            sorted(
                {step.source for steps in cycle_steps for step in steps}
                | {target for steps in cycle_steps for step in steps for target in step.targets}
            )
        )
        self._possible_cycles.append(
            _Draft(
            kind=FindingKind.EVENTLESS_CYCLE,
            certainty=Certainty.POSSIBLE,
            states=states,
            configuration=tuple(sorted(start)),
            record=record,
            suffix=tuple(step for steps in cycle_steps for step in steps),
            message=(
                f"Eventless transitions may form a loop through states {list(states)}, "
                "depending on guard outcomes or recorded history (not evaluated)."
            ),
            )
        )

    def _unreachable_drafts(self) -> "list[_Draft]":
        drafts = []
        for state_id, state in self.states_by_id.items():
            if state.is_history or state_id in self._state_certainty:
                continue
            if self.complete:
                certainty = Certainty.DEFINITE
                message = f"State '{state_id}' is not reachable from the initial configuration."
            else:
                certainty = Certainty.POSSIBLE
                message = (
                    f"State '{state_id}' was not reached, but the exploration is "
                    "incomplete (configuration limit reached): it may still be reachable."
                )
            drafts.append(
                _Draft(
                    kind=FindingKind.UNREACHABLE_STATE,
                    certainty=certainty,
                    states=(state_id,),
                    configuration=None,
                    record=None,
                    suffix=(),
                    message=message,
                )
            )
        return drafts

    def _finalize(self, draft: _Draft) -> Finding:
        if draft.record is None:
            path = None
            path_is_shortest = False
        else:
            path = draft.record.path + draft.suffix
            path_is_shortest = self.complete and len(draft.record.path) == draft.record.shortest
        return Finding(
            kind=draft.kind,
            certainty=draft.certainty,
            states=draft.states,
            configuration=draft.configuration,
            path=path,
            path_is_shortest=path_is_shortest,
            message=draft.message,
        )


def analyze_statechart(
    sm_class: "type[StateChart]", max_configurations: int = 1000
) -> AnalysisReport:
    """Statically analyze the configuration graph of a statechart class.

    Explores reachable configurations — expanding hierarchical entry, parallel
    regions, and history pseudo-states — and reports unreachable states,
    non-final configurations with no successors, and eventless-transition
    cycles. No user code (guards, validators, actions) is executed: unguarded
    transitions are certain edges and guarded transitions are unknown, so
    results distinguish ``Certainty.DEFINITE`` from ``Certainty.POSSIBLE``.

    .. note::
        This is a structural approximation, not a proof of business-level
        deadlock: guards may always pass at runtime, and declared events may
        never be sent.

    Args:
        sm_class: The :ref:`StateChart` class to analyze.
        max_configurations: Maximum number of configurations to explore.
            When the limit is reached, the returned report has
            ``complete=False`` and contains the partial results known so far.

    Returns:
        An :class:`AnalysisReport` with the findings.
    """
    if max_configurations < 1:
        raise ValueError("'max_configurations' must be >= 1")
    if sm_class._abstract:
        raise InvalidDefinition(_("There are no states or transitions."))
    return _Analyzer(sm_class, max_configurations).analyze()
