"""Static structural analysis of state machines.

The analyzer builds an abstract transition system over *configurations*
(sets of active atomic states, plus the current history snapshots and the
internal ``done.state.*`` queue) and explores it with a bounded breadth-first
search. It reports three kinds of structural issues:

* :data:`FindingKind.UNREACHABLE_STATE` -- states that are never entered in any
  explored configuration;
* :data:`FindingKind.DEAD_CONFIGURATION` -- non-terminal stable configurations
  from which no transition leads anywhere;
* :data:`FindingKind.EVENTLESS_CYCLE` -- cycles made of eventless (automatic)
  transitions that can make the processing loop spin without making progress.

Because guards and validators are *never executed*, every edge is classified
as one of:

* a **certain** edge -- the only enabled option is an unconditional transition,
  so it must be taken on every execution;
* a **possible** edge -- one of the enabled options carries a guard, or a
  history pseudo-state can restore more than one configuration, so the edge
  represents one structurally possible execution.

Reachability is reported on two levels: :data:`Certainty.DEFINITE` (reachable
through certain edges only) and :data:`Certainty.POSSIBLE` (reachable in at
least one structural execution). The analysis is a conservative
over-approximation: a *possible* dead configuration or cycle is **not** a
proof of a business deadlock or a live-lock -- a guard the analyzer cannot
evaluate may prevent it in practice.

Only the machine *class* is inspected; constructing the report never
instantiates the machine and never calls user code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING

from ..callbacks import CallbackGroup
from ..graph import iterate_states
from ..state import HistoryState
from ..state import State
from ..transition import Transition

if TYPE_CHECKING:
    from collections.abc import Iterable

_DEFAULT_MAX_CONFIGURATIONS = 10_000


class Certainty(Enum):
    """How strongly the analysis supports a conclusion.

    Attributes:
        DEFINITE: The conclusion follows from unconditional structure only;
            every reachable execution exhibits it.
        POSSIBLE: The conclusion holds in at least one structurally possible
            execution. Guards are treated as unknown, so this is an
            over-approximation, never a business-level proof.
    """

    DEFINITE = "definite"
    POSSIBLE = "possible"


class FindingKind(Enum):
    """Kind of structural issue reported by :func:`analyze_structure`."""

    UNREACHABLE_STATE = "unreachable_state"
    DEAD_CONFIGURATION = "dead_configuration"
    EVENTLESS_CYCLE = "eventless_cycle"


@dataclass(frozen=True)
class PathStep:
    """One microstep on a :class:`Path`.

    Attributes:
        trigger: What fired the step: ``"<initial>"``, an event id,
            ``"eventless"`` for an automatic transition, or a queued internal
            event id such as ``done.state.x``.
        transitions: Human-readable descriptions of the transitions fired in
            this microstep (``source -> target``), in document order.
        target_ids: Active atomic state ids of the configuration after the
            microstep.
    """

    trigger: str
    transitions: tuple[str, ...]
    target_ids: tuple[str, ...]


@dataclass(frozen=True)
class Path:
    """An event/transition path leading to (or around) a finding.

    Attributes:
        steps: Ordered microsteps from the initial configuration.
        is_shortest: ``True`` when the analyzer guarantees the path is a
            shortest one (in microsteps). Cycle reports describe one repeating
            loop rather than a shortest route, so they carry ``False`` and an
            explicit ``note``.
        note: Human-readable qualifier explaining how to read the path.
    """

    steps: tuple[PathStep, ...] = ()
    is_shortest: bool = True
    note: str = ""

    def __str__(self) -> str:
        if not self.steps:
            return "<empty path>"
        return " -> ".join(
            f"{step.trigger}[{'; '.join(step.transitions)}]" for step in self.steps
        )


@dataclass
class Finding:
    """A single structural issue.

    Attributes:
        kind: What was detected.
        certainty: :data:`Certainty.DEFINITE` or :data:`Certainty.POSSIBLE`.
        state_ids: State ids involved. For an unreachable state it is the
            single state id; for a dead configuration or a cycle it is the
            active atomic states of the configuration.
        path: A path that witnesses the issue. Unreachable states have no
            witness path (there is none by definition).
        detail: Short human-readable explanation.
    """

    kind: FindingKind
    certainty: Certainty
    state_ids: tuple[str, ...]
    path: Path = field(default_factory=Path)
    detail: str = ""


@dataclass
class StructureReport:
    """Result of :func:`analyze_structure`.

    Attributes:
        findings: All detected issues, grouped per configuration/state.
        reachable_states: Ids of atomic states reached in at least one
            structural execution.
        definite_reachable_states: Ids of atomic states reached through
            unconditional edges only.
        configurations_explored: Number of distinct configurations expanded.
        truncated: ``True`` when ``max_configurations`` was reached before the
            search completed. When truncated, the findings are still valid
            (no false positives are introduced) but may be incomplete.
        max_configurations: The configured exploration bound.
        note: Global caveat about the nature of the approximation.
    """

    findings: list[Finding] = field(default_factory=list)
    reachable_states: set[str] = field(default_factory=set)
    definite_reachable_states: set[str] = field(default_factory=set)
    configurations_explored: int = 0
    truncated: bool = False
    max_configurations: int = _DEFAULT_MAX_CONFIGURATIONS
    note: str = (
        "Static, structural analysis only. Guarded transitions are treated as "
        "unknown; 'possible' results are not proofs of business deadlock or "
        "livelock."
    )

    @property
    def unreachable_states(self) -> list[Finding]:
        """Findings of kind :data:`FindingKind.UNREACHABLE_STATE`."""
        return [f for f in self.findings if f.kind is FindingKind.UNREACHABLE_STATE]

    @property
    def dead_configurations(self) -> list[Finding]:
        """Findings of kind :data:`FindingKind.DEAD_CONFIGURATION`."""
        return [f for f in self.findings if f.kind is FindingKind.DEAD_CONFIGURATION]

    @property
    def eventless_cycles(self) -> list[Finding]:
        """Findings of kind :data:`FindingKind.EVENTLESS_CYCLE`."""
        return [f for f in self.findings if f.kind is FindingKind.EVENTLESS_CYCLE]


@dataclass(frozen=True)
class _Node:
    """An abstract machine configuration.

    Attributes:
        leaves: Active atomic states.
        snapshots: Mapping of history pseudo-state id to the remembered atomic
            states. One entry per history pseudo-state that may be consulted.
        pending: FIFO of internal events (``done.state.*``) to be processed
            before the next external event.
    """

    leaves: frozenset[State]
    snapshots: frozenset[tuple[str, tuple[State, ...]]]
    pending: tuple[str, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(s.id for s in self.leaves))


@dataclass(frozen=True)
class _Edge:
    target: _Node
    trigger: str
    fired: tuple[Transition, ...]
    certain: bool
    is_eventless: bool


@dataclass(frozen=True)
class _Option:
    """One concrete entry outcome of a transition.

    ``certain`` is ``False`` when the outcome is only one of several possible
    histories restored by a history pseudo-state.
    """

    leaves: frozenset[State]
    certain: bool


@dataclass
class _VisitedNode:
    node: _Node
    parent: _VisitedNode | None = None
    edge: _Edge | None = None
    depth: int = 0


def analyze_structure(
    machine_cls: type,
    max_configurations: int = _DEFAULT_MAX_CONFIGURATIONS,
) -> StructureReport:
    """Statically analyze a state machine class.

    The machine class is inspected as-is; it is never instantiated and none of
    its guards, validators, actions, entry or exit callbacks are executed.
    Unguarded edges are treated as definite; guarded edges (including
    ``validators``) are treated as edges with an unknown condition and only
    contribute *possible* results.

    Args:
        machine_cls: A ``StateChart``/``StateMachine`` subclass (the class, not
            an instance).
        max_configurations: Upper bound on the number of distinct
            configurations expanded. When the bound is reached the report is
            marked :attr:`StructureReport.truncated` and contains the findings
            known so far.

    Returns:
        A :class:`StructureReport` with reachable states, dead configurations,
        eventless cycles and witness paths.
    """
    return _Analyzer(machine_cls, max_configurations=max_configurations).run()


__all__ = [
    "Certainty",
    "Finding",
    "FindingKind",
    "Path",
    "PathStep",
    "StructureReport",
    "analyze_structure",
]


class _Analyzer:
    """Builds and explores the abstract configuration graph of one machine."""

    def __init__(self, machine_cls: type, max_configurations: int) -> None:
        self.machine_cls = machine_cls
        self.max_configurations = max_configurations
        self.all_states: list[State] = list(iterate_states(machine_cls.states))
        self.atomic_states = [s for s in self.all_states if s.is_atomic]
        self.events: list[str] = [str(e) for e in machine_cls._events]
        self.truncated = False

        # BFS bookkeeping
        self.visited: dict[_Node, _VisitedNode] = {}
        self.queue: deque[_Node] = deque()
        # Edges keyed by (source, target) so a cycle/path can be reconstructed.
        self.out_edges: dict[_Node, list[_Edge]] = {}
        self.eventless_edges: dict[_Node, list[_Edge]] = {}
        # Nodes from which a definite (unconditional) path exists.
        self.definite: set[_Node] = set()

    # ------------------------------------------------------------------ run

    def run(self) -> StructureReport:
        root_node = self._start_node()
        self._register(root_node, parent=None, edge=None, depth=0)
        self.definite.add(root_node)
        self.queue.append(root_node)

        while self.queue and len(self.visited) <= self.max_configurations:
            node = self.queue.popleft()
            self._expand(node)

        self.truncated = bool(self.queue)

        return self._build_report()

    # ------------------------------------------------------------- start/init

    def _start_node(self) -> _Node:
        initial_values = getattr(self.machine_cls, "start_configuration_values", []) or []
        if initial_values:
            roots = [self.machine_cls.states_map[v] for v in initial_values]
        else:
            roots = [self.machine_cls.initial_state]

        leaves, _ = self._descend_to_leaves(roots, within=None)
        snapshots = self._initial_snapshots(frozenset(leaves))
        # The BFS expands this node and naturally follows the unconditional
        # eventless/done cascade of machine activation via certain edges.
        return self._make_node(frozenset(leaves), snapshots, ())

    def _initial_snapshots(
        self, leaves: frozenset[State]
    ) -> frozenset[tuple[str, tuple[State, ...]]]:
        result: list[tuple[str, tuple[State, ...]]] = []
        for state in self.all_states:
            for history in state.history:
                result.append((history.id, self._snapshot_contents(leaves, state, history)))
        return frozenset(result)

    # ------------------------------------------------------------- expansion

    def _expand(self, node: _Node) -> None:
        """Generate one level of microsteps from *node*.

        Unstable nodes (eventless transition enabled, or a pending internal
        event) only offer automatic/internal steps. Stable nodes additionally
        offer one step per declared external event.
        """
        edges = self._microstep_edges(
            node, trigger="eventless", predicate=lambda t: t.is_eventless
        )
        if not edges and node.pending:
            head = node.pending[0]
            edges = self._microstep_edges(
                node, trigger=head, predicate=lambda t, e=head: t.match(e)
            )
            # No transition matches the head internal event: it is consumed.
            if not edges:
                dropped = self._make_node(node.leaves, node.snapshots, node.pending[1:])
                edges = [
                    _Edge(
                        target=dropped,
                        trigger=head,
                        fired=(),
                        certain=True,
                        is_eventless=False,
                    )
                ]

        self._record_edges(node, edges)

        if not edges and not node.pending:
            for event in self.events:
                event_edges = self._microstep_edges(
                    node, trigger=event, predicate=lambda t, e=event: t.match(e)
                )
                self._record_edges(node, event_edges)

    def _record_edges(self, source: _Node, edges: list[_Edge]) -> None:
        if not edges:
            return
        source_info = self.visited[source]
        bucket = self.out_edges.setdefault(source, [])
        for edge in edges:
            bucket.append(edge)
            if edge.is_eventless:
                self.eventless_edges.setdefault(source, []).append(edge)
            info = self._register(
                edge.target, parent=source_info, edge=edge, depth=source_info.depth + 1
            )
            if edge.certain and source in self.definite:
                self.definite.add(info.node)

    def _register(
        self,
        node: _Node,
        parent: _VisitedNode | None,
        edge: _Edge | None,
        depth: int,
    ) -> _VisitedNode:
        info = self.visited.get(node)
        if info is None:
            info = _VisitedNode(node=node, parent=parent, edge=edge, depth=depth)
            self.visited[node] = info
            self.queue.append(node)
        return info

    def _microstep_edges(self, node: _Node, trigger: str, predicate) -> list[_Edge]:
        """Enumerate all non-conflicting microstep outcomes for a trigger.

        Returns one :class:`_Edge` per distinct target configuration. Edges are
        ``certain`` only when every atomic state had a single unguarded enabled
        transition and no history restoration was ambiguous.
        """
        per_leaf: list[list[tuple[Transition, _Option]]] = []
        forced: list[tuple[Transition, _Option]] = []
        leaves = sorted(node.leaves, key=lambda s: s.document_order)
        for leaf in leaves:
            choices: list[tuple[Transition, _Option]] = []
            leaf_forced: tuple[Transition, _Option] | None = None
            search_chain = [leaf, *leaf.ancestors()]
            # done.state.* transitions declared on a finalized compound are
            # collected once as a compound-level choice below, not per leaf.
            for state in search_chain:
                for transition in state.transitions:
                    if transition.initial or not predicate(transition):
                        continue
                    options = self._transition_options(transition, node)
                    guarded = self._is_guarded(transition)
                    for option in options:
                        choice = (
                            transition,
                            _Option(leaves=option.leaves, certain=option.certain and not guarded),
                        )
                        choices.append(choice)
                    if leaf_forced is None and not guarded and options:
                        leaf_forced = (
                            transition,
                            _Option(leaves=options[0].leaves, certain=options[0].certain),
                        )
            if choices:
                per_leaf.append(choices)
                if leaf_forced is not None:
                    forced.append(leaf_forced)

        # Compound/parallel-level choices: a ``done.state.*`` transition whose
        # source is not any leaf's ancestor (the finalized compound itself).
        compound_choices: list[tuple[Transition, _Option]] = []
        compound_forced: tuple[Transition, _Option] | None = None
        for state in self.all_states:
            if state.is_atomic:
                continue
            for transition in state.transitions:
                if transition.initial or not predicate(transition):
                    continue
                if any(state is s for leaf in leaves for s in [leaf, *leaf.ancestors()]):
                    continue
                head = node.pending[0] if node.pending else None
                if not self._done_source_enabled(state, head, node):
                    continue
                for option in self._transition_options(transition, node):
                    guarded = self._is_guarded(transition)
                    compound_choices.append(
                        (
                            transition,
                            _Option(leaves=option.leaves, certain=option.certain and not guarded),
                        )
                    )
                if compound_forced is None and not self._is_guarded(transition):
                    compound_forced = (transition, self._transition_options(transition, node)[0])

        if not per_leaf and not compound_choices:
            return []

        choice_groups: list[list[tuple[Transition, _Option] | None]] = [
            [None, *choices] for choices in per_leaf
        ]
        if compound_choices:
            choice_groups.append([None, *compound_choices])

        combinations: list[tuple[tuple[Transition, _Option], ...]] = []
        for raw_combo in product(*choice_groups):
            combo = tuple(choice for choice in raw_combo if choice is not None)
            if not combo:
                continue
            if self._compatible(combo, node):
                combinations.append(combo)

        if compound_forced is not None:
            # A compound-level done transition preempts every leaf choice.
            forced = [compound_forced]
        return self._edges_from_combinations(node, trigger, combinations, forced)

    def _edges_from_combinations(
        self,
        node: _Node,
        trigger: str,
        combinations: list[tuple[tuple[Transition, _Option], ...]],
        forced: list[tuple[Transition, _Option]],
    ) -> list[_Edge]:
        by_target: dict[_Node, tuple[bool, tuple[Transition, ...]]] = {}
        is_eventless = trigger == "eventless"
        forced_key = {(id(t), o.leaves) for t, o in forced}

        for combo in combinations:
            transitions = self._dedup_transitions([t for t, _ in combo])
            options = {id(t): option for t, option in combo}
            certain = all(
                options[id(t)].certain and (id(t), options[id(t)].leaves) in forced_key
                for t in transitions
            )
            consumed = (trigger,) if node.pending and trigger == node.pending[0] else ()
            target = self._apply_combo(node, transitions, options, consumed=consumed)
            witness = tuple(transitions)
            existing = by_target.get(target)
            if existing is None:
                by_target[target] = (certain, witness)
            else:
                by_target[target] = (existing[0] and certain, existing[1])

        return [
            _Edge(
                target=target,
                trigger=trigger,
                fired=witness,
                certain=certain,
                is_eventless=is_eventless,
            )
            for target, (certain, witness) in by_target.items()
        ]

    # ------------------------------------------------ transition applicability

    @staticmethod
    def _is_guarded(transition: Transition) -> bool:
        """Whether *transition* carries a guard the analyzer must not evaluate.

        Both ``cond``/``unless`` conditions and ``validators`` make the edge
        conditional, so it is treated as an unknown rather than a definite edge.
        Convention-based specs (``on_enter_state`` and friends) are never guards.
        """
        groups = (CallbackGroup.COND, CallbackGroup.VALIDATOR)
        return any(
            spec.group in groups
            and not spec.is_convention
            for spec in transition._specs
        )

    def _compatible(
        self,
        combo: tuple[tuple[Transition, _Option], ...],
        node: _Node,
    ) -> bool:
        """Check the SCXML exit-set conflict rule for a candidate microstep."""
        exits = [self._exit_leaves(t, option.leaves, node) for t, option in combo]
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if exits[i] & exits[j]:
                    return False
        return True

    @staticmethod
    def _dedup_transitions(transitions: list[Transition]) -> list[Transition]:
        """Drop repeated transitions (one transition chosen by several leaves)."""
        seen: set[int] = set()
        result: list[Transition] = []
        for transition in transitions:
            if id(transition) not in seen:
                seen.add(id(transition))
                result.append(transition)
        return result

    def _done_source_enabled(
        self,
        source: State,
        head_event: str | None,
        node: _Node,
    ) -> bool:
        """Whether a ``done.state.*`` transition on *source* can be selected.

        The source compound/parallel must be in its final configuration and the
        pending internal event must announce precisely that state.
        """
        if head_event is None or head_event != f"done.state.{source.id}":
            return False
        return self._in_final_state(source, node.leaves)

    # --------------------------------------------------------- option resolution

    def _transition_options(self, transition: Transition, node: _Node) -> list[_Option]:
        """All concrete leaf sets entered when *transition* is taken.

        Targetless transitions keep the source leaves. History targets expand
        to every structurally possible restored content, plus the history's
        default content when no content has been remembered yet.
        """
        if not transition.targets:
            return [_Option(frozenset(node.leaves), certain=True)]

        option_lists: list[list[frozenset[State]]] = []
        for target in transition.targets:
            if isinstance(target, HistoryState):
                option_lists.append(self._history_contents(target, node))
            else:
                leaves, _ = self._descend_to_leaves([target], within=None)
                option_lists.append([frozenset(leaves)])

        combined: list[frozenset[State]] = []
        for targets in product(*option_lists):
            merged: set[State] = set()
            for target_leaves in targets:
                merged.update(target_leaves)
            combined.append(frozenset(merged))

        unique = self._unique(combined)
        certain = len(unique) == 1
        return [_Option(leaves=leaves, certain=certain) for leaves in unique]

    def _history_contents(self, history: HistoryState, node: _Node) -> list[frozenset[State]]:
        remembered = {content for hid, content in node.snapshots if hid == history.id}
        candidates: list[frozenset[State]] = []
        if remembered and all(content for content in remembered):
            for content in remembered:
                leaves: set[State] = set()
                for state in content:
                    descended, _ = self._descend_to_leaves([state], within=history.parent)
                    leaves.update(descended)
                candidates.append(frozenset(leaves))
        else:
            for default in history.transitions:
                for option in self._transition_options(default, node):
                    candidates.append(option.leaves)
        return self._unique(candidates)

    # ------------------------------------------------------------- entry/exit

    @staticmethod
    def _make_node(
        leaves: frozenset[State],
        snapshots: frozenset[tuple[str, tuple[State, ...]]],
        pending: tuple[str, ...],
    ) -> _Node:
        return _Node(
            leaves=leaves,
            snapshots=frozenset(snapshots),
            pending=tuple(dict.fromkeys(pending)),
        )

    def _descend_to_leaves(
        self,
        states: Iterable[State],
        within: State | None,
    ) -> tuple[set[State], set[State]]:
        """Leaf states and compound/parallel ancestors entered by default entry.

        Mirrors the engine's default entry: entering a compound follows its
        initial transition recursively; entering a parallel enters every region
        recursively. Ancestors strictly below *within* are not added (an
        internal transition into a descendant keeps its enclosing states).
        """
        leaves: set[State] = set()
        entered: set[State] = set()

        def add(state: State) -> None:
            if state is not within and state.parent is not None:
                entered.add(state)
            if state.parallel:
                for child in state.states:
                    add(child)
            elif state.is_compound:
                initial = next(t for t in state.transitions if t.initial)
                for target in initial.targets:
                    add(target)
            else:
                leaves.add(state)

        for state in states:
            add(state)
        return leaves, entered

    def _domain(self, transition: Transition, concrete_leaves: frozenset[State]) -> State | None:
        """Least common compound ancestor of source and concrete targets."""
        if transition.internal and transition.source.is_compound:
            if all(self._is_or_under(s, transition.source) for s in concrete_leaves):
                return transition.source
        if (
            transition.internal
            and transition.is_self
            and transition.target
            and transition.target.is_atomic
        ):
            return transition.source

        states = [transition.source, *concrete_leaves]
        head, *tail = states
        for ancestor in head.ancestors():
            if ancestor.is_compound and all(self._is_or_under(s, ancestor) for s in tail):
                return ancestor
        return None

    @staticmethod
    def _is_or_under(state: State, ancestor: State) -> bool:
        return state is ancestor or state.is_descendant(ancestor)

    def _exit_leaves(
        self,
        transition: Transition,
        concrete_leaves: frozenset[State],
        node: _Node,
    ) -> frozenset[State]:
        domain = self._domain(transition, concrete_leaves)
        if domain is None:
            return frozenset(node.leaves)
        return frozenset(leaf for leaf in node.leaves if leaf.is_descendant(domain))

    def _apply_combo(
        self,
        node: _Node,
        transitions: list[Transition],
        options: dict[int, _Option],
        consumed: tuple[str, ...] = (),
    ) -> _Node:
        exits: set[State] = set()
        domains: set[State | None] = set()
        for transition in transitions:
            option = options[id(transition)]
            domains.add(self._domain(transition, option.leaves))
            exits.update(self._exit_leaves(transition, option.leaves, node))

        remaining = frozenset(leaf for leaf in node.leaves if leaf not in exits)
        snapshots = self._update_snapshots(node, exits)

        entered_leaves: set[State] = set()
        for transition in transitions:
            option = options[id(transition)]
            domain = self._domain(transition, option.leaves)
            entered_leaves.update(self._entry_leaves(option.leaves, domain))

        leaves = frozenset(remaining | entered_leaves)
        remaining_pending = tuple(e for e in node.pending if e not in consumed)
        pending = self._done_events(remaining_pending, node.leaves, exits, leaves)
        return self._make_node(leaves, snapshots, pending)

    def _entry_leaves(
        self,
        concrete_leaves: frozenset[State],
        domain: State | None,
    ) -> set[State]:
        leaves: set[State] = set()
        for target in concrete_leaves:
            default_leaves, _ = self._descend_to_leaves([target], within=domain)
            leaves.update(default_leaves)

            # Walk up to the domain, filling every parallel region crossed.
            for ancestor in target.ancestors(parent=domain):
                if ancestor.parallel:
                    for region in ancestor.states:
                        if any(self._is_or_under(s, region) for s in leaves):
                            continue
                        region_leaves, _ = self._descend_to_leaves([region], within=domain)
                        leaves.update(region_leaves)
        return leaves

    # ------------------------------------------------------------- snapshots

    @staticmethod
    def _snapshot_contents(
        leaves: Iterable[State],
        compound: State,
        history: HistoryState,
    ) -> tuple[State, ...]:
        if history.type.is_deep:
            content = [s for s in leaves if s.is_descendant(compound)]
        else:
            content = [s for s in leaves if s.parent is compound]
        return tuple(sorted(content, key=lambda s: s.document_order))

    def _update_snapshots(
        self, node: _Node, exits: frozenset[State]
    ) -> frozenset[tuple[str, tuple[State, ...]]]:
        active = node.leaves
        updated: dict[str, tuple[State, ...]] = dict(node.snapshots)
        exited_compounds = {
            ancestor
            for leaf in exits
            for ancestor in leaf.ancestors()
            if ancestor.history
        }
        for compound in exited_compounds:
            for history in compound.history:
                updated[history.id] = self._snapshot_contents(active, compound, history)
        return frozenset(updated.items())

    # ------------------------------------------------------------- done events

    def _done_events(
        self,
        previous_pending: tuple[str, ...],
        previous_leaves: frozenset[State],
        exits: frozenset[State],
        new_leaves: frozenset[State],
    ) -> tuple[str, ...]:
        """Queue ``done.state.<parent>`` for compounds newly entered in final.

        A compound done event is raised only when its previous configuration
        was not already final. Root-level finals (parent is the machine, not a
        compound) terminate the machine instead of raising an event, mirroring
        the engine's ``_handle_final_state``.
        """
        events: list[str] = list(previous_pending)
        finalized_parents: list[State] = []
        for state in self.all_states:
            if state.is_atomic:
                continue
            if not self._in_final_state(state, new_leaves):
                continue
            was_final = self._in_final_state(state, previous_leaves)
            if not was_final:
                finalized_parents.append(state)

        for parent in finalized_parents:
            event = f"done.state.{parent.id}"
            if event not in events:
                events.append(event)
        return tuple(events)

    def _in_final_state(self, state: State, leaves: frozenset[State]) -> bool:
        if state.is_compound:
            return any(s.final for s in state.states if s in leaves)
        if state.parallel:
            return all(self._region_is_final(child, leaves) for child in state.states)
        return False

    def _region_is_final(self, region: State, leaves: frozenset[State]) -> bool:
        if region.is_compound:
            return any(s.final for s in region.states if s in leaves)
        if region.parallel:
            return all(self._region_is_final(child, leaves) for child in region.states)
        return region in leaves and region.final

    def _is_terminal(self, node: _Node) -> bool:
        """Whether the machine terminates in *node* (root final configuration)."""
        for leaf in node.leaves:
            if leaf.final and leaf.parent is None:
                return True
        for root in self.machine_cls.states:
            if root.parallel and all(self._is_under_root(s, root) for s in node.leaves):
                if self._in_final_state(root, node.leaves):
                    return True
        return False

    @staticmethod
    def _is_under_root(state: State, root: State) -> bool:
        return state is root or state.is_descendant(root)

    # --------------------------------------------------------------- reporting

    def _build_report(self) -> StructureReport:
        reachable_leaves: set[State] = set()
        definite_leaves: set[State] = set()
        for node in self.visited:
            reachable_leaves.update(node.leaves)
            if node in self.definite:
                definite_leaves.update(node.leaves)

        findings: list[Finding] = []
        findings.extend(self._unreachable_findings(reachable_leaves))
        findings.extend(self._dead_findings())
        findings.extend(self._cycle_findings())

        return StructureReport(
            findings=findings,
            reachable_states={s.id for s in reachable_leaves if s not in self._history_states()},
            definite_reachable_states={s.id for s in definite_leaves},
            configurations_explored=len(self.visited),
            truncated=self.truncated,
            max_configurations=self.max_configurations,
        )

    def _history_states(self) -> set[State]:
        return {s for s in self.all_states if isinstance(s, HistoryState)}

    def _unreachable_findings(self, reachable_leaves: set[State]) -> list[Finding]:
        result: list[Finding] = []
        reachable = set(reachable_leaves)
        # A compound/parallel is reachable when any descendant leaf is.
        for state in self.all_states:
            if isinstance(state, HistoryState):
                continue
            if state.is_atomic:
                active = state in reachable
            else:
                active = any(self._is_or_under(leaf, state) for leaf in reachable)
            if not active:
                result.append(
                    Finding(
                        kind=FindingKind.UNREACHABLE_STATE,
                        certainty=Certainty.DEFINITE,
                        state_ids=(state.id,),
                        detail="No structural path enters this state.",
                    )
                )
        return result

    def _dead_findings(self) -> list[Finding]:
        result: list[Finding] = []
        for node in self.visited:
            edges = self.out_edges.get(node, [])
            if edges or node.pending or self._is_terminal(node):
                continue
            certainty = Certainty.DEFINITE if node in self.definite else Certainty.POSSIBLE
            result.append(
                Finding(
                    kind=FindingKind.DEAD_CONFIGURATION,
                    certainty=certainty,
                    state_ids=node.ids,
                    path=self._path_to(node),
                    detail=(
                        "Configuration has no outgoing transition and is not final. "
                        "A structural trap, not a proof of business deadlock."
                    ),
                )
            )
        return result

    def _cycle_findings(self) -> list[Finding]:
        return self._tarjan_cycles()

    def _tarjan_cycles(self) -> list[Finding]:
        """Find strongly connected components made of eventless edges."""
        index: dict[_Node, int] = {}
        lowlink: dict[_Node, int] = {}
        stack: list[_Node] = []
        on_stack: set[_Node] = set()
        counter = 0
        sccs: list[set[_Node]] = []

        nodes = [info.node for info in self.visited.values()]

        def visit(start: _Node) -> None:
            nonlocal counter
            work: list[tuple[_Node, int]] = [(start, 0)]
            index[start] = lowlink[start] = counter
            counter += 1
            stack.append(start)
            on_stack.add(start)

            while work:
                node, pos = work[-1]
                neighbors = [
                    edge.target
                    for edge in self.eventless_edges.get(node, [])
                    if edge.target in self.visited
                ]
                if pos < len(neighbors):
                    work[-1] = (node, pos + 1)
                    nxt = neighbors[pos]
                    if nxt not in index:
                        index[nxt] = lowlink[nxt] = counter
                        counter += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append((nxt, 0))
                    elif nxt in on_stack:
                        lowlink[node] = min(lowlink[node], index[nxt])
                else:
                    work.pop()
                    if work:
                        parent = work[-1][0]
                        lowlink[parent] = min(lowlink[parent], lowlink[node])
                    if lowlink[node] == index[node]:
                        component: set[_Node] = set()
                        while True:
                            member = stack.pop()
                            on_stack.discard(member)
                            component.add(member)
                            if member is node:
                                break
                        sccs.append(component)

        for node in nodes:
            if node not in index:
                visit(node)

        findings: list[Finding] = []
        for component in sccs:
            finding = self._cycle_finding(component)
            if finding is not None:
                findings.append(finding)
        return findings

    def _cycle_finding(self, component: set[_Node]) -> Finding | None:
        internal = [
            edge
            for node in component
            for edge in self.eventless_edges.get(node, [])
            if edge.target in component
        ]
        # A singleton SCC is only a cycle when it has an eventless self-edge.
        if len(component) == 1 and not internal:
            return None

        entry = min(component, key=lambda n: self.visited[n].depth)
        definite = all(edge.certain for edge in internal) and entry in self.definite
        loop = self._describe_loop(component, entry)
        path = self._path_to(entry)
        cycle_path = Path(
            steps=path.steps + loop,
            is_shortest=False,
            note=(
                "Path to the cycle is a shortest entry path; the trailing steps "
                "are one repeating eventless loop (not claimed to be shortest)."
            ),
        )
        return Finding(
            kind=FindingKind.EVENTLESS_CYCLE,
            certainty=Certainty.DEFINITE if definite else Certainty.POSSIBLE,
            state_ids=entry.ids,
            path=cycle_path,
            detail="Eventless transitions can keep firing around this loop.",
        )

    def _describe_loop(self, component: set[_Node], entry: _Node) -> tuple[PathStep, ...]:
        steps: list[PathStep] = []
        current = entry
        seen: set[_Node] = set()
        while current not in seen:
            seen.add(current)
            edge = next(
                edge
                for edge in self.eventless_edges.get(current, [])
                if edge.target in component
            )
            steps.append(
                PathStep(
                    trigger="eventless",
                    transitions=tuple(self._describe(t) for t in edge.fired),
                    target_ids=edge.target.ids,
                )
            )
            current = edge.target
        return tuple(steps)

    # ------------------------------------------------------------- path helpers

    def _path_to(self, target: _Node) -> Path:
        steps: list[PathStep] = []
        info = self.visited.get(target)
        while info is not None and info.edge is not None:
            edge = info.edge
            steps.append(
                PathStep(
                    trigger=edge.trigger,
                    transitions=tuple(self._describe(t) for t in edge.fired),
                    target_ids=info.node.ids,
                )
            )
            info = info.parent
        steps.reverse()
        return Path(steps=tuple(steps), is_shortest=True)

    def _describe(self, transition: Transition) -> str:
        target = transition.target
        target_id = target.id if target is not None else transition.source.id
        return f"{transition.source.id} -> {target_id}"


    @staticmethod
    def _unique(items: list[frozenset[State]]) -> list[frozenset[State]]:
        seen: set[frozenset[State]] = set()
        result: list[frozenset[State]] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
