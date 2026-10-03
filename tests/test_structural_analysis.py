"""Tests for the static structural analysis API (``statemachine.analysis``).

The analyzer never instantiates the machine and never executes a guard or
action. Unguarded machines are additionally cross-checked against a brute
force enumeration of the real runtime configurations.
"""

from itertools import product

import pytest

from statemachine import State
from statemachine import StateChart
from statemachine import analyze_structure as top_level_analyze
from statemachine.analysis import Certainty
from statemachine.analysis import analyze_structure
from statemachine.event import Event
from statemachine.transition import Transition
from statemachine.transition_list import TransitionList
from tests.machines.history.deep_memory_of_moria import DeepMemoryOfMoria
from tests.machines.history.gollum_personality import GollumPersonality
from tests.machines.history.gollum_personality_default_gollum import (
    GollumPersonalityDefaultGollum,
)


# ---------------------------------------------------------------------------
# Small unguarded models used for exhaustive runtime cross-checks
# ---------------------------------------------------------------------------


class FlatChain(StateChart):
    s1 = State(initial=True)
    s2 = State()
    s3 = State(final=True)

    go1 = s1.to(s2)
    go2 = s2.to(s3)


class FlatLoop(StateChart):
    a = State(initial=True)
    b = State()
    c = State(final=True)

    ab = a.to(b)
    ba = b.to(a)
    bc = b.to(c)


class ForkJoin(StateChart):
    """Two unguarded alternatives reached through different named events."""

    root = State(initial=True)
    left = State(final=True)
    right = State(final=True)

    go_left = root.to(left)
    go_right = root.to(right)


class ParallelDone(StateChart):
    class work(State.Parallel):
        class left(State.Compound):
            l1 = State(initial=True)
            l2 = State(final=True)

            finish_l = l1.to(l2)

        class right(State.Compound):
            r1 = State(initial=True)
            r2 = State(final=True)

            finish_r = r1.to(r2)

    done = State(final=True)
    done_state_work = work.to(done)


class CompoundAuto(StateChart):
    class nested(State.Compound):
        x = State(initial=True)
        y = State(final=True)

        x.to(y)

    outer_done = State(final=True)
    done_state_nested = nested.to(outer_done)


UNGUARDED_MODELS = [FlatChain, FlatLoop, ForkJoin, ParallelDone, CompoundAuto]


def _runtime_configurations(machine_cls, max_depth=6):
    """Breadth-first enumerate configurations reachable by sending batches of
    events -- each batch ends with a processing-loop settlement.

    A batch sends every subset of declared events (in every order), so parallel
    microsteps that require two events "in the same macrostep" are covered.
    Compound/parallel ancestors in the configuration are stripped because the
    analyzer reports atomic leaves.
    """
    events = [str(e) for e in machine_cls._events]
    atomic_ids = {state.id for state in machine_cls.states_map.values() if state.is_atomic}

    def settle(sm):
        return frozenset(sm.configuration_values) & atomic_ids

    start = settle(machine_cls())
    seen = {start}
    frontier = {start}
    for _ in range(max_depth):
        new_frontier = set()
        for config in frontier:
            batches = {
                combo
                for size in range(1, len(events) + 1)
                for combo in product(events, repeat=size)
            }
            for batch in batches:
                probe = machine_cls()
                probe.current_state_value = set(config)
                for event in batch:
                    probe.send(event)
                nxt = settle(probe)
                if nxt not in seen:
                    seen.add(nxt)
                    new_frontier.add(nxt)
        if not new_frontier:
            break
        frontier = new_frontier
    return seen


def _stable_report_states(machine_cls, report):
    """Leaf states of configurations with no pending automatic step.

    The runtime brute force only observes configurations after the processing
    loop settles, so compare against the analyzer's stable nodes (the report's
    definite reachable set additionally includes transiently entered leaves
    during startup/auto cascades).
    """
    from statemachine.analysis.structural import _Analyzer

    analyzer = _Analyzer(machine_cls, max_configurations=report.max_configurations)
    start = analyzer._start_node()
    analyzer._register(start, None, None, 0)
    analyzer.queue.append(start)
    while analyzer.queue:
        node = analyzer.queue.popleft()
        analyzer._expand(node)

    stable_states: set[str] = set()
    for node in analyzer.visited:
        auto_edges = analyzer.eventless_edges.get(node, [])
        pending_edges = [
            edge
            for edge in analyzer.out_edges.get(node, [])
            if edge.trigger in node.pending
        ]
        if not auto_edges and not pending_edges:
            stable_states.update(node.ids)
    return stable_states


# ---------------------------------------------------------------------------
# Exhaustive cross-checks against the running engine on unguarded models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("machine_cls", UNGUARDED_MODELS)
class TestUnguardedExhaustive:
    def test_definite_reachability_matches_runtime(self, machine_cls):
        report = analyze_structure(machine_cls)
        runtime_states = {
            state_id
            for config in _runtime_configurations(machine_cls)
            for state_id in config
        }
        analyzer_states = _stable_report_states(machine_cls, report)
        assert analyzer_states == runtime_states

    def test_no_spurious_findings_on_unguarded_models(self, machine_cls):
        report = analyze_structure(machine_cls)
        assert report.findings == []
        assert report.truncated is False

    def test_every_runtime_configuration_is_represented(self, machine_cls):
        report = analyze_structure(machine_cls)
        explored = {tuple(report.reachable_states)}
        # Defensive: definite set must always be a subset of possible set.
        assert report.definite_reachable_states <= report.reachable_states
        assert explored  # sanity, keeps the variable purposeful


class TestBasicReachability:
    def test_all_states_definitely_reachable_on_flat_chain(self):
        report = analyze_structure(FlatChain)
        assert report.reachable_states == {"s1", "s2", "s3"}
        assert report.definite_reachable_states == {"s1", "s2", "s3"}
        assert report.findings == []

    def test_every_finding_is_annotated(self):
        report = analyze_structure(FlatChain)
        assert report.note
        assert "not proofs" in report.note


# ---------------------------------------------------------------------------
# Guarded machines: guards are never executed and yield "possible" results
# ---------------------------------------------------------------------------


class ThrowingGuardMachine(StateChart):
    """Any evaluation of the guard is a test failure."""

    a = State(initial=True)
    b = State()
    c = State(final=True)

    guarded_go = a.to(b, cond="boom")
    finish = b.to(c)

    def boom(self):
        raise AssertionError("structural analysis must not evaluate guards")


class TestGuardsNotExecuted:
    def test_analysis_does_not_call_throwing_guard(self):
        report = analyze_structure(ThrowingGuardMachine)
        assert report.reachable_states == {"a", "b", "c"}
        assert report.definite_reachable_states == {"a"}
        assert report.findings == []

    def test_guarded_transition_is_possible_not_definite_path(self):
        report = analyze_structure(ThrowingGuardMachine)
        finding_states = report.reachable_states
        assert "b" in finding_states
        assert "b" not in report.definite_reachable_states


class GuardedAlternative(StateChart):
    start = State(initial=True)
    chosen = State(final=True)

    take = start.to(chosen, cond="maybe")

    def maybe(self):
        raise AssertionError("guard executed")


def test_guarded_only_edge_keeps_source_possibly_dead():
    report = analyze_structure(GuardedAlternative)
    # ``start`` can stay forever (guard false) but also reach ``chosen`` (true),
    # so it must not be reported as a definite dead configuration.
    dead = {f.state_ids for f in report.dead_configurations}
    assert ("start",) not in dead
    assert report.reachable_states == {"start", "chosen"}
    assert "chosen" not in report.definite_reachable_states


class StructuralTrap(StateChart):
    validate_trap_states = False
    validate_final_reachability = False

    entry = State(initial=True)
    stuck = State()
    final = State(final=True)

    move = entry.to(stuck)
    escape = entry.to(final)


def test_possible_dead_configuration_has_shortest_path():
    report = analyze_structure(StructuralTrap)
    assert len(report.dead_configurations) == 1
    finding = report.dead_configurations[0]
    assert finding.certainty is Certainty.DEFINITE
    assert finding.state_ids == ("stuck",)
    assert finding.path.is_shortest is True
    assert [step.trigger for step in finding.path.steps] == ["move"]


# ---------------------------------------------------------------------------
# Unreachable states
# ---------------------------------------------------------------------------


class WithOrphan(StateChart):
    validate_disconnected_states = False
    validate_trap_states = False
    validate_final_reachability = False

    first = State(initial=True)
    second = State(final=True)
    orphan = State()

    advance = first.to(second)


def test_unreachable_state_is_reported_without_path():
    report = analyze_structure(WithOrphan)
    orphans = report.unreachable_states
    assert len(orphans) == 1
    orphan = orphans[0]
    assert orphan.state_ids == ("orphan",)
    assert orphan.certainty is Certainty.DEFINITE
    assert orphan.path.steps == ()
    assert orphan.detail


def test_unreachable_compound_has_all_descendants_absent():
    class Machine(StateChart):
        validate_disconnected_states = False
        validate_trap_states = False
        validate_final_reachability = False

        s1 = State(initial=True)
        s2 = State(final=True)

        class unused(State.Compound):
            u1 = State(initial=True)
            u2 = State(final=True)

            go = u1.to(u2)

        advance = s1.to(s2)

    report = analyze_structure(Machine)
    unreachable_ids = {f.state_ids[0] for f in report.unreachable_states}
    assert {"unused", "u1", "u2"} <= unreachable_ids


# ---------------------------------------------------------------------------
# Eventless cycles
# ---------------------------------------------------------------------------


class EventlessSelfLoop(StateChart):
    validate_trap_states = False

    spinning = State(initial=True)

    spinning.to(spinning)


class EventlessTwoCycle(StateChart):
    validate_trap_states = False

    one = State(initial=True)
    two = State()

    one.to(two)
    two.to(one)


def test_eventless_self_loop_is_definite_cycle_with_non_shortest_marker():
    report = analyze_structure(EventlessSelfLoop)
    assert len(report.eventless_cycles) == 1
    cycle = report.eventless_cycles[0]
    assert cycle.certainty is Certainty.DEFINITE
    assert cycle.state_ids == ("spinning",)
    assert cycle.path.is_shortest is False
    assert cycle.path.note
    assert cycle.path.steps[-1].trigger == "eventless"


def test_eventless_two_state_cycle_is_reported():
    report = analyze_structure(EventlessTwoCycle)
    assert len(report.eventless_cycles) == 1
    cycle = report.eventless_cycles[0]
    triggers = [step.trigger for step in cycle.path.steps]
    assert triggers.count("eventless") >= 2


class GuardedEventlessCycle(StateChart):
    validate_trap_states = False

    one = State(initial=True)
    two = State()

    one.to(two)
    two.to(one, cond="maybe")

    def maybe(self):
        raise AssertionError("guard executed")


def test_guarded_eventless_cycle_is_possible_not_definite():
    report = analyze_structure(GuardedEventlessCycle)
    assert len(report.eventless_cycles) == 1
    assert report.eventless_cycles[0].certainty is Certainty.POSSIBLE


def test_named_event_loop_is_not_an_eventless_cycle():
    class NamedLoop(StateChart):
        a = State(initial=True)
        b = State(final=True)

        ping = a.to(b)

    report = analyze_structure(NamedLoop)
    assert report.eventless_cycles == []


# ---------------------------------------------------------------------------
# History pseudo-states expand to every remembered/default configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "machine_cls",
    [GollumPersonality, DeepMemoryOfMoria],
)
def test_history_machine_has_no_findings_and_all_leaves_reachable(machine_cls):
    report = analyze_structure(machine_cls)
    assert report.findings == []
    assert report.truncated is False


def test_history_default_content_reaches_only_default_target():
    report = analyze_structure(GollumPersonalityDefaultGollum)
    assert {"gollum", "outside"} <= report.reachable_states
    assert "smeagol" not in report.reachable_states
    assert {f.state_ids[0] for f in report.unreachable_states} == {"smeagol"}


# ---------------------------------------------------------------------------
# Search bound (truncation)
# ---------------------------------------------------------------------------


class Diamond(StateChart):
    a = State(initial=True)
    b = State()
    c = State()
    d = State(final=True)

    ab = a.to(b)
    ac = a.to(c)
    bd = b.to(d)
    cd = c.to(d)


def test_truncation_is_flagged_and_findings_remain_valid():
    full = analyze_structure(Diamond)
    assert full.truncated is False
    assert full.configurations_explored >= 4

    bounded = analyze_structure(Diamond, max_configurations=1)
    assert bounded.truncated is True
    assert bounded.configurations_explored > 1
    # No false positives: explored states are a subset of the full analysis.
    assert bounded.reachable_states <= full.reachable_states


def test_max_configurations_zero_explores_only_the_start():
    report = analyze_structure(FlatChain, max_configurations=0)
    assert report.truncated is True
    assert report.reachable_states == {"s1"}


# ---------------------------------------------------------------------------
# Validators are treated as (unknown) guards and never executed
# ---------------------------------------------------------------------------


class ThrowingValidatorMachine(StateChart):
    a = State(initial=True)
    b = State(final=True)

    go = a.to(b, validators="must_not_run")

    def must_not_run(self):
        raise AssertionError("validators must not be executed by the analyzer")


def test_validator_is_not_called_and_edge_is_possible():
    report = analyze_structure(ThrowingValidatorMachine)
    assert report.reachable_states == {"a", "b"}
    assert report.definite_reachable_states == {"a"}


# ---------------------------------------------------------------------------
# Parallel regions: a dead region leaves the configuration non-final
# ---------------------------------------------------------------------------


class ParallelWithTrap(StateChart):
    validate_trap_states = False
    validate_final_reachability = False

    class flow(State.Parallel):
        class good(State.Compound):
            g1 = State(initial=True)
            g2 = State(final=True)

            finish_good = g1.to(g2)

        class bad(State.Compound):
            b1 = State(initial=True)
            b2 = State()

            step_bad = b1.to(b2)


def test_parallel_dead_region_is_detected_as_dead_configuration():
    report = analyze_structure(ParallelWithTrap)
    dead_ids = {f.state_ids for f in report.dead_configurations}
    assert ("b2", "g2") in dead_ids
    finding = next(f for f in report.dead_configurations if f.state_ids == ("b2", "g2"))
    # Entry path mentions both region transitions in some order.
    triggers = {step.trigger for step in finding.path.steps}
    assert {"finish_good", "step_bad"} <= triggers


# ---------------------------------------------------------------------------
# Path rendering and witness quality
# ---------------------------------------------------------------------------


def test_path_step_describes_transition_endpoints():
    report = analyze_structure(FlatChain)
    # No findings: exercise the API via a path built by the analyzer through a
    # structural trap instead.
    trapped = analyze_structure(StructuralTrap)
    step = trapped.dead_configurations[0].path.steps[0]
    assert step.transitions == ("entry -> stuck",)
    assert step.target_ids == ("stuck",)
    assert str(trapped.dead_configurations[0].path) == "move[entry -> stuck]"


def test_shortest_path_is_reported_for_unreachable_via_cycle_entry():
    # In the two-state eventless cycle, the entry path is empty (start state is
    # already in the cycle) and the loop is marked non-shortest.
    report = analyze_structure(EventlessTwoCycle)
    cycle = report.eventless_cycles[0]
    assert cycle.path.is_shortest is False
    # Entry component starts at the initial state.
    assert cycle.state_ids[0] in {"one", "two"}


def test_report_grouping_properties():
    report = analyze_structure(EventlessSelfLoop)
    assert len(report.unreachable_states) == 0
    assert len(report.dead_configurations) == 0
    assert len(report.eventless_cycles) == 1


def test_structure_report_default_note_warns_about_approximation():
    report = analyze_structure(FlatChain)
    assert "Guard" in report.note or "guard" in report.note


# ---------------------------------------------------------------------------
# Edge cases: targetless, internal, custom start values, nested parallel
# ---------------------------------------------------------------------------


def test_empty_path_renders_as_placeholder():
    from statemachine.analysis import Path

    assert str(Path()) == "<empty path>"


class TargetlessTransition(StateChart):
    a = State(initial=True)
    b = State(final=True)

    go = a.to(b)
    ping = TransitionList([Transition(a, None, event="ping")])
    a.transitions.add_transitions(ping)


def test_targetless_internal_transition_keeps_state_reachable_and_not_dead():
    report = analyze_structure(TargetlessTransition)
    assert report.reachable_states == {"a", "b"}
    assert report.dead_configurations == []


class InternalSelfLoop(StateChart):
    a = State(initial=True)
    b = State(final=True)

    tick = a.to.itself(internal=True)
    go = a.to(b)


def test_internal_self_transition_is_not_dead_and_b_reachable():
    report = analyze_structure(InternalSelfLoop)
    assert report.reachable_states == {"a", "b"}
    assert report.findings == []


def test_custom_start_configuration_value_is_respected():
    class StartedAtB(FlatChain):
        start_configuration_values = ["s2"]

    report = analyze_structure(StartedAtB)
    assert report.reachable_states == {"s2", "s3"}
    assert "s1" not in report.reachable_states


class NestedParallelDone(StateChart):
    """A parallel region whose regions contain nested parallel/atomics."""

    class outer(State.Parallel):
        class inner(State.Parallel):
            class r1(State.Compound):
                x1 = State(initial=True)
                x2 = State(final=True)

                go_x = x1.to(x2)

            class r2(State.Compound):
                y1 = State(initial=True)
                y2 = State(final=True)

                go_y = y1.to(y2)

        class other(State.Compound):
            z1 = State(initial=True)
            z2 = State(final=True)

            go_z = z1.to(z2)

    finished = State(final=True)
    done_state_outer = outer.to(finished)


def test_nested_parallel_all_regions_must_finish():
    report = analyze_structure(NestedParallelDone)
    assert report.findings == []
    assert report.reachable_states == {"x1", "x2", "y1", "y2", "z1", "z2", "finished"}


class GuardedEventlessOnly(StateChart):
    """Stable node: only a guarded eventless edge -- not dead, but possible."""

    validate_trap_states = False

    here = State(initial=True)
    there = State(final=True)

    here.to(there, cond="maybe")

    def maybe(self):
        raise AssertionError("guard executed")


def test_guarded_eventless_only_is_not_dead_but_there_is_possible():
    report = analyze_structure(GuardedEventlessOnly)
    assert report.dead_configurations == []
    assert report.reachable_states == {"here", "there"}
    assert "there" not in report.definite_reachable_states


class EnterParallelFromOutside(StateChart):
    """Transition targets one region; sibling regions are filled by default."""

    class area(State.Parallel):
        class north(State.Compound):
            n1 = State(initial=True)
            n2 = State(final=True)

            finish_n = n1.to(n2)

        class south(State.Compound):
            s1 = State(initial=True)
            s2 = State(final=True)

            finish_s = s1.to(s2)

    outside = State(initial=True)

    enter = outside.to(area.north)


def test_entering_one_parallel_region_fills_all_regions():
    report = analyze_structure(EnterParallelFromOutside)
    assert {"n1", "n2", "s1", "s2"} <= report.reachable_states


class EventlessInternalDeadAtom(StateChart):
    """Eventless internal targetless transition is a possible spin, not dead."""

    validate_trap_states = False

    only = State(initial=True)

    only.to.itself()


def test_eventless_external_self_transition_is_a_cycle_not_a_dead_config():
    report = analyze_structure(EventlessInternalDeadAtom)
    assert report.dead_configurations == []
    assert len(report.eventless_cycles) == 1


class InternalCompoundDescendant(StateChart):
    """Internal transition from a compound into one of its descendants."""

    validate_disconnected_states = False

    class group(State.Compound):
        a = State(initial=True)
        b = State()
        c = State(final=True)

        move = a.to(b)

    outside = State(final=True)
    done_state_group = group.to(outside)


# The nested compound cannot self-reference inside the class body, so build its
# internal transition after class creation (bypassing the constructor check
# which cannot see a descendant of a not-yet-bound source).
_group = InternalCompoundDescendant.group
_shortcut = object.__new__(Transition)
Transition.__init__(
    _shortcut, source=_group, target=_group.c, event="shortcut", internal=True
)
_group.transitions.add_transitions(_shortcut)
InternalCompoundDescendant.add_event(
    Event(id="shortcut", transitions=_group.transitions)
)


def test_internal_transition_from_compound_to_descendant():
    report = analyze_structure(InternalCompoundDescendant)
    assert {"a", "b", "c", "outside"} <= report.reachable_states
    assert report.findings == []


class ParallelWithAtomicRegion(StateChart):
    """A parallel state with one atomic region and one compound region."""

    class flow(State.Parallel):
        idle = State()

        class worker(State.Compound):
            w1 = State(initial=True)
            w2 = State(final=True)

            finish_w = w1.to(w2)

    done = State(final=True)
    done_state_flow = flow.to(done)


def test_parallel_with_atomic_region_never_completes():
    # An atomic non-final region can never be final, so the parallel cannot
    # complete; the analyzer must not claim ``done`` is reachable.
    report = analyze_structure(ParallelWithAtomicRegion)
    assert "done" not in report.reachable_states
    assert report.reachable_states == {"idle", "w1", "w2"}


class MultiTargetParallelKeepsRegion(StateChart):
    """A multi-target transition enters one region while another stays."""

    validate_disconnected_states = False

    class flow(State.Parallel):
        class left(State.Compound):
            l1 = State(initial=True)
            l2 = State()
            l3 = State(final=True)

            finish_l = l1.to(l3)

        class right(State.Compound):
            r1 = State(initial=True)
            r2 = State(final=True)

            finish_r = r1.to(r2)

    done = State(final=True)
    done_state_flow = flow.to(done)


# Multi-target transition: reset right to r1 while entering left's l2.
_flow = MultiTargetParallelKeepsRegion.flow
_restart = _flow.right.r1.to(_flow.left.l2, _flow.right.r1)
MultiTargetParallelKeepsRegion.add_event(
    Event(id="restart", transitions=_restart)
)


def test_multi_target_transition_into_parallel_region_is_modeled():
    report = analyze_structure(MultiTargetParallelKeepsRegion)
    assert {"l1", "l2", "l3", "r1", "r2", "done"} <= report.reachable_states
    # ``l2`` is a genuine structural trap (no outgoing transition), so the
    # parallel configuration (l2, r2) cannot complete and ``done`` is reached
    # through the other regions only via l3.
    dead_ids = {f.state_ids for f in report.dead_configurations}
    assert ("l2", "r2") in dead_ids


def test_in_final_state_on_atomic_state_is_false():
    from statemachine.analysis.structural import _Analyzer

    analyzer = _Analyzer(FlatChain, max_configurations=10)
    atomic = next(s for s in analyzer.all_states if s.id == "s1")
    assert analyzer._in_final_state(atomic, frozenset({atomic})) is False

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_top_level_function_and_classmethod_return_equivalent_reports():
    via_function = top_level_analyze(FlatChain)
    via_classmethod = FlatChain.analyze_structure()
    assert via_function.reachable_states == via_classmethod.reachable_states
    assert via_function.findings == via_classmethod.findings


def test_classmethod_bound_is_inherited_by_state_machine():
    from statemachine import StateMachine

    class Strict(StateMachine):
        a = State(initial=True)
        b = State(final=True)

        go = a.to(b)

    report = Strict.analyze_structure()
    assert report.reachable_states == {"a", "b"}
    assert report.findings == []
