"""Structural (static) analysis of statechart configurations.

Tests exercise unreachable-state detection, dead configurations, eventless
cycles, conservative guard handling (guards and actions are never executed),
hierarchical / parallel / history expansion, the configuration budget, and
witness paths. Small unguarded machines are cross-checked exhaustively
against real runtime exploration.
"""

from collections import deque

import pytest

from statemachine import HistoryState
from statemachine import State
from statemachine import StateChart
from statemachine.analysis import Certainty
from statemachine.analysis import FindingKind
from statemachine.analysis import analyze_statechart
from statemachine.exceptions import InvalidDefinition
from statemachine.transition import Transition


def atomic_configuration(sm):
    return frozenset(state.id for state in sm.configuration if state.is_atomic)


def runtime_configurations(sm_class):
    """Exhaustively explore a (unguarded) machine by replaying event sequences."""
    events = [event.id for event in sm_class._events]
    seen = {}

    def replay(path):
        sm = sm_class()
        for event in path:
            if sm.is_terminated:
                break
            sm.send(event)
        return sm

    start = atomic_configuration(replay(()))
    seen[start] = ()
    queue = deque([start])
    while queue:
        config = queue.popleft()
        path = seen[config]
        for event in events:
            next_config = atomic_configuration(replay((*path, event)))
            if next_config not in seen:
                seen[next_config] = (*path, event)
                queue.append(next_config)
    return seen


class FlatMachine(StateChart):
    a = State(initial=True)
    b = State()
    c = State(final=True)
    go = a.to(b)
    finish = b.to(c)
    back = b.to(a)


class NestedMachine(StateChart):
    class outer(State.Compound):
        class inner(State.Compound):
            x = State(initial=True)
            y = State()
            move = x.to(y)
            back_inner = y.to(x)

        z = State(initial=True)
        enter_inner = z.to(inner)

    idle = State(initial=True)
    start = idle.to(outer)
    reset = outer.to(idle)
    escape = outer.inner.x.to(outer.z)


class ParallelMachine(StateChart):
    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)
            b = State()
            go_a = a.to(b)
            back_a = b.to(a)

        class r2(State.Compound):
            c = State(initial=True)
            d = State()
            go_c = c.to(d)
            back_c = d.to(c)

    done = State(final=True)
    finish = p.to(done)
    ping = p.r1.a.to(p.r1.b) | p.r2.c.to(p.r2.d)


class EventlessMachine(StateChart):
    a = State(initial=True)
    b = State()
    c = State(final=True)
    a.to(b)
    finish = b.to(c)


class InternalMachine(StateChart):
    class outer(State.Compound):
        x = State(initial=True)
        y = State()
        move = x.to(y)
        refresh = x.to.itself(internal=True)

    idle = State(initial=True)
    start = idle.to(outer)
    reset_inner = outer.to(outer.x, internal=True)
    tick = outer.to.itself(internal=True)


class ConflictMachine(StateChart):
    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)
            b = State()
            move1 = a.to(b)

        class r2(State.Compound):
            c = State(initial=True)
            d = State()
            move2 = c.to(d)

    idle = State(initial=True)
    start = idle.to(p)
    go = p.to(idle) | p.r2.to(idle)


class ShallowHistoryMachine(StateChart):
    class process(State.Compound):
        step1 = State(initial=True)
        step2 = State()
        advance = step1.to(step2)
        back = step2.to(step1)
        h = HistoryState()

    paused = State(initial=True)
    pause = process.to(paused)
    resume = paused.to(process.h)
    begin = paused.to(process)


class DeepHistoryMachine(StateChart):
    class outer(State.Compound):
        class inner(State.Compound):
            p = State(initial=True)
            q = State()
            go = p.to(q)
            back = q.to(p)

        s = State(initial=True)
        enter = s.to(inner)
        h = HistoryState(type="deep")

    away = State(initial=True)
    dive = away.to(outer)
    leave = outer.to(away)
    restore = away.to(outer.h)


class TestRuntimeCrossCheck:
    """Unguarded machines: analysis must match exhaustive runtime exploration."""

    @pytest.mark.parametrize(
        "sm_class",
        [
            FlatMachine,
            NestedMachine,
            ParallelMachine,
            EventlessMachine,
            InternalMachine,
            ConflictMachine,
        ],
    )
    def test_definitely_reachable_configurations_match_runtime(self, sm_class):
        runtime = runtime_configurations(sm_class)
        report = analyze_statechart(sm_class)
        assert report.complete
        assert report.definitely_reachable_configurations == set(runtime)
        assert report.possibly_reachable_configurations == set(runtime)

    @pytest.mark.parametrize("sm_class", [ShallowHistoryMachine, DeepHistoryMachine])
    def test_history_machines_runtime_is_covered(self, sm_class):
        """History restores depend on runtime data: runtime ⊆ possible ⊇ definite."""
        runtime = runtime_configurations(sm_class)
        report = analyze_statechart(sm_class)
        assert report.complete
        assert report.definitely_reachable_configurations <= set(runtime)
        assert set(runtime) <= report.possibly_reachable_configurations

    @pytest.mark.parametrize(
        "sm_class",
        [FlatMachine, NestedMachine, ParallelMachine, EventlessMachine, InternalMachine],
    )
    def test_unguarded_machines_have_no_findings(self, sm_class):
        report = analyze_statechart(sm_class)
        assert report.ok


class TestParallelExpansion:
    def test_all_region_combinations_are_explored(self):
        report = analyze_statechart(ParallelMachine)
        expected = {
            frozenset({"a", "c"}),
            frozenset({"a", "d"}),
            frozenset({"b", "c"}),
            frozenset({"b", "d"}),
            frozenset({"done"}),
        }
        assert report.definitely_reachable_configurations == expected

    def test_ancestor_states_are_marked_reachable(self):
        report = analyze_statechart(ParallelMachine)
        assert {"p", "r1", "r2"} <= report.definitely_reachable_states


class TestHistoryExpansion:
    def test_shallow_history_restores_are_possible(self):
        report = analyze_statechart(ShallowHistoryMachine)
        assert frozenset({"step1"}) in report.possibly_reachable_configurations
        assert frozenset({"step2"}) in report.possibly_reachable_configurations

    def test_deep_history_restores_nested_leaves(self):
        report = analyze_statechart(DeepHistoryMachine)
        assert frozenset({"q"}) in report.possibly_reachable_configurations
        assert frozenset({"p"}) in report.possibly_reachable_configurations

    def test_runtime_configurations_are_covered(self):
        runtime = runtime_configurations(DeepHistoryMachine)
        report = analyze_statechart(DeepHistoryMachine)
        assert set(runtime) <= report.possibly_reachable_configurations


class GuardedOnlyMachine(StateChart):
    a = State(initial=True)
    b = State()
    c = State(final=True)
    go = a.to(b, cond="exploding_guard")
    finish = b.to(c, cond="exploding_guard")

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class GuardedActionMachine(StateChart):
    a = State(initial=True)
    b = State(final=True)
    go = a.to(b, on="exploding_action")

    def exploding_action(self):
        raise AssertionError("structural analysis must not execute actions")


class ShadowedMachine(StateChart):
    validate_trap_states = False
    validate_final_reachability = False
    a = State(initial=True)
    b = State()
    c = State(final=True)
    go = a.to(b, cond="exploding_guard") | a.to(c)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class GuardedConflictMachine(StateChart):
    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)
            b = State()
            move1 = a.to(b)

        class r2(State.Compound):
            c = State(initial=True)
            d = State()
            move2 = c.to(d)

    idle = State(initial=True)
    start = idle.to(p)
    go = p.to(idle, cond="exploding_guard") | p.r2.to(idle)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class TestGuardsAreNeverExecuted:
    def test_guarded_targets_are_only_possible(self):
        report = analyze_statechart(GuardedOnlyMachine)
        assert report.complete
        assert report.definitely_reachable_states == {"a"}
        assert report.possibly_reachable_states == {"a", "b", "c"}

    def test_actions_are_not_executed(self):
        report = analyze_statechart(GuardedActionMachine)
        assert report.definitely_reachable_states == {"a", "b"}
        assert report.ok

    def test_unguarded_transition_shadowed_by_guard_is_possible(self):
        report = analyze_statechart(ShadowedMachine)
        assert report.definitely_reachable_states == {"a"}
        assert report.possibly_reachable_states == {"a", "b", "c"}

    def test_guarded_conflict_resolution(self):
        report = analyze_statechart(GuardedConflictMachine)
        assert frozenset({"idle"}) in report.definitely_reachable_configurations
        assert frozenset({"idle"}) in report.possibly_reachable_configurations


class DisconnectedMachine(StateChart):
    validate_disconnected_states = False
    a = State(initial=True)
    b = State()
    orphan = State()
    go = a.to(b) | b.to(a)
    spin = orphan.to.itself()


class TestUnreachableStates:
    def test_unreachable_state_is_definite(self):
        report = analyze_statechart(DisconnectedMachine)
        (finding,) = report.unreachable_states
        assert finding.kind is FindingKind.UNREACHABLE_STATE
        assert finding.certainty is Certainty.DEFINITE
        assert finding.states == ("orphan",)
        assert finding.path is None
        assert not finding.path_is_shortest
        assert "orphan" in finding.message

    def test_unreachable_is_only_possible_when_incomplete(self):
        report = analyze_statechart(DisconnectedMachine, max_configurations=1)
        assert not report.complete
        assert report.unreachable_states
        for finding in report.unreachable_states:
            assert finding.certainty is Certainty.POSSIBLE
            assert "incomplete" in finding.message


class TrapMachine(StateChart):
    validate_trap_states = False
    a = State(initial=True)
    b = State()
    go = a.to(b)


class TwoWaysTrapMachine(StateChart):
    validate_trap_states = False
    a = State(initial=True)
    b = State()
    t = State()
    go = a.to(t)
    also = b.to(t)
    other = a.to(b)


class UpgradeMachine(StateChart):
    validate_trap_states = False
    a = State(initial=True)
    b = State()
    c = State()
    go = a.to(b, cond="exploding_guard")
    direct = a.to(c)
    land = c.to(b)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class ParallelStuckMachine(StateChart):
    validate_trap_states = False

    class p(State.Parallel):
        z = State()

        class r1(State.Compound):
            a = State(initial=True)
            a_done = State(final=True)
            finish1 = a.to(a_done)

        class r2(State.Compound):
            b = State(initial=True)
            b_done = State(final=True)
            finish2 = b.to(b_done)


class ParallelFinalMachine(StateChart):
    validate_trap_states = False

    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)
            a_done = State(final=True)
            finish1 = a.to(a_done)

        class r2(State.Compound):
            b = State(initial=True)
            b_done = State(final=True)
            finish2 = b.to(b_done)


class TestDeadConfigurations:
    def test_trap_configuration_is_definite_dead_end(self):
        report = analyze_statechart(TrapMachine)
        (finding,) = report.dead_configurations
        assert finding.certainty is Certainty.DEFINITE
        assert finding.configuration == ("b",)
        assert finding.path_is_shortest

    def test_witness_path_replays_on_a_real_machine(self):
        report = analyze_statechart(TrapMachine)
        (finding,) = report.dead_configurations
        assert finding.path is not None
        assert finding.configuration is not None
        sm = TrapMachine()
        for step in finding.path:
            if step.kind == "event":
                assert step.event is not None
                sm.send(step.event)
        assert atomic_configuration(sm) == set(finding.configuration)

    def test_witness_path_steps(self):
        report = analyze_statechart(TrapMachine)
        (finding,) = report.dead_configurations
        assert finding.path is not None
        initial, step = finding.path
        assert initial.kind == "initial"
        assert initial.targets == ("a",)
        assert step.kind == "event"
        assert step.event == "go"
        assert step.source == "a"
        assert step.targets == ("b",)
        assert not step.guarded

    def test_guarded_only_outgoing_is_possible_dead_end(self):
        report = analyze_statechart(GuardedOnlyMachine)
        dead = {finding.configuration for finding in report.dead_configurations}
        assert ("a",) in dead
        assert ("b",) in dead
        assert all(
            finding.certainty is Certainty.POSSIBLE
            for finding in report.dead_configurations
        )

    def test_dead_end_found_once_with_two_incoming_paths(self):
        report = analyze_statechart(TwoWaysTrapMachine)
        (finding,) = report.dead_configurations
        assert finding.configuration == ("t",)
        assert finding.certainty is Certainty.DEFINITE

    def test_certainty_is_upgraded_when_definite_path_exists(self):
        report = analyze_statechart(UpgradeMachine)
        assert frozenset({"b"}) in report.definitely_reachable_configurations
        (finding,) = report.dead_configurations
        assert finding.configuration == ("b",)
        assert finding.certainty is Certainty.DEFINITE

    def test_parallel_region_without_final_blocks_termination(self):
        report = analyze_statechart(ParallelStuckMachine)
        (finding,) = report.dead_configurations
        assert finding.certainty is Certainty.DEFINITE
        assert finding.configuration == ("a_done", "b_done", "z")

    def test_all_regions_final_terminates_normally(self):
        report = analyze_statechart(ParallelFinalMachine)
        assert report.ok
        assert frozenset({"a_done", "b_done"}) in (
            report.definitely_reachable_configurations
        )


class EventlessLoopMachine(StateChart):
    a = State(initial=True)
    b = State()
    a.to(b)
    b.to(a)


class GuardedEventlessLoopMachine(StateChart):
    a = State(initial=True)
    b = State()
    a.to(b, cond="exploding_guard")
    b.to(a)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class TargetlessLoopMachine(StateChart):
    a = State(initial=True)
    a.to(None)


class DoubleEntryLoopMachine(StateChart):
    validate_trap_states = False
    entry = State(initial=True)
    alt = State()
    a = State()
    b = State()
    go = entry.to(alt)
    start_loop = entry.to(a)
    alt.to(b)
    a.to(b)
    b.to(a)


class UpgradeCycleMachine(StateChart):
    validate_trap_states = False
    a = State(initial=True)
    b = State()
    x = State()
    y = State()
    go = a.to(x, cond="exploding_guard")
    direct = a.to(b)
    land = b.to(x)
    x.to(y)
    y.to(x)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class TwoLoopsMachine(StateChart):
    validate_trap_states = False
    start = State(initial=True)
    a = State()
    b = State()
    c = State()
    d = State()
    go1 = start.to(a)
    go2 = start.to(c)
    a.to(b, cond="exploding_guard")
    b.to(a)
    c.to(d, cond="exploding_guard")
    d.to(c)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class DiamondEventlessMachine(StateChart):
    validate_trap_states = False
    a = State(initial=True)
    b = State()
    c = State()
    d = State()
    a.to(b, cond="exploding_guard")
    a.to(c, cond="exploding_guard")
    b.to(d, cond="exploding_guard")
    c.to(d, cond="exploding_guard")

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class EventlessHistoryMachine(StateChart):
    class process(State.Compound):
        s1 = State(initial=True)
        s2 = State()
        go = s1.to(s2)
        back = s2.to(s1)
        h = HistoryState()

    idle = State(initial=True)
    idle.to(process.h)


class TestEventlessCycles:
    def test_unguarded_eventless_loop_is_definite(self):
        report = analyze_statechart(EventlessLoopMachine)
        (finding,) = report.eventless_cycles
        assert finding.certainty is Certainty.DEFINITE
        assert finding.states == ("a", "b")
        assert finding.path_is_shortest
        assert "never reaches a stable configuration" in finding.message

    def test_definite_cycle_witness_path(self):
        report = analyze_statechart(EventlessLoopMachine)
        (finding,) = report.eventless_cycles
        assert finding.path is not None
        kinds = [step.kind for step in finding.path]
        assert kinds[0] == "initial"
        assert set(kinds[1:]) == {"eventless"}
        assert all(not step.guarded for step in finding.path)

    def test_guarded_eventless_loop_is_possible(self):
        report = analyze_statechart(GuardedEventlessLoopMachine)
        assert report.eventless_cycles
        assert all(
            finding.certainty is Certainty.POSSIBLE
            for finding in report.eventless_cycles
        )
        assert any(
            set(finding.states) == {"a", "b"} for finding in report.eventless_cycles
        )

    def test_targetless_eventless_transition_loops(self):
        report = analyze_statechart(TargetlessLoopMachine)
        (finding,) = report.eventless_cycles
        assert finding.certainty is Certainty.DEFINITE
        assert finding.states == ("a",)

    def test_cycle_is_reported_once_with_two_entry_points(self):
        report = analyze_statechart(DoubleEntryLoopMachine)
        (finding,) = report.eventless_cycles
        assert finding.certainty is Certainty.DEFINITE
        assert set(finding.states) == {"a", "b"}

    def test_cycle_certainty_is_upgraded_when_definite_path_exists(self):
        report = analyze_statechart(UpgradeCycleMachine)
        (finding,) = report.eventless_cycles
        assert finding.certainty is Certainty.DEFINITE
        assert set(finding.states) == {"x", "y"}

    def test_independent_loops_are_all_reported(self):
        report = analyze_statechart(TwoLoopsMachine)
        loop_states = {frozenset(finding.states) for finding in report.eventless_cycles}
        assert frozenset({"a", "b"}) in loop_states
        assert frozenset({"c", "d"}) in loop_states

    def test_acyclic_eventless_diamond_has_no_cycle(self):
        report = analyze_statechart(DiamondEventlessMachine)
        assert report.eventless_cycles == ()

    def test_eventless_history_branching_is_possible(self):
        report = analyze_statechart(EventlessHistoryMachine)
        assert frozenset({"s1"}) in report.possibly_reachable_configurations
        assert frozenset({"s2"}) in report.possibly_reachable_configurations
        assert frozenset({"s1"}) not in report.definitely_reachable_configurations


class WideParallelMachine(StateChart):
    validate_trap_states = False

    class p(State.Parallel):
        h = HistoryState(type="deep")

        class r1(State.Compound):
            a1 = State(initial=True)
            b1 = State()
            c1 = State()
            d1 = State()
            e1 = State()
            cycle1 = a1.to(b1) | b1.to(c1) | c1.to(d1) | d1.to(e1) | e1.to(a1)

        class r2(State.Compound):
            a2 = State(initial=True)
            b2 = State()
            c2 = State()
            d2 = State()
            e2 = State()
            cycle2 = a2.to(b2) | b2.to(c2) | c2.to(d2) | d2.to(e2) | e2.to(a2)

        class r3(State.Compound):
            a3 = State(initial=True)
            b3 = State()
            c3 = State()
            d3 = State()
            e3 = State()
            cycle3 = a3.to(b3) | b3.to(c3) | c3.to(d3) | d3.to(e3) | e3.to(a3)

    idle = State(initial=True)
    start = idle.to(p.r1.a1)
    enter_all = idle.to(p)
    resume = idle.to(p.h)


class ChainMachine(StateChart):
    a = State(initial=True)
    b = State()
    c = State(final=True)
    a.to(b)
    b.to(c)


class TestConfigurationBudget:
    def test_cap_marks_report_incomplete_with_known_results(self):
        report = analyze_statechart(ParallelMachine, max_configurations=2)
        assert not report.complete
        assert report.configurations_explored <= 2
        assert report.max_configurations == 2
        assert report.possibly_reachable_states

    def test_parallel_combinatorial_explosion_is_capped(self):
        report = analyze_statechart(WideParallelMachine, max_configurations=16)
        assert not report.complete
        assert report.configurations_explored <= 16

    def test_eventless_chain_is_bounded_by_the_budget(self):
        report = analyze_statechart(ChainMachine, max_configurations=1)
        assert not report.complete

    def test_budget_must_be_positive(self):
        with pytest.raises(ValueError, match="max_configurations"):
            analyze_statechart(FlatMachine, max_configurations=0)


class CustomStartMachine(StateChart):
    validate_disconnected_states = False
    validate_trap_states = False
    validate_final_reachability = False
    start_configuration_values = ["b"]
    a = State(initial=True)
    b = State()
    c = State(final=True)
    finish = b.to(c)


class HistoryInitialMachine(StateChart):
    class outer(State.Compound, initial=True):
        x = State(initial=True)
        y = State()
        h = HistoryState()
        move = x.to(y)
        back = y.to(x)

    outer.to(outer.h, initial=True)


class PossibleStableMachine(StateChart):
    a = State(initial=True)
    b = State()
    c = State(final=True)
    go = a.to(b, cond="exploding_guard")
    b.to(c)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class HistoryDefaultMachine(StateChart):
    class personality(State.Compound):
        smeagol = State(initial=True)
        gollum = State()
        h = HistoryState()
        dark_side = smeagol.to(gollum)
        light_side = gollum.to(smeagol)
        _ = h.to(smeagol)

    outside = State(initial=True)
    enter_via_history = outside.to(personality.h)
    leave = personality.to(outside)


class ParallelHistoryMachine(StateChart):
    class p(State.Parallel):
        h = HistoryState()

        class r1(State.Compound):
            a = State(initial=True)
            b = State()
            go = a.to(b)
            back = b.to(a)

        class r2(State.Compound):
            c = State(initial=True)
            d = State()
            move = c.to(d)
            back_c = d.to(c)

    idle = State(initial=True)
    start = idle.to(p)
    leave = p.to(idle)
    resume = idle.to(p.h)


class TestMisc:
    def test_analyze_structure_classmethod(self):
        report = FlatMachine.analyze_structure()
        assert report.complete
        assert report.ok

    def test_abstract_class_is_rejected(self):
        with pytest.raises(InvalidDefinition, match="no states or transitions"):
            analyze_statechart(StateChart)

    def test_custom_start_configuration_values(self):
        report = analyze_statechart(CustomStartMachine)
        assert frozenset({"b"}) in report.definitely_reachable_configurations
        assert frozenset({"a"}) not in report.possibly_reachable_configurations
        (finding,) = report.unreachable_states
        assert finding.states == ("a",)
        assert finding.certainty is Certainty.DEFINITE

    def test_initial_entry_through_history_is_possible_only(self):
        report = analyze_statechart(HistoryInitialMachine)
        assert frozenset({"x"}) in report.possibly_reachable_configurations
        assert frozenset({"y"}) in report.possibly_reachable_configurations
        assert report.definitely_reachable_configurations == frozenset()

    def test_guarded_stable_configuration_is_explored_once(self):
        report = analyze_statechart(PossibleStableMachine)
        assert report.complete
        assert frozenset({"c"}) in report.possibly_reachable_configurations
        assert frozenset({"c"}) not in report.definitely_reachable_configurations

    def test_history_default_transition_targets(self):
        report = analyze_statechart(HistoryDefaultMachine)
        assert frozenset({"smeagol"}) in report.possibly_reachable_configurations
        assert frozenset({"gollum"}) in report.possibly_reachable_configurations
        assert frozenset({"smeagol"}) not in report.definitely_reachable_configurations

    def test_shallow_history_of_parallel_parent(self):
        report = analyze_statechart(ParallelHistoryMachine)
        assert frozenset({"a", "c"}) in report.possibly_reachable_configurations
        assert frozenset({"b", "c"}) in report.definitely_reachable_configurations


class NestedParallelFinalMachine(StateChart):
    validate_trap_states = False

    class p(State.Parallel):
        class q(State.Parallel):
            class q1(State.Compound):
                qa = State(initial=True)
                qa_done = State(final=True)
                qfin1 = qa.to(qa_done)

            class q2(State.Compound):
                qb = State(initial=True)
                qb_done = State(final=True)
                qfin2 = qb.to(qb_done)

        class r(State.Compound):
            c = State(initial=True)
            c_done = State(final=True)
            cfin = c.to(c_done)


class TestNestedParallel:
    def test_nested_parallel_all_final_terminates(self):
        report = analyze_statechart(NestedParallelFinalMachine)
        assert report.ok
        final_config = frozenset({"qa_done", "qb_done", "c_done"})
        assert final_config in report.definitely_reachable_configurations

    def test_nested_parallel_matches_runtime(self):
        runtime = runtime_configurations(NestedParallelFinalMachine)
        report = analyze_statechart(NestedParallelFinalMachine)
        assert report.definitely_reachable_configurations == set(runtime)


class TestEventlessEdgesBeyondTheBudget:
    def test_edges_to_unexplored_configurations_are_ignored(self):
        report = analyze_statechart(DiamondEventlessMachine, max_configurations=2)
        assert not report.complete


class ConvergingTrapMachine(StateChart):
    validate_trap_states = False
    a = State(initial=True)
    b = State()
    c = State()
    t = State()
    go_b = a.to(b)
    go_c = a.to(c)
    b.to(t)
    c.to(t)


class AtomicRegionMachine(StateChart):
    validate_trap_states = False

    class p(State.Parallel):
        z = State()

        class r(State.Compound):
            a = State(initial=True)
            b = State()
            go = a.to(b)

    idle = State(initial=True)
    start = idle.to(p.z)


class ParallelEventlessMachine(StateChart):
    validate_trap_states = False

    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)
            b = State()
            a.to(b)

        class r2(State.Compound):
            c = State(initial=True)
            d = State()
            c.to(d, cond="exploding_guard")

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class ConflictEventlessMachine(StateChart):
    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)

        class r2(State.Compound):
            c = State(initial=True)
            d = State()
            c.to(d)

        class r3(State.Compound):
            e = State(initial=True)
            f = State()
            e.to(f, cond="exploding_guard")

    idle = State(final=True)
    p.to(idle)

    def exploding_guard(self):
        raise AssertionError("structural analysis must not execute guards")


class BranchyRegionMachine(StateChart):
    validate_trap_states = False

    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)
            b = State()
            go = a.to(b)

        class r2(State.Compound):
            h2 = HistoryState(type="deep")

            class w2(State.Parallel):
                class q1(State.Compound):
                    x1 = State(initial=True)
                    x2 = State()
                    x3 = State()
                    cx1 = x1.to(x2) | x2.to(x3) | x3.to(x1)

                class q2(State.Compound):
                    y1 = State(initial=True)
                    y2 = State()
                    cy2 = y1.to(y2) | y2.to(y1)

        class r3(State.Compound):
            h3 = HistoryState(type="deep")

            class w3(State.Parallel):
                class q3(State.Compound):
                    u1 = State(initial=True)
                    u2 = State()
                    u3 = State()
                    cu1 = u1.to(u2) | u2.to(u3) | u3.to(u1)

                class q4(State.Compound):
                    v1 = State(initial=True)
                    v2 = State()
                    cv2 = v1.to(v2) | v2.to(v1)

    p.r2.to(p.r2.h2, initial=True)
    p.r3.to(p.r3.h3, initial=True)
    idle = State(initial=True)
    start = idle.to(p.r1.a)


class TestBranchCoverageEdgeCases:
    def test_converging_eventless_chains_report_dead_config_once(self):
        report = analyze_statechart(ConvergingTrapMachine)
        (finding,) = report.dead_configurations
        assert finding.configuration == ("t",)
        assert finding.certainty is Certainty.DEFINITE

    def test_transition_into_atomic_parallel_region_completes_siblings(self):
        report = analyze_statechart(AtomicRegionMachine)
        assert frozenset({"z", "a"}) in report.definitely_reachable_configurations

    def test_atomic_region_matches_runtime(self):
        runtime = runtime_configurations(AtomicRegionMachine)
        report = analyze_statechart(AtomicRegionMachine)
        assert report.definitely_reachable_configurations == set(runtime)

    def test_parallel_eventless_mixes_definite_and_guarded(self):
        report = analyze_statechart(ParallelEventlessMachine)
        assert frozenset({"b", "c"}) in report.definitely_reachable_configurations
        assert frozenset({"b", "d"}) in report.possibly_reachable_configurations
        assert frozenset({"b", "d"}) not in report.definitely_reachable_configurations

    def test_conflicting_eventless_selection(self):
        report = analyze_statechart(ConflictEventlessMachine)
        assert frozenset({"idle"}) in report.definitely_reachable_configurations
        assert "f" in report.possibly_reachable_states
        assert "f" not in report.definitely_reachable_states

    def test_branched_default_entry_of_sibling_region_is_capped(self):
        report = analyze_statechart(BranchyRegionMachine, max_configurations=16)
        assert not report.complete


class MultiTargetMachine(StateChart):
    validate_disconnected_states = False
    validate_trap_states = False
    validate_final_reachability = False

    class p(State.Parallel):
        class r1(State.Compound):
            a = State(initial=True)
            b = State()

        class r2(State.Compound):
            c = State(initial=True)
            d = State()

    idle = State(initial=True)
    done = State(final=True)
    finish = p.to(done)


# Multi-target transition as produced by the SCXML parser (single transition,
# several targets): sibling regions must not be default-entered twice.
MultiTargetMachine.idle.transitions.add_transitions(
    Transition(
        MultiTargetMachine.idle,
        [MultiTargetMachine.p.r1.b, MultiTargetMachine.p.r2.d],
        event="jump",
    )
)


class TestMultiTargetTransitions:
    def test_multi_target_entry_matches_runtime(self):
        report = analyze_statechart(MultiTargetMachine)
        assert frozenset({"b", "d"}) in report.definitely_reachable_configurations
        assert frozenset({"a", "b", "c", "d"}) not in (
            report.possibly_reachable_configurations
        )
        sm = MultiTargetMachine()
        sm.send("jump")
        assert atomic_configuration(sm) == {"b", "d"}
