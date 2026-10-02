"""Decision tracing: why transitions were or were not executed within a macrostep.

Covers candidate discovery, guard outcomes (including not-evaluated guards),
parent/child priority, exit-set conflicts, eventless/internal microstep
attribution, sync/async structural equality, and tracing transparency
(identical state results, callback order, and guard call counts).
"""

from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum

import pytest

from statemachine import State
from statemachine import StateChart
from statemachine.trace import CandidateFate
from statemachine.trace import DecisionTracer
from statemachine.trace import GuardOutcome
from statemachine.trace import transition_definition_id
from statemachine.transition import Transition


def make_frontline_battle(guard_calls):
    """Parent transition competes with two parallel child transitions."""

    class FrontlineBattle(StateChart):
        class war(State.Parallel):
            class front_a(State.Compound):
                a1 = State(initial=True)
                a2 = State(final=True)

                advance = a1.to(a2)

            class front_b(State.Compound):
                b1 = State(initial=True)
                b2 = State(final=True)

                advance = b1.to(b2)
                hold = b1.to(b1, event="hold")

        done = State(final=True)
        retreat = war.to(done, event="advance", cond="parent_guard")

        def parent_guard(self):
            guard_calls.append("parent_guard")
            return True

    return FrontlineBattle


def make_flanked_battle(guard_calls):
    """Only one region has a child transition; the parent transition competes."""

    class FlankedBattle(StateChart):
        class war(State.Parallel):
            class front_a(State.Compound):
                a1 = State(initial=True)
                a2 = State(final=True)

                advance = a1.to(a2)

            class front_b(State.Compound):
                b1 = State(initial=True)
                b2 = State(final=True)

                hold = b1.to(b2, event="hold")

        done = State(final=True)
        retreat = war.to(done, event="advance", cond="parent_guard")

        def parent_guard(self):
            guard_calls.append("parent_guard")
            return True

    return FlankedBattle


def make_overwhelmed_battle(guard_calls):
    """Like the flanked battle, but the child transition is selected after the
    parent's, so the child preempts the parent during conflict resolution."""

    class OverwhelmedBattle(StateChart):
        class war(State.Parallel):
            class front_a(State.Compound):
                a1 = State(initial=True)
                a2 = State(final=True)

                hold = a1.to(a2, event="hold")

            class front_b(State.Compound):
                b1 = State(initial=True)
                b2 = State(final=True)

                advance = b1.to(b2)

        done = State(final=True)
        retreat = war.to(done, event="advance", cond="parent_guard")

        def parent_guard(self):
            guard_calls.append("parent_guard")
            return True

    return OverwhelmedBattle


def make_breakthrough_battle():
    """A child transition targets outside the parallel, conflicting by exit set."""

    class BreakthroughBattle(StateChart):
        class war(State.Parallel):
            class front_a(State.Compound):
                a1 = State(initial=True)
                a2 = State(final=True)

                regroup = a1.to(a2, event="regroup")

            class front_b(State.Compound):
                b1 = State(initial=True)
                b2 = State(final=True)

                advance = b1.to(b2)

        done = State(final=True)
        advance = war.front_a.a1.to(done)

    return BreakthroughBattle


def make_lazy_flow():
    """Eventless transition fires only after a flag is set on the model."""

    class LazyFlow(StateChart):
        s1 = State(initial=True)
        s2 = State()
        s3 = State(final=True)

        s1.to(s2, cond="ready")
        poke = s2.to(s3)

        def ready(self):
            return getattr(self.model, "flag", False)

    return LazyFlow


def make_guarded_flow(guard_calls, callback_log):
    class GuardedFlow(StateChart):
        s1 = State(initial=True)
        s2 = State()
        s3 = State(final=True)

        go = s1.to(s2, cond="allow")
        s2.to(s3, cond="ready")

        def allow(self):
            guard_calls.append("allow")
            return True

        def ready(self):
            guard_calls.append("ready")
            return True

        def on_enter_state(self, state):
            callback_log.append(f"enter:{state.id}")

        def on_exit_state(self, state):
            callback_log.append(f"exit:{state.id}")

        def on_transition(self, event):
            callback_log.append(f"on:{event}")

    return GuardedFlow


def make_fragile_flow(explode_calls):
    class FragileFlow(StateChart):
        s1 = State(initial=True)
        s2 = State(final=True)
        s3 = State(final=True)

        go = s1.to(s2, cond="explodes")
        error_execution = s1.to(s3)

        def explodes(self):
            explode_calls.append("explodes")
            raise RuntimeError("boom")

    return FragileFlow


def make_strict_flow():
    class StrictFlow(StateChart):
        catch_errors_as_events = False

        s1 = State(initial=True)
        s2 = State(final=True)

        go = s1.to(s2, cond="explodes")

        def explodes(self):
            raise RuntimeError("boom")

    return StrictFlow


def make_double_take(guard_calls):
    class DoubleTake(StateChart):
        s1 = State(initial=True)
        s2 = State(final=True)

        go = s1.to(s2, cond="first_wins")
        again = s1.to(s2, event="go")

        def first_wins(self):
            guard_calls.append("first_wins")
            return True

    return DoubleTake


def candidates_by_id(microstep):
    return {(c.transition_id, c.evaluated_for): c for c in microstep.candidates}


def microsteps_for(tracer, event):
    """The microsteps of the (single) macrostep recorded for ``event``."""
    (macrostep,) = [m for m in tracer.macrosteps if m.event == event]
    return macrostep.microsteps


def snapshot(tracer, event):
    """Normalized structure of the macrosteps recorded for ``event``."""
    return [
        [
            (
                microstep.event,
                microstep.kind,
                tuple(microstep.executed),
                tuple(
                    (
                        c.transition_id,
                        c.evaluated_for,
                        c.guard.value,
                        c.fate.value,
                        c.rule,
                        c.eliminated_by,
                        c.exit_set_overlap,
                    )
                    for c in microstep.candidates
                ),
            )
            for microstep in macrostep.microsteps
        ]
        for macrostep in tracer.macrosteps
        if macrostep.event == event
    ]


@pytest.mark.timeout(5)
class TestParentChildCompetition:
    async def test_parent_shadowed_by_two_parallel_children(self, sm_runner):
        guard_calls = []
        sm = await sm_runner.start(make_frontline_battle(guard_calls))
        sm.decision_tracer.enabled = True
        await sm_runner.send(sm, "hold")
        await sm_runner.send(sm, "advance")

        assert {"war", "front_a", "a2", "front_b", "b2"} == {
            s.id for s in sm.configuration
        }
        # The parent's guard was never called: it lost by child priority.
        assert guard_calls == []
        # Each external event gets its own macrostep, in causal order.
        assert [m.event for m in sm.decision_tracer.macrosteps] == ["hold", "advance"]

        assert snapshot(sm.decision_tracer, "advance") == [
            [
                (
                    "advance",
                    "external",
                    ("a1->a2:advance#0", "b1->b2:advance#0"),
                    (
                        ("a1->a2:advance#0", "a1", "passed", "executed", None, None, ()),
                        (
                            "war->done:advance retreat#0",
                            "a1",
                            "not_evaluated",
                            "shadowed",
                            "child_priority",
                            "a1->a2:advance#0",
                            (),
                        ),
                        ("b1->b2:advance#0", "b1", "passed", "executed", None, None, ()),
                        (
                            "war->done:advance retreat#0",
                            "b1",
                            "not_evaluated",
                            "shadowed",
                            "child_priority",
                            "b1->b2:advance#0",
                            (),
                        ),
                    ),
                )
            ]
        ]

    async def test_parent_preempted_by_descendant_priority(self, sm_runner):
        guard_calls = []
        sm = await sm_runner.start(make_flanked_battle(guard_calls))
        sm.decision_tracer.enabled = True
        await sm_runner.send(sm, "advance")

        assert {"war", "front_a", "a2", "front_b", "b1"} == {
            s.id for s in sm.configuration
        }
        # Eval: the parent guard ran once (for the region without a child transition)
        assert guard_calls == ["parent_guard"]

        (microstep,) = microsteps_for(sm.decision_tracer, "advance")
        candidates = candidates_by_id(microstep)

        child = candidates[("a1->a2:advance#0", "a1")]
        assert child.fate is CandidateFate.EXECUTED

        parent = candidates[("war->done:advance retreat#0", "b1")]
        assert parent.guard is GuardOutcome.PASSED
        assert parent.fate is CandidateFate.PREEMPTED
        assert parent.rule == "descendant_priority"
        assert parent.eliminated_by == "a1->a2:advance#0"
        assert parent.exit_set_overlap == ("a1",)

        shadowed = candidates[("war->done:advance retreat#0", "a1")]
        assert shadowed.guard is GuardOutcome.NOT_EVALUATED
        assert shadowed.fate is CandidateFate.SHADOWED

        assert microstep.executed == ["a1->a2:advance#0"]

    async def test_parent_preempted_by_later_child(self, sm_runner):
        """The child transition is selected after the parent's and still wins."""
        guard_calls = []
        sm = await sm_runner.start(make_overwhelmed_battle(guard_calls))
        sm.decision_tracer.enabled = True
        await sm_runner.send(sm, "advance")

        assert {"war", "front_a", "a1", "front_b", "b2"} == {
            s.id for s in sm.configuration
        }
        assert guard_calls == ["parent_guard"]

        (microstep,) = microsteps_for(sm.decision_tracer, "advance")
        candidates = candidates_by_id(microstep)

        child = candidates[("b1->b2:advance#0", "b1")]
        assert child.fate is CandidateFate.EXECUTED

        parent = candidates[("war->done:advance retreat#0", "a1")]
        assert parent.guard is GuardOutcome.PASSED
        assert parent.fate is CandidateFate.PREEMPTED
        assert parent.rule == "descendant_priority"
        assert parent.eliminated_by == "b1->b2:advance#0"
        assert parent.exit_set_overlap == ("b1",)

        assert microstep.executed == ["b1->b2:advance#0"]

    async def test_child_preempted_by_document_order(self, sm_runner):
        sm = await sm_runner.start(make_breakthrough_battle())
        sm.decision_tracer.enabled = True
        await sm_runner.send(sm, "advance")

        assert {"done"} == {s.id for s in sm.configuration}

        (microstep,) = microsteps_for(sm.decision_tracer, "advance")
        candidates = candidates_by_id(microstep)

        winner = candidates[("a1->done:advance#1", "a1")]
        assert winner.fate is CandidateFate.EXECUTED

        loser = candidates[("b1->b2:advance#0", "b1")]
        assert loser.guard is GuardOutcome.PASSED
        assert loser.fate is CandidateFate.PREEMPTED
        assert loser.rule == "document_order"
        assert loser.eliminated_by == "a1->done:advance#1"
        assert loser.exit_set_overlap == ("b1",)

    async def test_same_state_shadowing_uses_document_order(self, sm_runner):
        guard_calls = []
        sm = await sm_runner.start(make_double_take(guard_calls))
        sm.decision_tracer.enabled = True
        await sm_runner.send(sm, "go")

        assert {"s2"} == {s.id for s in sm.configuration}
        assert guard_calls == ["first_wins"]

        (microstep,) = microsteps_for(sm.decision_tracer, "go")
        candidates = candidates_by_id(microstep)
        shadowed = candidates[("s1->s2:go again#1", "s1")]
        assert shadowed.guard is GuardOutcome.NOT_EVALUATED
        assert shadowed.fate is CandidateFate.SHADOWED
        assert shadowed.rule == "document_order"
        assert shadowed.eliminated_by == "s1->s2:go#0"


@pytest.mark.timeout(5)
class TestMacrostepAttribution:
    async def test_eventless_microstep_belongs_to_the_triggering_macrostep(self, sm_runner):
        sm = await sm_runner.start(make_lazy_flow())
        sm.decision_tracer.enabled = True
        sm.model.flag = True
        await sm_runner.send(sm, "poke")

        assert {"s3"} == {s.id for s in sm.configuration}

        macrosteps = sm.decision_tracer.macrosteps
        assert [m.event for m in macrosteps] == [None, "poke"]

        # The eventless microstep fired before the external event, in causal order.
        (eventless,) = macrosteps[0].microsteps
        assert eventless.kind == "eventless"
        assert eventless.event is None
        assert eventless.executed == ["s1->s2:<eventless>#0"]

        (external,) = macrosteps[1].microsteps
        assert external.kind == "external"
        assert external.executed == ["s2->s3:poke#0"]

    async def test_internal_error_event_recorded_in_the_same_macrostep(self, sm_runner):
        explode_calls = []
        sm = await sm_runner.start(make_fragile_flow(explode_calls))
        sm.decision_tracer.enabled = True
        await sm_runner.send(sm, "go")

        assert {"s3"} == {s.id for s in sm.configuration}
        # The guard was evaluated exactly once — tracing never re-invokes guards.
        assert explode_calls == ["explodes"]

        (macrostep,) = snapshot(sm.decision_tracer, "go")
        external, internal = macrostep

        (event, kind, executed, candidates) = external
        assert (event, kind, executed) == ("go", "external", ())
        ((t_id, _for, guard, fate, rule, _by, _overlap),) = candidates
        assert (t_id, guard, fate, rule) == ("s1->s2:go#0", "failed", "rejected", "guard")

        (event, kind, executed, candidates) = internal
        assert (event, kind) == ("error.execution", "internal")
        assert executed == ("s1->s3:error_execution error.execution#1",)


@pytest.mark.timeout(5)
class TestTracingTransparency:
    async def test_same_results_with_tracing_on_and_off(self, sm_runner):
        outcomes = []
        for enabled in (False, True):
            guard_calls, callback_log = [], []
            sm = await sm_runner.start(make_guarded_flow(guard_calls, callback_log))
            sm.decision_tracer.enabled = enabled
            await sm_runner.send(sm, "go")
            outcomes.append(
                (sorted(s.id for s in sm.configuration), guard_calls, callback_log)
            )
        assert outcomes[0] == outcomes[1]

    async def test_trace_contains_no_model_or_sensitive_payload(self, sm_runner):
        sm = await sm_runner.start(make_frontline_battle([]))
        sm.decision_tracer.enabled = True
        await sm_runner.send(sm, "advance", token="s3cr3t", password="hunter2")

        blob = repr(sm.decision_tracer.macrosteps)
        assert "s3cr3t" not in blob
        assert "hunter2" not in blob

        def walk(obj):
            if obj is None or isinstance(obj, (str, int, float, bool, Enum)):
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
                return
            if is_dataclass(obj):
                for f in fields(obj):
                    walk(getattr(obj, f.name))
                return
            raise AssertionError(f"unexpected object in trace: {type(obj)}")

        walk(sm.decision_tracer.macrosteps)


@pytest.mark.timeout(5)
class TestPartialTraceOnError:
    async def test_partial_trace_available_when_guard_raises(self, sm_runner):
        sm = await sm_runner.start(make_strict_flow())
        sm.decision_tracer.enabled = True

        with pytest.raises(RuntimeError, match="boom"):
            await sm_runner.send(sm, "go")

        assert "s1" in sm.configuration_values
        current = sm.decision_tracer.current
        assert current is not None
        assert current.event == "go"
        assert sm.decision_tracer.macrosteps[-1] is current


class TestDecisionTracerUnit:
    def test_disabled_by_default_and_all_methods_are_noops(self):
        tracer = DecisionTracer()
        assert tracer.enabled is False
        assert tracer.current is None

        tracer.begin_macrostep(1, "go")
        tracer.begin_microstep("go", "external")
        tracer.record_guard(None, None, True)
        tracer.record_shadowed(None, None, None)
        tracer.record_preemption(None, None, "rule", ())
        tracer.record_execution([None])
        tracer.clear()

        assert tracer.macrosteps == []
        assert tracer.current is None

    def test_begin_macrostep_on_fresh_tracer(self):
        tracer = DecisionTracer()
        tracer.enabled = True
        tracer.begin_macrostep(1, "go")
        assert tracer.current is not None
        assert tracer.current.event == "go"
        assert tracer.macrosteps == [tracer.current]

    def test_execution_without_selection_materializes_defensively(self):
        tracer = DecisionTracer()
        tracer.enabled = True
        tracer.record_execution([])
        assert tracer.current is not None
        assert len(tracer.current.microsteps) == 1

    def test_clear_drops_recorded_traces(self):
        tracer = DecisionTracer()
        tracer.enabled = True
        tracer.begin_macrostep(1, "go")
        tracer.clear()
        assert tracer.macrosteps == []
        assert tracer.current is None

    def test_selected_candidate_not_executed_keeps_selected_fate(self):
        class Tiny(StateChart):
            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(s2)
            stay = s1.to(s1, event="stay")

        t_go, t_stay = Tiny.s1.transitions

        tracer = DecisionTracer()
        tracer.enabled = True
        tracer.begin_microstep("go", "external")
        tracer.record_guard(t_go, Tiny.s1, True)
        tracer.record_guard(t_stay, Tiny.s1, True)
        tracer.record_execution([t_go])

        microstep = tracer.current.microsteps[0]
        fates = {c.transition_id: c.fate for c in microstep.candidates}
        assert fates == {
            "s1->s2:go#0": CandidateFate.EXECUTED,
            "s1->s1:stay#1": CandidateFate.SELECTED,
        }
        # A single microstep record holds both candidates.
        assert len(tracer.current.microsteps) == 1

    def test_preemption_ignores_candidates_already_rejected(self):
        class Tiny(StateChart):
            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(s2)
            stay = s1.to(s1, event="stay")

        t_go, t_stay = Tiny.s1.transitions

        tracer = DecisionTracer()
        tracer.enabled = True
        tracer.begin_microstep("go", "external")
        tracer.record_guard(t_stay, Tiny.s1, False)
        tracer.record_preemption(t_stay, t_go, "document_order", ())

        (candidate,) = tracer.current.microsteps[0].candidates
        assert candidate.fate is CandidateFate.REJECTED
        assert candidate.eliminated_by is None

    def test_transition_definition_id_targetless(self):
        source = State()
        transition = Transition(source, None)
        source.transitions.add_transitions(transition)
        assert transition_definition_id(transition) == "-><none>:<eventless>#0"
