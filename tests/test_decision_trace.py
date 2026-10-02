from collections import Counter
from dataclasses import asdict
from dataclasses import is_dataclass
from inspect import getsourcelines

import pytest

from statemachine import State
from statemachine import StateChart


class ParallelDecisionChart(StateChart):
    start = State(initial=True)
    cancelled = State(final=True)
    failed = State(final=True)

    class work(State.Parallel):
        class region_a(State.Compound):
            a = State(initial=True)
            a_ready = State()
            a_done = State(final=True)

        class region_b(State.Compound):
            b = State(initial=True)
            b_ready = State()
            b_done = State(final=True)

    finish = (
        work.region_a.a.to(work.region_a.a_ready, cond="child_a_allows")
        | work.region_b.b.to(work.region_b.b_ready, cond="child_b_allows")
        | work.to(cancelled, cond="parent_allows")
    )
    error_execution = work.to(failed)
    begin = start.to(work)

    work.region_a.a_ready.to(work.region_a.a_done)
    work.region_b.b_ready.to(work.region_b.b_done)

    start_configuration_values = ["work"]

    def __init__(
        self,
        *,
        child_a_allows=True,
        child_b_allows=True,
        parent_allows=True,
        raise_child_a=False,
        enable_decision_trace=False,
        listeners=None,
    ):
        self.calls: list[str] = []
        self.guard_calls = Counter()
        self.child_a_allows_value = child_a_allows
        self.child_b_allows_value = child_b_allows
        self.parent_allows_value = parent_allows
        self.raise_child_a = raise_child_a
        super().__init__(listeners=listeners, enable_decision_trace=enable_decision_trace)

    def _guard(self, name, allowed):
        self.guard_calls[name] += 1
        self.calls.append(f"guard:{name}")
        if self.raise_child_a and name == "child_a_allows":
            raise RuntimeError("secret-token should never be copied to trace")
        return allowed

    def child_a_allows(self):
        return self._guard("child_a_allows", self.child_a_allows_value)

    def child_b_allows(self):
        return self._guard("child_b_allows", self.child_b_allows_value)

    def parent_allows(self):
        return self._guard("parent_allows", self.parent_allows_value)

    def before_transition(self, event: str, source: State):
        self.calls.append(f"before:{event}:{source.id}")

    def on_transition(self, event: str, source: State, target: State):
        self.calls.append(f"on:{event}:{source.id}->{target.id}")

    def after_transition(self, event: str, source: State):
        self.calls.append(f"after:{event}:{source.id}")

    def on_enter_state(self, event: str, state: State):
        self.calls.append(f"enter:{event or '<eventless>'}:{state.id}")

    def on_exit_state(self, event: str, state: State):
        self.calls.append(f"exit:{event or '<eventless>'}:{state.id}")


class ParallelDecisionMachine(ParallelDecisionChart):
    catch_errors_as_events = False


def _plain(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return value


def _send(sm_runner, sm, event="finish"):
    return sm_runner.send(sm, event)


def _candidate(step, active_state, source_state):
    for candidate in step["candidates"]:
        if candidate["active_state"] == active_state and candidate["source_state"] == source_state:
            return candidate
    raise AssertionError(f"candidate {active_state}/{source_state} not found")


def _selected(step):
    return {
        (candidate["active_state"], candidate["source_state"])
        for candidate in step["candidates"]
        if candidate["id"] in step["selected_candidate_ids"]
    }


def _finish_definition_line():
    source_lines, start = getsourcelines(ParallelDecisionChart)
    for offset, line in enumerate(source_lines):
        if line.lstrip().startswith("finish ="):
            return start + offset
    raise AssertionError("finish definition not found")


@pytest.mark.timeout(5)
async def test_macrostep_trace_explains_parallel_parent_and_child_decisions(sm_runner):
    sm = await sm_runner.start(
        ParallelDecisionChart, enable_decision_trace=True
    )
    await _send(sm_runner, sm)

    assert set(sm.configuration_values) == {
        "work",
        "region_a",
        "region_b",
        "a_done",
        "b_done",
    }
    assert sm.guard_calls == Counter(
        child_a_allows=1, child_b_allows=1, parent_allows=0
    )

    trace = _plain(sm.decision_trace)
    assert trace["enabled"] is True
    assert len(trace["macrosteps"]) == 1
    macrostep = trace["macrosteps"][0]
    assert macrostep["external_event"] == "finish"
    assert macrostep["status"] == "completed"
    assert [step["trigger_origin"] for step in macrostep["steps"]] == [
        "external",
        "eventless",
    ]

    external_step, eventless_step = macrostep["steps"]
    assert external_step["executed"] is True
    assert eventless_step["executed"] is True
    assert len(external_step["candidates"]) == 4
    assert _selected(external_step) == {
        ("a", "a"),
        ("b", "b"),
    }
    assert _selected(eventless_step) == {
        ("a_ready", "a_ready"),
        ("b_ready", "b_ready"),
    }

    expected_line = _finish_definition_line()
    child_a = _candidate(external_step, "a", "a")
    child_b = _candidate(external_step, "b", "b")
    parent_from_a = _candidate(external_step, "a", "work")
    parent_from_b = _candidate(external_step, "b", "work")

    for candidate in (child_a, child_b, parent_from_a, parent_from_b):
        assert candidate["definition"]["module"] == "tests.test_decision_trace"
        assert candidate["definition"]["filename"] == "test_decision_trace.py"
        assert candidate["definition"]["line"] == expected_line

    assert child_a["guards"] == [{"name": "child_a_allows", "status": "true", "error": None}]
    assert child_b["guards"] == [{"name": "child_b_allows", "status": "true", "error": None}]
    for parent in (parent_from_a, parent_from_b):
        assert parent["guards"] == [
            {"name": "parent_allows", "status": "not_evaluated", "error": None}
        ]
        assert parent["eliminated_by_rule"] == "descendant_source_priority"
        assert parent["conflicting_states"] == ()
    assert parent_from_a["eliminated_by_candidate_id"] == child_a["id"]
    assert parent_from_b["eliminated_by_candidate_id"] == child_b["id"]

    serialized = repr(sm.decision_trace)
    assert "secret-token" not in serialized
    assert "model" not in serialized
    assert "token" not in serialized


@pytest.mark.timeout(5)
async def test_trace_records_guard_denial_and_exit_set_conflict(sm_runner):
    sm = await sm_runner.start(
        ParallelDecisionChart, child_a_allows=False, enable_decision_trace=True
    )
    await _send(sm_runner, sm)

    assert set(sm.configuration_values) == {"work", "region_a", "region_b", "a", "b_done"}
    assert sm.guard_calls == Counter(
        child_a_allows=1, parent_allows=1, child_b_allows=1
    )

    macrostep = _plain(sm.decision_trace)["macrosteps"][0]
    external_step = macrostep["steps"][0]
    child_a = _candidate(external_step, "a", "a")
    child_b = _candidate(external_step, "b", "b")
    parent_from_a = _candidate(external_step, "a", "work")
    parent_from_b = _candidate(external_step, "b", "work")

    assert child_a["guards"][0]["status"] == "false"
    assert child_a["eliminated_by_rule"] == "guard_denied"
    assert parent_from_a["guards"][0]["status"] == "true"
    assert parent_from_a["eliminated_by_rule"] == "exit_set_conflict"
    assert parent_from_a["eliminated_by_candidate_id"] == child_b["id"]
    assert "b" in parent_from_a["conflicting_states"]
    assert "work" in parent_from_a["conflicting_domains"]
    assert parent_from_b["guards"][0]["status"] == "not_evaluated"
    assert external_step["selected_candidate_ids"] == [child_b["id"]]


@pytest.mark.timeout(5)
async def test_trace_has_no_behavioral_or_guard_count_effect(sm_runner):
    traced = await sm_runner.start(ParallelDecisionChart, enable_decision_trace=True)
    plain = await sm_runner.start(ParallelDecisionChart, enable_decision_trace=False)

    await _send(sm_runner, traced)
    await _send(sm_runner, plain)

    assert traced.configuration_values == plain.configuration_values
    assert traced.calls == plain.calls
    assert traced.guard_calls == plain.guard_calls
    assert plain.decision_trace.enabled is False
    assert plain.decision_trace.macrosteps == []


@pytest.mark.timeout(5)
async def test_guard_error_is_traced_and_partial_trace_survives(sm_runner):
    traced = await sm_runner.start(
        ParallelDecisionMachine, raise_child_a=True, enable_decision_trace=True
    )
    plain = await sm_runner.start(
        ParallelDecisionMachine, raise_child_a=True, enable_decision_trace=False
    )

    with pytest.raises(RuntimeError, match="secret-token"):
        await _send(sm_runner, traced)
    with pytest.raises(RuntimeError, match="secret-token"):
        await _send(sm_runner, plain)

    assert traced.configuration_values == plain.configuration_values
    assert traced.calls == plain.calls
    assert traced.guard_calls == plain.guard_calls

    macrostep = _plain(traced.decision_trace)["macrosteps"][0]
    assert macrostep["status"] == "error"
    assert macrostep["error"] == "RuntimeError"
    external_step = macrostep["steps"][0]
    child_a = _candidate(external_step, "a", "a")
    assert child_a["guards"] == [
        {"name": "child_a_allows", "status": "error", "error": "RuntimeError"}
    ]
    assert child_a["eliminated_by_rule"] == "guard_error"
    assert {candidate["active_state"] for candidate in external_step["candidates"]} == {"a"}
    serialized = repr(traced.decision_trace)
    assert "secret-token" not in serialized
    assert "token" not in serialized


@pytest.mark.timeout(5)
async def test_statechart_guard_error_internal_handler_stays_in_same_macrostep(sm_runner):
    traced = await sm_runner.start(
        ParallelDecisionChart, raise_child_a=True, enable_decision_trace=True
    )
    plain = await sm_runner.start(
        ParallelDecisionChart, raise_child_a=True, enable_decision_trace=False
    )

    await _send(sm_runner, traced)
    await _send(sm_runner, plain)

    assert set(traced.configuration_values) == {"failed"}
    assert traced.configuration_values == plain.configuration_values
    assert traced.calls == plain.calls
    assert traced.guard_calls == plain.guard_calls

    macrostep = _plain(traced.decision_trace)["macrosteps"][0]
    assert macrostep["status"] == "completed"
    assert [step["trigger_origin"] for step in macrostep["steps"]] == [
        "external",
        "eventless",
        "internal",
    ]
    external_step, _eventless_step, internal_step = macrostep["steps"]
    assert external_step["status"] == "executed"
    assert _candidate(external_step, "a", "a")["eliminated_by_rule"] == "guard_error"
    assert internal_step["trigger_event"] == "error.execution"
    assert internal_step["selected_candidate_ids"]


@pytest.mark.timeout(5)
async def test_multiple_guards_keep_later_conditions_unevaluated(sm_runner):
    class MultiGuardChart(StateChart):
        s1 = State(initial=True)
        s2 = State(final=True)

        go = s1.to(s2, cond=["first_guard", "second_guard"])

        def __init__(self, listeners=None):
            self.guard_calls = Counter()
            super().__init__(
                listeners=list(listeners or []), enable_decision_trace=True
            )

        def first_guard(self):
            self.guard_calls["first_guard"] += 1
            return False

        def second_guard(self):
            self.guard_calls["second_guard"] += 1
            return True

    sm = await sm_runner.start(MultiGuardChart)
    await sm_runner.send(sm, "go")

    assert sm.guard_calls == Counter(first_guard=1, second_guard=0)
    candidate = _plain(sm.decision_trace)["macrosteps"][0]["steps"][0]["candidates"][0]
    assert [guard["status"] for guard in candidate["guards"]] == [
        "false",
        "not_evaluated",
    ]
