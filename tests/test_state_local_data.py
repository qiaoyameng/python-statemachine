import asyncio

import pytest

from statemachine import State
from statemachine import StateChart
from statemachine import StateLocalData
from statemachine import StateMachine
from statemachine.exceptions import StateMachineError
from tests.conftest import _AsyncListener


def make_items(items=None):
    return [] if items is None else list(items)


class ApprovalMachine(StateChart):
    validate_trap_states: bool = False

    class review(State.Compound):
        local_data = StateLocalData(files=lambda: [])
        pending = State(initial=True, local_data=StateLocalData(items=make_items))
        done = State()
        finish = pending.to(done)

    archived = State()
    archive = review.to(archived)
    retry = archived.to(review.pending)


class TestStateLocalData:
    async def test_lifecycle_shadowing_and_reentry(self, sm_runner):
        sm = await sm_runner.start(ApprovalMachine)
        assert sm.state_data("pending")["files"] == []
        sm.state_data("pending")["files"].append("contract")
        sm.state_data("pending")["items"].append(1)
        await sm_runner.send(sm, "finish")
        assert sm.state_data("done")["files"] == ["contract"]
        with pytest.raises(KeyError):
            sm.state_data("done")["items"]
        await sm_runner.send(sm, "archive")
        await sm_runner.send(sm, "retry")
        assert sm.state_data("pending")["files"] == []

    async def test_shadowing_ancestor_and_external_self(self, sm_runner):
        class Machine(StateMachine):
            class parent(State.Compound):
                local_data = StateLocalData(value=lambda: "parent")
                class child(State.Compound):
                    local_data = StateLocalData(value=lambda: "child")
                    leaf = State(initial=True, local_data=StateLocalData(value=lambda: "leaf"))
                    change = leaf.to.itself(internal=True)
            done = State(final=True)
            leave = parent.to(done)

        sm = await sm_runner.start(Machine)
        ctx = sm.state_data("leaf")
        assert ctx["value"] == "leaf"
        ctx["value"] = "changed"
        assert sm.state_data("child")["value"] == "child"
        assert sm.state_data("parent")["value"] == "parent"
        await sm_runner.send(sm, "change")
        assert sm.state_data("leaf")["value"] == "changed"

    async def test_parallel_regions_have_isolated_frames(self, sm_runner):
        class Machine(StateChart):
            class work(State.Parallel):
                local_data = StateLocalData(scopes=lambda: {"region": "parent"})
                class left(State.Compound):
                    a = State(initial=True, local_data=StateLocalData(scopes=make_items))
                class right(State.Compound):
                    c = State(initial=True, local_data=StateLocalData(scopes=make_items))
            done = State(final=True)
            finish = work.to(done)

        sm = await sm_runner.start(Machine)
        left = sm.state_data("a")
        right = sm.state_data("c")
        left["scopes"].append("left")
        right["scopes"].append("right")
        assert left["scopes"] == ["left"]
        assert right["scopes"] == ["right"]

    async def test_internal_transition_keeps_frame_external_rebuilds(self, sm_runner):
        class Machine(StateMachine):
            waiting = State(initial=True, local_data=StateLocalData(items=make_items))
            internal = waiting.to.itself(internal=True)
            external = waiting.to.itself()
            done = State(final=True)
            stop = waiting.to(done)

        sm = await sm_runner.start(Machine)
        first = sm.state_data("waiting")
        first["items"].append("one")
        other = ApprovalMachine()
        assert other.state_data("pending")["files"] == []
        await sm_runner.send(sm, "internal")
        assert sm.state_data("waiting").frame is first.frame
        await sm_runner.send(sm, "external")
        second = sm.state_data("waiting")
        assert second is not first
        assert second["items"] == []
        with pytest.raises(StateMachineError):
            first["items"] = []

    async def test_factory_failure_is_atomic(self, sm_runner):
        def boom():
            raise RuntimeError("boom")

        class Machine(StateMachine):
            start = State(initial=True)
            class review(State.Compound):
                local_data = StateLocalData(parent=lambda: {})
                checking = State(initial=True, local_data=StateLocalData(items=boom))
            enter = start.to(review.checking)
            back = review.to(start)

        sm = await sm_runner.start(Machine)
        with pytest.raises(RuntimeError, match="boom"):
            await sm_runner.send(sm, "enter")
        assert "review" not in sm.configuration_values

    async def test_async_factory(self, sm_runner):
        async def boom():
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        class Machine(StateMachine):
            start = State(initial=True)
            class review(State.Compound):
                checking = State(initial=True, local_data=StateLocalData(items=boom))
            enter = start.to(review.checking)

        sm = await sm_runner.start(Machine)
        with pytest.raises(RuntimeError, match="boom"):
            await sm_runner.send(sm, "enter")

    async def test_async_callback_context(self, sm_runner):
        class Machine(StateChart):
            waiting = State(initial=True, local_data=StateLocalData(items=make_items))
            external = waiting.to.itself()
            done = State(final=True)
            stop = waiting.to(done)
            def on_enter_waiting(self, state_data):
                state_data["items"].append("enter")
            async def on_external(self, state_data):
                assert state_data["items"] == ["enter"]

        sm = await sm_runner.start(Machine)
        await sm_runner.send(sm, "external")
        assert sm.state_data("waiting")["items"] == ["enter"]
