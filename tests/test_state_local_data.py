import asyncio
import pickle

import pytest

from statemachine import HistoryState
from statemachine import State
from statemachine import StateChart
from statemachine import StateMachine
from statemachine.exceptions import InvalidDefinition
from statemachine.exceptions import StateMachineError


def make_review_data():
    return {"note": "new", "tags": []}


def make_pickle_data():
    return {"note": "draft"}


class PickleLocalDataMachine(StateMachine):
    waiting = State(initial=True, local_data=make_pickle_data)
    done = State(final=True)
    finish = waiting.to(done)


def test_factory_creates_independent_frame_per_machine_and_reentry():
    created = []

    def factory():
        value = object()
        created.append(value)
        return {"value": value}

    class ReentryMachine(StateMachine):
        waiting = State(initial=True, local_data=factory)
        done = State(final=True)
        loop = waiting.to.itself()
        finish = waiting.to(done)

    first = ReentryMachine()
    second = ReentryMachine()

    assert created[0] is not created[1]
    assert first.waiting.is_active
    assert second.waiting.is_active

    first.send("loop")
    assert created[2] is not created[0]
    assert len(created) == 3


def test_child_reads_ancestor_frame_and_writes_shadow_parent():
    captured = []

    class ReviewMachine(StateChart):
        class review(State.Compound, local_data=lambda: {"note": "parent"}):
            waiting = State(initial=True)
            done = State(final=True)
            finish = waiting.to(done)

        outside = State(final=True)
        start = State(initial=True).to(review)
        review.to(outside)

        def on_enter_waiting(self, local_data):
            assert local_data["note"] == "parent"
            local_data["note"] = "child"
            type(self).captured.append(local_data)

        def on_exit_review(self, local_data):
            assert local_data["note"] == "parent"

    ReviewMachine.captured = captured
    sm = ReviewMachine()
    sm.send("start")

    assert captured[0]["note"] == "child"
    sm.send("finish")


def test_released_child_frame_keeps_ancestor_snapshot():
    released = []

    class SnapshotMachine(StateChart):
        class review(State.Compound, local_data=lambda: {"note": "parent"}):
            waiting = State(initial=True)
            done = State(final=True)
            finish = waiting.to(done)

            def on_exit_waiting(self, local_data):
                type(self).released.append(local_data)

        outside = State(final=True)
        start = State(initial=True).to(review)
        close = review.to(outside)

        def on_exit_review(self, local_data):
            local_data["note"] = "changed after child exit"

    SnapshotMachine.released = released
    sm = SnapshotMachine()
    sm.send("start")
    sm.send("finish")
    sm.send("close")

    assert released[0]["note"] == "parent"


def test_dict_factory_declaration_initializes_each_key_for_each_frame():
    class FactoryMapMachine(StateMachine):
        waiting = State(initial=True, local_data={"items": list, "count": int})
        done = State(final=True)
        finish = waiting.to(done)

        def on_finish(self, local_data):
            local_data["items"].append("ok")
            return local_data["items"], local_data["count"]

    first = FactoryMapMachine()
    second = FactoryMapMachine()

    assert first.send("finish") == (["ok"], 0)
    assert second.waiting.is_active


async def test_local_data_injection_on_sync_and_async_engines(sm_runner):
    class Approval(StateChart):
        waiting = State(initial=True, local_data=lambda: {"note": "draft"})
        approved = State(final=True, local_data=lambda: {"note": ""})
        approve = waiting.to(approved)

        def on_enter_approved(self, local_data):
            local_data["note"] = "approved"
            type(self).result = local_data["note"]

    Approval.result = None

    sm = await sm_runner.start(Approval)
    await sm_runner.send(sm, "approve")

    assert Approval.result == "approved"


def test_local_data_frames_are_transient_runtime_state():
    sm = PickleLocalDataMachine()
    restored = pickle.loads(pickle.dumps(sm))

    assert restored.waiting.is_active
    restored.send("finish")
    assert restored.done.is_active


def test_local_data_rejects_non_factory_declaration():
    with pytest.raises(InvalidDefinition, match="factory"):
        State(local_data={"note": "shared mutable default"})


def test_local_data_factory_must_return_mapping():
    class InvalidFactoryMachine(StateMachine):
        idle = State(initial=True)
        waiting = State(local_data=lambda: ("not", "mapping"))
        enter = idle.to(waiting)
        back = waiting.to(idle)

    sm = InvalidFactoryMachine()
    with pytest.raises(InvalidDefinition, match="mapping"):
        sm.send("enter")

    assert sm.idle.is_active


def test_released_frame_reference_is_read_only():
    released = []

    class ReleasedFrameMachine(StateChart):
        waiting = State(initial=True, local_data=make_review_data)
        done = State(final=True)
        finish = waiting.to(done)

        def on_finish(self, local_data):
            type(self).released.append(local_data)

    ReleasedFrameMachine.released = released
    sm = ReleasedFrameMachine()
    sm.send("finish")

    assert released[0]["note"] == "new"
    with pytest.raises(StateMachineError, match="released"):
        released[0]["note"] = "stale"
    with pytest.raises(StateMachineError, match="released"):
        del released[0]["note"]


def test_transition_on_callback_gets_read_only_source_frame():
    class SourceFrameMachine(StateChart):
        waiting = State(initial=True, local_data=lambda: {"note": "old"})
        done = State(final=True)
        finish = waiting.to(done)

        def on_finish(self, local_data):
            assert local_data["note"] == "old"
            with pytest.raises(StateMachineError, match="released"):
                local_data["note"] = "new"

    sm = SourceFrameMachine()
    sm.send("finish")

    assert sm.done.is_active


def test_parallel_region_frames_do_not_override_each_other():
    region_values = {}
    parent_after_exit = {}

    def enter_region(state, local_data):
        local_data["shared"] = state.parent.id
        region_values[state.parent.id] = local_data["shared"]

    class ParallelLocalDataMachine(StateChart):
        class work(State.Parallel, local_data=lambda: {"shared": "parent"}):
            class left(State.Compound):
                running_left = State(initial=True, enter=enter_region)
                left_done = State(final=True)
                finish = running_left.to(left_done)

            class right(State.Compound):
                running_right = State(initial=True, enter=enter_region)
                right_done = State(final=True)
                finish = running_right.to(right_done)

        completed = State(final=True)
        idle = State(initial=True)
        begin = idle.to(work)
        done_state_work = work.to(completed)

        def on_exit_work(self, local_data):
            type(self).parent_after_exit["shared"] = local_data["shared"]

    ParallelLocalDataMachine.parent_after_exit = parent_after_exit
    sm = ParallelLocalDataMachine()
    sm.send("begin")

    assert region_values == {"left": "left", "right": "right"}

    sm.send("finish")
    sm.send("done.state.work")
    assert parent_after_exit == {"shared": "parent"}


def test_internal_transition_reuses_frame_external_self_transition_rebuilds_it():
    identities = []

    class SelfTransitionMachine(StateChart):
        waiting = State(initial=True, local_data=lambda: {"value": object()})
        done = State(final=True)
        internal_touch = waiting.to.itself(internal=True)
        external_touch = waiting.to.itself()
        finish = waiting.to(done)

        def on_internal_touch(self, local_data):
            type(self).identities.append(local_data["value"])

        def on_external_touch(self, local_data):
            type(self).identities.append(local_data["value"])

    SelfTransitionMachine.identities = identities
    sm = SelfTransitionMachine()

    sm.send("internal_touch")
    sm.send("internal_touch")
    sm.send("external_touch")

    assert identities[0] is identities[1]
    assert identities[2] is not identities[0]
    sm.send("finish")


def test_history_reentry_creates_new_frames():
    identities = []

    class HistoryMachine(StateChart):
        outside = State(initial=True)

        class review(State.Compound):
            h = HistoryState()
            waiting = State(local_data=lambda: {"value": object()})
            checked = State()
            check = waiting.to(checked)
            h.to(waiting)

        enter = outside.to(review.h)
        leave = review.to(outside)
        return_ = outside.to(review.h)

        def on_enter_waiting(self, local_data):
            type(self).identities.append(local_data["value"])

    HistoryMachine.identities = identities
    sm = HistoryMachine()

    sm.send("enter")
    sm.send("leave")
    sm.send("return_")

    assert identities[0] is not identities[1]


def test_failed_factory_rolls_back_all_new_frames():
    left_values = []
    left_frames = []
    attempts = 0

    def left_factory():
        value = object()
        left_frames.append(value)
        return {"ok": True, "value": value}

    def enter_left(local_data):
        left_values.append((local_data["value"], local_data["ok"]))

    def right_factory():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first initialization fails")
        return {"ready": True}

    class FailedFactoryMachine(StateChart):
        outside = State(initial=True)

        class work(State.Parallel):
            class left(State.Compound):
                running = State(
                    initial=True,
                    local_data=left_factory,
                    enter=enter_left,
                )
                left_done = State(final=True)
                finish = running.to(left_done)

            class right(State.Compound):
                running_right = State(initial=True, local_data=right_factory)
                right_done = State(final=True)
                finish = running_right.to(right_done)

        enter = outside.to(work)
        error_execution = outside.to(outside)

    sm = FailedFactoryMachine()
    sm.send("enter")

    assert sm.outside.is_active
    assert left_values == []

    sm.send("enter")

    assert sm.work.is_active
    assert len(left_frames) == 2
    assert left_frames[0] is not left_frames[1]
    assert len(left_values) == 1
    assert left_values[0][1] is True
    assert left_values[0][0] is left_frames[1]

    sm.send("finish")


@pytest.mark.asyncio
async def test_async_context_suspended_across_exit_and_reentry_rejects_stale_write():
    class AsyncReentryMachine(StateChart):
        waiting = State(initial=True, local_data=lambda: {"value": 0})
        done = State(final=True)
        loop = waiting.to.itself()
        finish = waiting.to(done)

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.writer = None
            self.stale_error = None
            super().__init__()

        async def _write_after_reentry(self, local_data):
            self.started.set()
            await self.release.wait()
            try:
                local_data["value"] = 1
            except StateMachineError as error:
                self.stale_error = error

        async def on_loop(self, local_data):
            if self.writer is None:
                self.writer = asyncio.create_task(self._write_after_reentry(local_data))
            await self.started.wait()

    sm = AsyncReentryMachine()
    await sm.activate_initial_state()

    await sm.send("loop")
    await sm.send("loop")

    sm.release.set()
    await sm.writer

    assert isinstance(sm.stale_error, StateMachineError)
    await sm.send("finish")
