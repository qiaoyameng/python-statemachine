"""Tests for the opt-in compensable microstep (saga-style rollback) mode."""

import asyncio

import pytest

from statemachine import HistoryState
from statemachine import State
from statemachine import StateChart
from statemachine import StateMachine
from statemachine import compensable
from statemachine import side_effect_free
from statemachine.compensation import ActionKind
from statemachine.compensation import get_action_meta
from statemachine.exceptions import CompensationError
from statemachine.exceptions import UnsupportedCompensableAction


class FakeResource:
    """Fake external resource recording every action/compensation call.

    Each action returns a credential (``"reserve-<n>"`` etc.) that its
    compensation must hand back, proving the engine doesn't guess the
    external side effect but uses the original return value.
    """

    def __init__(self, fail_on=()):
        self.calls = []
        self._fail_on = set(fail_on)
        self._seq = 0

    def _next(self):
        self._seq += 1
        return self._seq

    def reserve(self):
        self.calls.append(("reserve", None))
        credential = f"reserve-{self._next()}"
        self._maybe_fail("reserve")
        return credential

    def release(self, credential, **kwargs):
        self.calls.append(("release", credential))

    def deduct(self):
        self.calls.append(("deduct", None))
        credential = f"deduct-{self._next()}"
        self._maybe_fail("deduct")
        return credential

    def refund(self, credential, **kwargs):
        self.calls.append(("refund", credential))

    def create_record(self):
        self.calls.append(("create", None))
        credential = f"record-{self._next()}"
        self._maybe_fail("create")
        return credential

    def delete_record(self, credential, **kwargs):
        self.calls.append(("delete", credential))

    def bad_compensation(self, credential, **kwargs):
        self.calls.append(("bad_compensation", credential))
        raise RuntimeError("compensation failed")

    def _maybe_fail(self, action):
        if action in self._fail_on:
            raise RuntimeError(f"{action} failed")


def build_order_machine(
    resource, *, action_fail="create", compensable_mode=True, async_mode=False
):
    """s1 -> s2 microstep: reserve (on) -> deduct (before) -> create (entry).

    The entry action is side-effect-free and raises ``ActionError`` so the
    transaction always rolls back on demand.
    """

    class ActionError(RuntimeError):
        pass

    class OrderSaga(StateChart):
        compensable_microstep = compensable_mode

        s1 = State(initial=True)
        s2 = State(final=True)

        go = s1.to(s2)

        if async_mode:

            @compensable(undo=resource.refund)
            async def before_go(self):
                return resource.deduct()

            @compensable(undo="release_reservation")
            async def on_go(self):
                return resource.reserve()

            @side_effect_free
            async def on_enter_s2(self):
                if action_fail:
                    raise ActionError(action_fail)

            async def release_reservation(self, credential, **kwargs):
                resource.release(credential)

        else:

            @compensable(undo=resource.refund)
            def before_go(self):
                return resource.deduct()

            @compensable(undo="release_reservation")
            def on_go(self):
                return resource.reserve()

            @side_effect_free
            def on_enter_s2(self):
                if action_fail:
                    raise ActionError(action_fail)

            def release_reservation(self, credential, **kwargs):
                resource.release(credential)

    OrderSaga.ActionError = ActionError
    return OrderSaga


# ---------------------------------------------------------------------------
# Decorators / metadata
# ---------------------------------------------------------------------------


def test_compensable_decorator_attaches_metadata():
    def undo(credential, **kwargs):
        pass

    @compensable(undo=undo, label="reserve")
    def action():
        pass

    meta = get_action_meta(action)
    assert meta is not None
    assert meta.kind is ActionKind.COMPENSABLE
    assert meta.undo is undo
    assert meta.label == "reserve"


def test_side_effect_free_decorator_attaches_metadata():
    @side_effect_free
    def action():
        pass

    meta = get_action_meta(action)
    assert meta is not None
    assert meta.kind is ActionKind.SIDE_EFFECT_FREE


# ---------------------------------------------------------------------------
# Failure during the action (transition on / before) and during entry
# ---------------------------------------------------------------------------


def test_failure_in_entry_compensates_in_reverse_order_sync():
    resource = FakeResource()
    OrderSaga = build_order_machine(resource)
    sm = OrderSaga()

    with pytest.raises(OrderSaga.ActionError, match="create"):
        sm.send("go")

    # deduct -> reserve -> [create raises] -> release(reserve cred) -> refund(deduct cred)
    assert resource.calls == [
        ("deduct", None),
        ("reserve", None),
        ("release", "reserve-2"),
        ("refund", "deduct-1"),
    ]
    assert sm.configuration_values == {"s1"}


def test_failure_mid_action_compensates_only_completed_actions():
    resource = FakeResource(fail_on={"create"})

    class ActionError(RuntimeError):
        pass

    class MidFailure(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.refund)
        def before_go(self):
            return resource.deduct()

        @compensable(undo="release_reservation")
        def on_go(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self):
            result = resource.create_record()  # declared failure
            return result

        def release_reservation(self, credential, **kwargs):
            resource.release(credential)

    sm = MidFailure()
    with pytest.raises(RuntimeError, match="create failed"):
        sm.send("go")

    # create_record raised, so delete_record must never be called.
    names = [name for name, _ in resource.calls]
    assert names == ["deduct", "reserve", "create", "release", "refund"]
    assert all(name != "delete" for name in names)
    assert sm.configuration_values == {"s1"}


def test_each_action_compensated_at_most_once():
    resource = FakeResource()
    OrderSaga = build_order_machine(resource)
    sm = OrderSaga()

    with pytest.raises(OrderSaga.ActionError):
        sm.send("go")

    compensations = [name for name, _ in resource.calls if name in {"release", "refund", "delete"}]
    assert sorted(compensations) == ["refund", "release"]


# ---------------------------------------------------------------------------
# Continuation after rollback
# ---------------------------------------------------------------------------


def test_machine_continues_after_successful_retry():
    resource = FakeResource()

    class RetrySaga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        entry_should_fail = True

        @compensable(undo=resource.refund)
        def before_go(self):
            return resource.deduct()

        @compensable(undo="release_reservation")
        def on_go(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self):
            if self.entry_should_fail:
                raise RuntimeError("entry failed")
            return resource.create_record()

        def release_reservation(self, credential, **kwargs):
            resource.release(credential)

    sm = RetrySaga()

    with pytest.raises(RuntimeError, match="entry failed"):
        sm.send("go")
    assert sm.configuration_values == {"s1"}

    sm.entry_should_fail = False
    result = sm.send("go")

    assert sm.is_terminated
    assert sm.configuration_values == {"s2"}
    # New successful microstep performed fresh actions (no reuse of credentials).
    assert resource.calls[-3:] == [
        ("deduct", None),
        ("reserve", None),
        ("create", None),
    ]
    assert result == ["deduct-3", "reserve-4"]


# ---------------------------------------------------------------------------
# Compensation failures
# ---------------------------------------------------------------------------


def test_compensation_failure_keeps_undoing_and_wraps_original_error():
    resource = FakeResource()

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.bad_compensation)
        def before_go(self):
            return resource.deduct()

        @compensable(undo=resource.release)
        def on_go(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self):
            raise RuntimeError("entry failed")

    sm = Saga()
    with pytest.raises(CompensationError) as exc_info:
        sm.send("go")

    error = exc_info.value
    assert isinstance(error.original_error, RuntimeError)
    assert str(error.original_error) == "entry failed"
    assert len(error.compensation_errors) == 1
    assert "compensation failed" in str(error.compensation_errors[0])

    # Reverse order: reserve's release fails first; nevertheless the engine
    # keeps going and runs the deduct refund.
    names = [name for name, _ in resource.calls]
    assert names == ["deduct", "reserve", "release", "bad_compensation"]
    assert sm.configuration_values == {"s1"}


def test_multiple_compensation_failures_are_all_retained():
    resource = FakeResource()

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.bad_compensation)
        def before_go(self):
            return resource.deduct()

        @compensable(undo=resource.bad_compensation)
        def on_go(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self):
            raise RuntimeError("entry failed")

    sm = Saga()
    with pytest.raises(CompensationError) as exc_info:
        sm.send("go")

    assert len(exc_info.value.compensation_errors) == 2
    assert sm.configuration_values == {"s1"}


# ---------------------------------------------------------------------------
# Upfront rejection
# ---------------------------------------------------------------------------


def test_undeclared_action_is_rejected_before_anything_runs():
    resource = FakeResource()

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        def on_go(self):  # not declared — must be rejected up-front
            return resource.reserve()

    sm = Saga()
    with pytest.raises(UnsupportedCompensableAction):
        sm.send("go")

    assert resource.calls == []
    assert sm.configuration_values == {"s1"}


def test_undeclared_entry_action_is_rejected_before_anything_runs():
    resource = FakeResource()

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.release)
        def on_go(self):
            return resource.reserve()

        def on_enter_s2(self):  # undeclared side effect
            pass

    sm = Saga()
    with pytest.raises(UnsupportedCompensableAction):
        sm.send("go")
    assert resource.calls == []


def test_invoke_on_entered_state_is_rejected():
    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        def on_invoke_s2(self):
            return "spawned"

        @side_effect_free
        def on_enter_s2(self):
            pass

    sm = Saga()
    with pytest.raises(UnsupportedCompensableAction, match="invoke"):
        sm.send("go")


def test_invoke_on_exited_state_is_rejected():
    class Listener:
        def on_invoke_active(self):
            return "spawned"

    class Saga(StateChart):
        compensable_microstep = True

        s0 = State(initial=True)
        active = State()
        s2 = State(final=True)
        enter = s0.to(active)
        go = active.to(s2)

        @side_effect_free
        def on_enter_active(self):
            pass

        @side_effect_free
        def on_exit_active(self):
            pass

        @side_effect_free
        def on_go(self):
            pass

        @side_effect_free
        def on_enter_s2(self):
            pass

    # Entering 'active' is allowed in the default (non-compensable) mode; the
    # very first compensable microstep that would exit it is what gets rejected.
    class NonCompensable(Saga):
        compensable_microstep = False

    sm = NonCompensable(listeners=[Listener()])
    sm.send("enter")
    assert sm.configuration_values == {"active"}

    sm.__class__.compensable_microstep = True
    with pytest.raises(UnsupportedCompensableAction, match="invoke"):
        sm.send("go")
    assert sm.configuration_values == {"active"}


def test_missing_compensation_name_is_rejected():
    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo="does_not_exist")
        def on_go(self):
            return 1

        @side_effect_free
        def on_enter_s2(self):
            raise RuntimeError("boom")

    sm = Saga()
    with pytest.raises(UnsupportedCompensableAction):
        sm.send("go")


# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------


async def test_async_failure_compensates_and_restores_configuration():
    resource = FakeResource()

    class AsyncSaga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.refund)
        async def before_go(self):
            return resource.deduct()

        @compensable(undo="release_reservation")
        async def on_go(self):
            return resource.reserve()

        @side_effect_free
        async def on_enter_s2(self):
            raise RuntimeError("async entry failed")

        async def release_reservation(self, credential, **kwargs):
            resource.release(credential)

    sm = AsyncSaga()
    await sm.activate_initial_state()

    with pytest.raises(RuntimeError, match="async entry failed"):
        await sm.send("go")

    assert resource.calls == [
        ("deduct", None),
        ("reserve", None),
        ("release", "reserve-2"),
        ("refund", "deduct-1"),
    ]
    assert sm.configuration_values == {"s1"}


async def test_async_retry_succeeds_after_compensation():
    resource = FakeResource()

    class AsyncSaga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        entry_should_fail = True

        @compensable(undo=resource.refund)
        async def before_go(self):
            return resource.deduct()

        @compensable(undo="release_reservation")
        async def on_go(self):
            return resource.reserve()

        @side_effect_free
        async def on_enter_s2(self):
            if self.entry_should_fail:
                raise RuntimeError("entry failed")

        async def release_reservation(self, credential, **kwargs):
            resource.release(credential)

    sm = AsyncSaga()
    await sm.activate_initial_state()

    with pytest.raises(RuntimeError):
        await sm.send("go")
    sm.entry_should_fail = False
    await sm.send("go")
    assert sm.is_terminated
    assert sm.configuration_values == {"s2"}


async def test_async_compensation_failure_wraps_original():
    resource = FakeResource()

    class AsyncSaga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.bad_compensation)
        async def on_go(self):
            return resource.reserve()

        @side_effect_free
        async def on_enter_s2(self):
            raise RuntimeError("entry failed")

    sm = AsyncSaga()
    await sm.activate_initial_state()

    with pytest.raises(CompensationError) as exc_info:
        await sm.send("go")
    assert str(exc_info.value.original_error) == "entry failed"
    assert len(exc_info.value.compensation_errors) == 1
    assert sm.configuration_values == {"s1"}


async def test_async_cancellation_runs_compensations():
    resource = FakeResource()
    entered = asyncio.Event()

    class AsyncSaga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo="release_reservation")
        async def on_go(self):
            return resource.reserve()

        @side_effect_free
        async def on_enter_s2(self):
            entered.set()
            await asyncio.sleep(3600)  # cancelled by the test

        async def release_reservation(self, credential, **kwargs):
            await asyncio.sleep(0)
            resource.release(credential)

    sm = AsyncSaga()
    await sm.activate_initial_state()

    task = asyncio.ensure_future(sm.send("go"))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert resource.calls == [("reserve", None), ("release", "reserve-1")]
    assert sm.configuration_values == {"s1"}


async def test_async_cancellation_with_failing_compensation_preserves_cancel():
    resource = FakeResource()
    entered = asyncio.Event()

    class AsyncSaga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.bad_compensation)
        async def on_go(self):
            return resource.reserve()

        @side_effect_free
        async def on_enter_s2(self):
            entered.set()
            await asyncio.sleep(3600)

    sm = AsyncSaga()
    await sm.activate_initial_state()

    task = asyncio.ensure_future(sm.send("go"))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    # Cancellation type preserved, but compensation errors stay attached.
    assert len(exc_info.value.compensation_errors) == 1
    assert sm.configuration_values == {"s1"}


def test_sync_baseexception_triggers_compensation():
    """A non-Exception exit (e.g. a custom BaseException) is also compensated."""
    resource = FakeResource()

    class Abort(BaseException):
        pass

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.release)
        def on_go(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self):
            raise Abort("aborted")

    sm = Saga()
    with pytest.raises(Abort):
        sm.send("go")

    assert resource.calls == [("reserve", None), ("release", "reserve-1")]
    assert sm.configuration_values == {"s1"}


def test_undo_without_credential_slot_is_rejected_during_compensation():
    resource = FakeResource()
    resource.no_slot_undo = lambda: None  # type: ignore[attr-defined]

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=resource.no_slot_undo)
        def on_go(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self):
            raise RuntimeError("entry failed")

    sm = Saga()
    with pytest.raises(CompensationError) as exc_info:
        sm.send("go")
    assert isinstance(exc_info.value.compensation_errors[0], UnsupportedCompensableAction)


def test_var_positional_undo_receives_credential():
    resource = FakeResource()
    seen = []

    def undo(*args, **kwargs):
        seen.append((args, kwargs))

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        @compensable(undo=undo)
        def on_go(self, amount):
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self, amount):
            raise RuntimeError("entry failed")

    sm = Saga()
    with pytest.raises(RuntimeError):
        sm.send("go", 5)

    args, kwargs = seen[0]
    assert args[-1] == "reserve-1"  # credential appended after injected args
    assert sm.configuration_values == {"s1"}


# ---------------------------------------------------------------------------
# History + configuration rollback
# ---------------------------------------------------------------------------


def test_history_and_configuration_are_restored_on_failure():
    resource = FakeResource()

    class HistorySaga(StateChart):
        compensable_microstep = True

        class personality(State.Compound):
            smeagol = State(initial=True)
            gollum = State()
            h = HistoryState()

            dark_side = smeagol.to(gollum)
            light_side = gollum.to(smeagol)

        outside = State()
        back_out = personality.to(outside)
        return_via_history = outside.to(personality.h)

        # Entering gollum reserves; its entry fails -> roll back and leave
        # history untouched.
        @compensable(undo=resource.release)
        def on_dark_side(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_gollum(self):
            raise RuntimeError("gollum entry failed")

    sm = HistorySaga()
    assert "smeagol" in sm.configuration_values
    with pytest.raises(RuntimeError, match="gollum entry failed"):
        sm.send("dark_side")

    assert "smeagol" in sm.configuration_values
    assert sm.history_values.get("h", []) == []
    assert resource.calls == [("reserve", None), ("release", "reserve-1")]


def test_history_recorded_in_earlier_microstep_is_preserved_when_later_one_fails():
    """History recorded by a stable microstep survives a later rollback."""
    resource = FakeResource()

    class Saga(StateChart):
        compensable_microstep = True

        class personality(State.Compound):
            smeagol = State(initial=True)
            gollum = State()
            h = HistoryState()

            dark_side = smeagol.to(gollum)
            light_side = gollum.to(smeagol)

        outside = State()
        leave = personality.to(outside)
        return_via_history = outside.to(personality.h)

        # Failing re-entry through history: undo the reserve and keep the
        # previously recorded shallow history (gollum).
        @compensable(undo=resource.release)
        def on_return_via_history(self):
            return resource.reserve()

        @side_effect_free
        def on_enter_gollum(self):
            # Only the history re-entry fails; the first dark_side succeeds.
            if getattr(self, "fail_gollum_entry", False):
                raise RuntimeError("gollum entry failed")

    sm = Saga()
    assert "smeagol" in sm.configuration_values
    sm.send("dark_side")
    assert "gollum" in sm.configuration_values
    sm.send("leave")  # history now remembers gollum
    assert "outside" in sm.configuration_values

    sm.fail_gollum_entry = True
    with pytest.raises(RuntimeError, match="gollum entry failed"):
        sm.send("return_via_history")

    assert "outside" in sm.configuration_values
    remembered = sm.history_values[Saga.personality.h.id]
    assert sm.states_map["gollum"] in remembered
    # Every successful reserve was undone (the engine runs the event action
    # twice for history re-entry, so both reservations are compensated).
    reserves = [name for name, _ in resource.calls if name == "reserve"]
    releases = [cred for name, cred in resource.calls if name == "release"]
    assert len(reserves) == len(releases)
    assert set(releases) == {"reserve-1", "reserve-2"}


def test_internal_events_raised_during_failed_microstep_are_discarded():
    resource = FakeResource()

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State()
        s3 = State(final=True)
        go = s1.to(s2)
        jump = s2.to(s3)

        @compensable(undo=resource.release)
        def on_go(self):
            self.raise_("jump")
            return resource.reserve()

        @side_effect_free
        def on_enter_s2(self):
            raise RuntimeError("entry failed")

        @side_effect_free
        def on_jump(self):
            pass

        @side_effect_free
        def on_enter_s3(self):
            pass

    sm = Saga()
    with pytest.raises(RuntimeError, match="entry failed"):
        sm.send("go")

    # The internal 'jump' queued inside the failed action must not survive the
    # rollback — the machine stays in s1 instead of cascading into s3.
    assert sm.configuration_values == {"s1"}
    assert sm._engine.internal_queue.is_empty()


# ---------------------------------------------------------------------------
# Default mode semantics unchanged
# ---------------------------------------------------------------------------


def test_default_mode_propagates_exception_without_compensation():
    resource = FakeResource()

    class PlainMachine(StateMachine):
        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        def on_go(self):
            resource.reserve()

        def on_enter_s2(self):
            raise RuntimeError("plain failure")

    sm = PlainMachine()
    with pytest.raises(RuntimeError, match="plain failure"):
        sm.send("go")

    # No compensation ran in the default mode.
    assert resource.calls == [("reserve", None)]
    assert sm.configuration_values == {"s1"}


def test_default_mode_flag_is_off_by_default():
    assert StateMachine.compensable_microstep is False
    assert StateChart.compensable_microstep is False


# ---------------------------------------------------------------------------
# sm_runner: same machine on both sync and async engines
# ---------------------------------------------------------------------------


async def test_compensable_microstep_on_both_engines(sm_runner):
    resource = FakeResource()

    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        fail_entry = True

        @compensable(undo=resource.refund)
        def before_go(self):
            return resource.deduct()

        @compensable(undo="release_reservation")
        def on_go(self):
            return resource.reserve()

        def release_reservation(self, credential, **kwargs):
            resource.release(credential)

        @side_effect_free
        def on_enter_s2(self):
            if self.fail_entry:
                raise RuntimeError("entry failed")

    sm = await sm_runner.start(Saga)

    with pytest.raises(RuntimeError, match="entry failed"):
        await sm_runner.send(sm, "go")
    assert "s1" in sm.configuration_values

    sm.fail_entry = False
    await sm_runner.send(sm, "go")
    assert "s2" in sm.configuration_values


# ---------------------------------------------------------------------------
# Initial state entry
# ---------------------------------------------------------------------------


def test_initial_entry_undeclared_action_rejected():
    class Saga(StateChart):
        compensable_microstep = True

        s1 = State(initial=True)
        s2 = State(final=True)
        done = s1.to(s2)

        def on_enter_s1(self):  # undeclared side effect
            pass

    with pytest.raises(UnsupportedCompensableAction):
        Saga()


def test_initial_entry_failure_compensates():
    resource = FakeResource()

    class FailingSaga(StateChart):
        compensable_microstep = True

        s0 = State(
            initial=True,
            enter=("reserve_initial", "fail_initial"),
        )
        s2 = State(final=True)
        done = s0.to(s2)

        @compensable(undo="rollback")
        def reserve_initial(self):
            return resource.reserve()

        @side_effect_free
        def fail_initial(self):
            raise RuntimeError("initial entry failed")

        def rollback(self, credential, **kwargs):
            resource.release(credential)

    with pytest.raises(RuntimeError, match="initial entry failed"):
        FailingSaga()

    assert resource.calls == [("reserve", None), ("release", "reserve-1")]
