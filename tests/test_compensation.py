"""Tests for the compensatable microstep mode (``compensate_on_failure``).

Covers sync and async engines, plus the cancellation path: when a microstep
fails, the compensations of the actions that completed successfully within
that microstep run in reverse order (each receiving the credential returned
by its action), and the configuration and history are restored.
"""

import asyncio

import pytest
from statemachine.compensation import CompensationError
from statemachine.compensation import compensatable
from statemachine.compensation import pure
from statemachine.exceptions import AttrNotFound
from statemachine.exceptions import InvalidDefinition
from statemachine.exceptions import StateMachineError

from statemachine import Event
from statemachine import HistoryState
from statemachine import State
from statemachine import StateChart
from statemachine import StateMachine


class FakeResource:
    """Fake external resource that records the call order.

    Actions return a *credential* that the compensation later receives back,
    so compensations never have to guess which side effects were performed.
    """

    def __init__(self, fail_on=None, comp_fail_on=None):
        self.log = []
        self.fail_on = fail_on
        self.comp_fail_on = comp_fail_on

    def do(self, name):
        self.log.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"{name} failed")
        return f"credential:{name}"

    def undo(self, name, credential):
        self.log.append(f"undo:{name}:{credential}")
        if self.comp_fail_on == name:
            raise RuntimeError(f"undo {name} failed")

    def undo_credential(self, credential):
        self.log.append(f"undo:bound:{credential}")


def saga_machine_class(resource, base=StateChart, compensate=True, fail_entry=False):
    """Saga machine: reserve -> debit -> create_record, each compensatable."""

    class SagaMachine(base):
        compensate_on_failure = compensate

        s1 = State(initial=True)
        s2 = State(enter=compensatable("provision", "unprovision") if fail_entry else None)
        error_state = State()

        go = s1.to(
            s2,
            on=[
                compensatable("reserve", "release"),
                compensatable("debit", "refund"),
                compensatable("create_record", "drop_record"),
            ],
        )
        error_execution = Event(s1.to(error_state) | s2.to(error_state), id="error.execution")
        reset = error_state.to(s1)

        def reserve(self):
            return resource.do("reserve")

        def release(self, credential):
            resource.undo("reserve", credential)

        def debit(self):
            return resource.do("debit")

        def refund(self, credential):
            resource.undo("debit", credential)

        def create_record(self):
            return resource.do("create_record")

        def drop_record(self, credential):
            resource.undo("create_record", credential)

        def provision(self):
            return resource.do("provision")

        def unprovision(self, credential):
            resource.undo("provision", credential)

    return SagaMachine


class TestCompensatedMicrostep:
    async def test_action_failure_compensates_in_reverse_order(self, sm_runner):
        resource = FakeResource(fail_on="create_record")
        sm = await sm_runner.start(saga_machine_class(resource))

        await sm_runner.send(sm, "go")

        assert resource.log == [
            "reserve",
            "debit",
            "create_record",
            "undo:debit:credential:debit",
            "undo:reserve:credential:reserve",
        ]
        # configuration rolled back, then error.execution moved to error_state
        assert sm.configuration == {sm.error_state}

    async def test_entry_failure_compensates_transition_actions(self, sm_runner):
        resource = FakeResource(fail_on="provision")
        sm = await sm_runner.start(saga_machine_class(resource, fail_entry=True))

        await sm_runner.send(sm, "go")

        assert resource.log == [
            "reserve",
            "debit",
            "create_record",
            "provision",
            "undo:create_record:credential:create_record",
            "undo:debit:credential:debit",
            "undo:reserve:credential:reserve",
        ]
        assert sm.configuration == {sm.error_state}

    async def test_successful_microstep_does_not_compensate(self, sm_runner):
        resource = FakeResource()
        sm = await sm_runner.start(saga_machine_class(resource))

        await sm_runner.send(sm, "go")

        assert resource.log == ["reserve", "debit", "create_record"]
        assert sm.configuration == {sm.s2}

    async def test_exit_action_is_compensated(self, sm_runner):
        resource = FakeResource(fail_on="reserve")

        class ExitSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True, exit=compensatable("checkout", "uncheckout"))
            s2 = State(final=True)

            go = s1.to(s2, on=compensatable("reserve", "release"))

            def checkout(self):
                return resource.do("checkout")

            def uncheckout(self, credential):
                resource.undo("checkout", credential)

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

        sm = await sm_runner.start(ExitSC)
        await sm_runner.send(sm, "go")

        # reserve failed: only the completed exit action is compensated, exactly once
        assert resource.log == ["checkout", "reserve", "undo:checkout:credential:checkout"]
        assert sm.configuration == {sm.s1}

    async def test_compensation_failure_continues_and_preserves_all_errors(self, sm_runner):
        resource = FakeResource(fail_on="create_record", comp_fail_on="debit")
        captured = []

        class CompFailSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)
            error_state = State(final=True)

            go = s1.to(
                s2,
                on=[
                    compensatable("reserve", "release"),
                    compensatable("debit", "refund"),
                    compensatable("create_record", "drop_record"),
                ],
            )
            error_execution = s1.to(error_state, on="capture")

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

            def create_record(self):
                return resource.do("create_record")

            def drop_record(self, credential):
                resource.undo("create_record", credential)

            @pure
            def capture(self, error):
                captured.append(error)

        sm = await sm_runner.start(CompFailSC)
        await sm_runner.send(sm, "go")

        # a failing compensation does not stop the remaining ones
        assert resource.log == [
            "reserve",
            "debit",
            "create_record",
            "undo:debit:credential:debit",
            "undo:reserve:credential:reserve",
        ]
        assert len(captured) == 1
        error = captured[0]
        assert isinstance(error, CompensationError)
        assert isinstance(error.original, RuntimeError)
        assert "create_record failed" in str(error.original)
        assert len(error.compensation_errors) == 1
        assert "undo debit failed" in str(error.compensation_errors[0])
        assert sm.configuration == {sm.error_state}

    async def test_machine_continues_after_compensation(self, sm_runner):
        resource = FakeResource(fail_on="create_record")
        sm = await sm_runner.start(saga_machine_class(resource))

        await sm_runner.send(sm, "go")
        assert sm.configuration == {sm.error_state}

        await sm_runner.send(sm, "reset")
        assert sm.configuration == {sm.s1}

        resource.fail_on = None
        resource.log.clear()
        await sm_runner.send(sm, "go")
        assert resource.log == ["reserve", "debit", "create_record"]
        assert sm.configuration == {sm.s2}

    async def test_default_mode_keeps_existing_semantics(self, sm_runner):
        resource = FakeResource(fail_on="debit")
        sm = await sm_runner.start(saga_machine_class(resource, compensate=False))

        await sm_runner.send(sm, "go")

        # no compensation in default mode; existing error.execution semantics
        assert resource.log == ["reserve", "debit", "create_record"]
        assert sm.configuration == {sm.error_state}


class TestStateMachineBase:
    """StateMachine keeps ``catch_errors_as_events=False``: errors propagate."""

    def test_raises_original_and_restores_configuration(self):
        resource = FakeResource(fail_on="debit")
        sm = saga_machine_class(resource, base=StateMachine)()

        with pytest.raises(RuntimeError, match="debit failed"):
            sm.send("go")

        assert resource.log == ["reserve", "debit", "undo:reserve:credential:reserve"]
        assert sm.configuration == {sm.s1}

    def test_compensation_failure_raises_compensation_error(self):
        resource = FakeResource(fail_on="create_record", comp_fail_on="debit")
        sm = saga_machine_class(resource, base=StateMachine)()

        with pytest.raises(CompensationError) as exc_info:
            sm.send("go")

        error = exc_info.value
        assert isinstance(error.original, RuntimeError)
        assert "create_record failed" in str(error.original)
        assert len(error.compensation_errors) == 1
        assert "undo debit failed" in str(error.compensation_errors[0])
        # remaining compensations still ran
        assert resource.log == [
            "reserve",
            "debit",
            "create_record",
            "undo:debit:credential:debit",
            "undo:reserve:credential:reserve",
        ]
        assert sm.configuration == {sm.s1}

    def test_entry_failure_compensates(self):
        resource = FakeResource(fail_on="provision")
        sm = saga_machine_class(resource, base=StateMachine, fail_entry=True)()

        with pytest.raises(RuntimeError, match="provision failed"):
            sm.send("go")

        assert resource.log == [
            "reserve",
            "debit",
            "create_record",
            "provision",
            "undo:create_record:credential:create_record",
            "undo:debit:credential:debit",
            "undo:reserve:credential:reserve",
        ]
        assert sm.configuration == {sm.s1}


class TestValidation:
    async def test_unmarked_action_rejected_before_execution(self, sm_runner):
        resource = FakeResource()

        class BadSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(s2, on=[compensatable("reserve", "release"), "create_record"])

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            def create_record(self):
                return resource.do("create_record")

        sm = await sm_runner.start(BadSC)
        with pytest.raises(InvalidDefinition, match="Compensatable microstep rejected"):
            await sm_runner.send(sm, "go")

        # rejected before executing anything
        assert resource.log == []
        assert sm.configuration == {sm.s1}

    async def test_unmarked_entry_callback_rejected(self, sm_runner):
        resource = FakeResource()

        class BadEntrySC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(enter="provision", final=True)

            go = s1.to(s2, on=compensatable("reserve", "release"))

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            def provision(self):
                return resource.do("provision")

        sm = await sm_runner.start(BadEntrySC)
        with pytest.raises(InvalidDefinition, match="Compensatable microstep rejected"):
            await sm_runner.send(sm, "go")

        assert resource.log == []
        assert sm.configuration == {sm.s1}

    async def test_pure_callbacks_are_accepted(self, sm_runner):
        resource = FakeResource()

        class PureSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(s2, on=compensatable("reserve", "release"))

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            @pure
            def on_exit_s1(self):
                resource.log.append("exit:s1")

            @pure
            def on_enter_s2(self):
                resource.log.append("enter:s2")

        sm = await sm_runner.start(PureSC)
        await sm_runner.send(sm, "go")

        assert resource.log == ["exit:s1", "reserve", "enter:s2"]
        assert sm.configuration == {sm.s2}

    async def test_invoke_rejected_before_execution(self, sm_runner):
        resource = FakeResource()

        class InvokeSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(invoke="child_factory", final=True)

            go = s1.to(s2, on=compensatable("reserve", "release"))

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            def child_factory(self):
                return None

        sm = await sm_runner.start(InvokeSC)
        with pytest.raises(InvalidDefinition, match="invoke"):
            await sm_runner.send(sm, "go")

        assert resource.log == []
        assert sm.configuration == {sm.s1}

    def test_compensation_name_not_found(self):
        class BadSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(s2, on=compensatable("reserve", "missing_compensation"))

            def reserve(self):
                return "credential:reserve"

        with pytest.raises(AttrNotFound):
            BadSC()


class TestHistoryRestore:
    async def test_history_and_configuration_restored(self, sm_runner):
        resource = FakeResource(fail_on="debit")

        class HistorySC(StateChart):
            compensate_on_failure = True

            class p(State.Compound):
                a = State(initial=True)
                b = State()
                h = HistoryState()

                move = a.to(b)

            out = State(initial=True)

            enter_p = out.to(p)
            leave = p.to(out)
            back = out.to(p.h)
            go = p.b.to(
                out,
                on=[compensatable("reserve", "release"), compensatable("debit", "refund")],
            )

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

        sm = await sm_runner.start(HistorySC)
        await sm_runner.send(sm, "enter_p")
        await sm_runner.send(sm, "leave")  # records history p.h = [a]
        await sm_runner.send(sm, "back")  # re-enters a via history
        await sm_runner.send(sm, "move")  # a -> b
        assert [s.id for s in sm.history_values["h"]] == ["a"]

        await sm_runner.send(sm, "go")  # fails on debit

        assert resource.log == ["reserve", "debit", "undo:reserve:credential:reserve"]
        assert sm.configuration == {sm.p, sm.b}
        # history restored to the pre-microstep snapshot
        assert [s.id for s in sm.history_values["h"]] == ["a"]

        # the machine keeps processing events per contract
        await sm_runner.send(sm, "leave")  # records history p.h = [b]
        await sm_runner.send(sm, "back")  # re-enters b via history
        assert sm.configuration == {sm.p, sm.b}


class TestCancellationAndBaseExceptions:
    async def test_async_cancellation_compensates(self):
        resource = FakeResource()
        started = asyncio.Event()

        class CancelSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(
                s2,
                on=[compensatable("reserve", "release"), compensatable("block", "unblock")],
            )

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            async def block(self):
                started.set()
                await asyncio.sleep(60)
                return resource.do("block")  # pragma: no cover - never reached

            def unblock(self, credential):  # pragma: no cover - block never completes
                resource.undo("block", credential)

        sm = CancelSC()
        await sm.activate_initial_state()
        task = asyncio.create_task(sm.send("go"))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # cancellation compensated the completed action and restored the configuration
        assert resource.log == ["reserve", "undo:reserve:credential:reserve"]
        assert sm.configuration == {sm.s1}

    def test_sync_base_exception_compensates_and_propagates(self):
        resource = FakeResource()

        class InterruptSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(s2, on=[compensatable("reserve", "release"), "boom"])

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            @pure
            def boom(self):
                raise KeyboardInterrupt

        sm = InterruptSC()
        with pytest.raises(KeyboardInterrupt):
            sm.send("go")

        assert resource.log == ["reserve", "undo:reserve:credential:reserve"]
        assert sm.configuration == {sm.s1}


class TestAsyncCompensations:
    async def test_async_engine_awaits_async_compensation(self):
        resource = FakeResource(fail_on="debit")

        class AsyncCompSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(
                s2,
                on=[compensatable("reserve", "release"), compensatable("debit", "refund")],
            )

            async def reserve(self):
                return resource.do("reserve")

            async def release(self, credential):
                resource.undo("reserve", credential)

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

        sm = AsyncCompSC()
        await sm.activate_initial_state()
        await sm.send("go")

        assert resource.log == ["reserve", "debit", "undo:reserve:credential:reserve"]
        assert sm.configuration == {sm.s1}

    def test_sync_engine_drives_async_compensation(self):
        resource = FakeResource(fail_on="debit")

        class SyncCompSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(
                s2,
                on=[compensatable("reserve", "release"), compensatable("debit", "refund")],
            )

            def reserve(self):
                return resource.do("reserve")

            async def release(self, credential):
                resource.undo("reserve", credential)

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

        sm = SyncCompSC()
        sm.send("go")

        assert resource.log == ["reserve", "debit", "undo:reserve:credential:reserve"]
        assert sm.configuration == {sm.s1}

    async def test_sync_engine_rejects_async_compensation_with_running_loop(self):
        resource = FakeResource(fail_on="debit")
        captured = []

        class MixedSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)
            error_state = State(final=True)

            go = s1.to(
                s2,
                on=[compensatable("reserve", "release"), compensatable("debit", "refund")],
            )
            error_execution = s1.to(error_state, on="capture")

            def reserve(self):
                return resource.do("reserve")

            async def release(self, credential):
                resource.undo("reserve", credential)  # pragma: no cover - never awaited

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

            @pure
            def capture(self, error):
                captured.append(error)

        sm = MixedSC()
        sm.send("go")

        assert resource.log == ["reserve", "debit"]
        assert len(captured) == 1
        error = captured[0]
        assert isinstance(error, CompensationError)
        assert len(error.compensation_errors) == 1
        assert "async compensation" in str(error.compensation_errors[0])
        assert sm.configuration == {sm.error_state}


class TestCompensationResolution:
    def test_compensation_as_bound_callable(self):
        resource = FakeResource(fail_on="debit")

        class BoundSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(
                s2,
                on=[
                    compensatable("reserve", resource.undo_credential),
                    compensatable("debit", "refund"),
                ],
            )

            def reserve(self):
                return resource.do("reserve")

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

        sm = BoundSC()
        sm.send("go")

        assert resource.log == ["reserve", "debit", "undo:bound:credential:reserve"]

    def test_compensation_as_plain_function(self):
        resource = FakeResource(fail_on="debit")

        def undo_reserve(credential):
            resource.log.append(f"undo:fn:{credential}")

        class FnSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            go = s1.to(
                s2,
                on=[compensatable("reserve", undo_reserve), compensatable("debit", "refund")],
            )

            def reserve(self):
                return resource.do("reserve")

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

        sm = FnSC()
        sm.send("go")

        assert resource.log == ["reserve", "debit", "undo:fn:credential:reserve"]

    def test_compensation_as_unbound_method(self):
        resource = FakeResource(fail_on="debit")

        class UnboundSC(StateChart):
            compensate_on_failure = True

            s1 = State(initial=True)
            s2 = State(final=True)

            def reserve(self):
                return resource.do("reserve")

            def release(self, credential):
                resource.undo("reserve", credential)

            def debit(self):
                return resource.do("debit")

            def refund(self, credential):
                resource.undo("debit", credential)

            go = s1.to(
                s2,
                on=[
                    compensatable("reserve", release),
                    compensatable("debit", "refund"),
                ],
            )

        sm = UnboundSC()
        sm.send("go")

        assert resource.log == ["reserve", "debit", "undo:reserve:credential:reserve"]


class TestCompensationError:
    def test_attributes_and_message(self):
        original = RuntimeError("debit failed")
        compensation_errors = [ValueError("refund failed")]
        error = CompensationError(original, compensation_errors)

        assert error.original is original
        assert error.compensation_errors == compensation_errors
        assert "debit failed" in str(error)
        assert "refund failed" in str(error)

    def test_is_state_machine_error(self):
        error = CompensationError(RuntimeError("x"), [])
        assert isinstance(error, StateMachineError)
