"""Explicit, quiescent snapshot/restore API (workflow continuation after restart).

Snapshots are JSON-safe documents: they never contain functions, locks, futures
or arbitrary pickled objects. Restore only accepts stable configurations and
validates the document *before* any user callback can run.
"""

import json

import pytest

from statemachine import HistoryState
from statemachine import State
from statemachine import StateChart
from statemachine import StateMachine
from statemachine import restore_snapshot
from statemachine import SnapshotDefinitionMismatch
from statemachine import SnapshotError
from statemachine import SnapshotNotStable
from statemachine.snapshot import SNAPSHOT_VERSION
from tests.machines.history.deep_memory_of_moria import DeepMemoryOfMoria
from tests.machines.history.gollum_personality import GollumPersonality
from tests.machines.parallel.war_of_the_ring import WarOfTheRing


class Tracer:
    """Listener recording the generic lifecycle callback trace."""

    def __init__(self):
        self.trace: list[str] = []

    def on_enter_state(self, state):
        self.trace.append(f"enter:{state.id}")

    def on_exit_state(self, state):
        self.trace.append(f"exit:{state.id}")

    def before_transition(self, event, state):
        self.trace.append(f"before:{event.name}:{state.id}")

    def on_transition(self, event, state):
        self.trace.append(f"on:{event.name}:{state.id}")

    def after_transition(self, event, state):
        self.trace.append(f"after:{event.name}:{state.id}")


class NestedCompound(StateChart):
    """compound > compound > atomic, plus an outside sibling and a final state."""

    class outer(State.Compound):
        class inner(State.Compound):
            a = State(initial=True)
            b = State()

            go_b = a.to(b)

        h = HistoryState()

    outside = State()
    finished = State(final=True)

    leave = outer.inner.to(outside)
    return_via_history = outside.to(outer.h)
    finish = outside.to(finished)


class OrderModel:
    def __init__(self):
        self.state = None
        self.order_id = "ORD-1"
        self.amount = 100
        self.unjsonable = object()


class OrderMachine(StateMachine):
    draft = State(initial=True)
    review = State()
    done = State(final=True)

    submit = draft.to(review)
    approve = review.to(done)


class BlockingInvoke(StateChart):
    working = State(initial=True, invoke="do_work")
    done = State(final=True)
    finish = working.to(done)

    def do_work(self, **kwargs):
        self._done_event.wait(1.0)

    def __init__(self):
        import threading

        self._done_event = threading.Event()
        super().__init__()


# ---------------------------------------------------------------------------
# Snapshot document basics
# ---------------------------------------------------------------------------


def test_snapshot_document_is_json_safe_and_versioned():
    sm = OrderMachine(OrderModel())
    snapshot = sm.snapshot(model_fields=["order_id", "amount"])

    assert snapshot["version"] == SNAPSHOT_VERSION
    assert snapshot["machine"] == "OrderMachine"
    assert snapshot["fingerprint"]
    assert snapshot["state_field"] == "state"
    assert snapshot["terminated"] is False
    assert snapshot["configuration"] == ["draft"]
    assert snapshot["history"] == {}
    assert snapshot["model"] == {"order_id": "ORD-1", "amount": 100}

    # The document round-trips through plain JSON (no functions/locks/pickles).
    encoded = json.dumps(snapshot)
    assert json.loads(encoded) == snapshot


def test_snapshot_rejects_non_json_model_data():
    sm = OrderMachine(OrderModel())
    with pytest.raises(SnapshotError, match="JSON"):
        sm.snapshot(model_fields=["unjsonable"])


def test_snapshot_without_model_fields_exports_no_model_data():
    sm = OrderMachine(OrderModel())
    assert sm.snapshot()["model"] == {}


def test_snapshot_rejects_unknown_model_field():
    sm = OrderMachine(OrderModel())
    with pytest.raises(SnapshotError, match="order_id_missing"):
        sm.snapshot(model_fields=["order_id_missing"])


# ---------------------------------------------------------------------------
# Quiescence gates
# ---------------------------------------------------------------------------


def test_snapshot_rejected_when_external_queue_nonempty():
    from statemachine.event import BoundEvent

    sm = OrderMachine()
    # Queue a (delayed) event without running the processing loop.
    BoundEvent(id="submit", delay=60_000, _sm=sm).put(send_id="pending-submit")
    assert not sm._engine.external_queue.is_empty()
    with pytest.raises(SnapshotNotStable, match="external event queue is not empty"):
        sm.snapshot()
    sm.cancel_event("pending-submit")


def test_snapshot_rejected_with_active_invoke():
    sm = BlockingInvoke()
    try:
        with pytest.raises(SnapshotNotStable, match="active invocation"):
            sm.snapshot()
    finally:
        sm._done_event.set()


def test_snapshot_rejected_from_inside_callback():
    class ReentrantMachine(StateMachine):
        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

        def on_transition(self, event, state):
            with pytest.raises(SnapshotNotStable, match="callback or transition"):
                self.snapshot()

    sm = ReentrantMachine()
    sm.go()


# ---------------------------------------------------------------------------
# Restore validation (must fail before any user callback runs)
# ---------------------------------------------------------------------------


def _make_order_snapshot(**overrides):
    sm = OrderMachine(OrderModel())
    sm.submit()
    snapshot = sm.snapshot(model_fields=["order_id", "amount"])
    snapshot.update(overrides)
    return snapshot


def test_restore_rejects_unknown_version():
    snapshot = _make_order_snapshot(version="nope")
    with pytest.raises(SnapshotDefinitionMismatch, match="[Vv]ersion"):
        OrderMachine.restore(snapshot, OrderModel())


def test_restore_rejects_machine_id_mismatch():
    snapshot = _make_order_snapshot(machine="SomethingElse")
    with pytest.raises(SnapshotDefinitionMismatch, match="was created for machine"):
        OrderMachine.restore(snapshot, OrderModel())


def test_restore_rejects_tampered_fingerprint():
    snapshot = _make_order_snapshot(fingerprint="deadbeef" * 8)
    with pytest.raises(SnapshotDefinitionMismatch, match="fingerprint"):
        OrderMachine.restore(snapshot, OrderModel())


@pytest.mark.parametrize(
    "configuration, message",
    [
        (["nope"], "Unknown state"),
        (["done"], "final"),
    ],
)
def test_restore_rejects_invalid_configuration(configuration, message):
    snapshot = _make_order_snapshot(configuration=configuration)
    with pytest.raises(SnapshotDefinitionMismatch, match=message):
        OrderMachine.restore(snapshot, OrderModel())


def test_restore_rejects_sibling_states_in_compound():
    sm = NestedCompound()
    snapshot = sm.snapshot()
    snapshot["configuration"] = ["outer", "inner", "a", "b"]
    with pytest.raises(SnapshotDefinitionMismatch, match="sibling states"):
        NestedCompound.restore(snapshot)


def test_restore_rejects_parallel_region_missing():
    sm = WarOfTheRing()
    snapshot = sm.snapshot()
    active = list(snapshot["configuration"])
    assert "war" in active
    # Drop a leaf of one region: 'shire' lives in frodos_quest.
    tampered = [state_id for state_id in active if state_id != "shire"]
    snapshot["configuration"] = tampered
    with pytest.raises(SnapshotDefinitionMismatch, match="parallel state"):
        WarOfTheRing.restore(snapshot)


def test_restore_rejects_history_pseudostate_in_configuration():
    sm = GollumPersonality()
    snapshot = sm.snapshot()
    snapshot["configuration"] = ["outside", "h"]
    with pytest.raises(SnapshotDefinitionMismatch, match="[Hh]istory"):
        GollumPersonality.restore(snapshot)


def test_restore_rejects_history_with_unknown_state():
    sm = GollumPersonality()
    sm.dark_side()
    sm.leave()
    snapshot = sm.snapshot()
    assert snapshot["history"] == {"h": ["gollum"]}
    snapshot["history"] = {"h": ["unknown"]}
    with pytest.raises(SnapshotDefinitionMismatch, match="[Hh]istory"):
        GollumPersonality.restore(snapshot)


def test_restore_rejects_history_with_ancestor_mismatch():
    sm = GollumPersonality()
    sm.dark_side()
    sm.leave()
    snapshot = sm.snapshot()
    snapshot["history"] = {"h": ["outside"]}  # outside is not a child of personality
    with pytest.raises(SnapshotDefinitionMismatch, match="[Hh]istory"):
        GollumPersonality.restore(snapshot)


def test_restore_does_not_invoke_callbacks_or_guess_initial_state():
    """All validation happens before a single user callback could run."""

    sm = OrderMachine(OrderModel())
    sm.submit()

    class GuardTracer(Tracer):
        def on_enter_state(self, state):
            raise AssertionError("no callbacks must run on restore")

    # Tampered fingerprint: rejected before the machine (and its callbacks) exist.
    tampered = sm.snapshot(model_fields=["order_id"])
    tampered["fingerprint"] = "0" * 64
    with pytest.raises(SnapshotDefinitionMismatch):
        OrderMachine.restore(tampered, OrderModel(), listeners=[GuardTracer()])

    # A valid restore also performs no entry/exit actions.
    good = sm.snapshot(model_fields=["order_id"])
    tracer = GuardTracer()
    restored = OrderMachine.restore(good, OrderModel(), listeners=[tracer])
    assert tracer.trace == []
    assert "review" in restored.configuration_values
    assert restored.model.order_id == "ORD-1"


# ---------------------------------------------------------------------------
# Round trips: configuration + callback trace equivalence
# ---------------------------------------------------------------------------


def test_roundtrip_hierarchical_compound():
    model = OrderModel()
    sm = NestedCompound(model)
    sm.go_b()
    snapshot = sm.snapshot(model_fields=["order_id"])

    assert set(snapshot["configuration"]) == {"outer", "inner", "b"}
    restored = NestedCompound.restore(snapshot, OrderModel())
    assert {s.id for s in restored.configuration} == {"outer", "inner", "b"}
    assert restored.model.order_id == "ORD-1"
    # Restore performed no entry/exit work.
    assert "b" in restored.configuration_values

    restored.leave()
    assert {s.id for s in restored.configuration} == {"outside"}
    restored.finish()
    assert {s.id for s in restored.configuration} == {"finished"}
    assert restored.is_terminated


def test_roundtrip_hierarchical_compound_trace_equivalence():
    sm = NestedCompound()
    sm.go_b()
    snapshot = sm.snapshot()

    original_tracer = Tracer()
    original = NestedCompound()
    original.go_b()
    original.add_listener(original_tracer)

    restored_tracer = Tracer()
    restored = NestedCompound.restore(snapshot, listeners=[restored_tracer])
    assert restored_tracer.trace == []

    for event in ["leave", "return_via_history", "go_b", "leave", "finish"]:
        original.send(event)
        restored.send(event)
        assert {s.id for s in restored.configuration} == {
            s.id for s in original.configuration
        }

    assert restored_tracer.trace == original_tracer.trace
    assert restored.is_terminated and original.is_terminated


def test_roundtrip_parallel_regions():
    sm = WarOfTheRing()
    sm.journey()
    snapshot = sm.snapshot()

    restored = WarOfTheRing.restore(snapshot)
    assert {s.id for s in restored.configuration} == {
        s.id for s in sm.configuration
    }

    original_tracer = Tracer()
    restored_tracer = Tracer()
    sm.add_listener(original_tracer)
    restored.add_listener(restored_tracer)

    for event in ["coronation"]:
        sm.send(event)
        restored.send(event)
        assert {s.id for s in restored.configuration} == {s.id for s in sm.configuration}

    assert restored_tracer.trace == original_tracer.trace


def test_roundtrip_shallow_history():
    sm = GollumPersonality()
    sm.dark_side()
    sm.leave()
    snapshot = sm.snapshot()
    assert snapshot["history"] == {"h": ["gollum"]}

    restored = GollumPersonality.restore(snapshot)
    assert {s.id for s in restored.configuration} == {"outside"}
    restored.return_via_history()
    assert "gollum" in restored.configuration_values

    # Replay the same post-snapshot sequence on a fresh machine driven to the
    # same stable point with its tracer attached at the same moment.
    original = GollumPersonality()
    original.dark_side()
    original.leave()
    original.return_via_history()
    original_tracer = Tracer()
    original.add_listener(original_tracer)
    restored_tracer = Tracer()
    restored.add_listener(restored_tracer)

    for event in ["light_side"]:
        original.send(event)
        restored.send(event)
        assert {s.id for s in restored.configuration} == {
            s.id for s in original.configuration
        }
    assert restored_tracer.trace == original_tracer.trace


def test_roundtrip_deep_history():
    sm = DeepMemoryOfMoria()
    sm.explore()
    sm.escape()
    snapshot = sm.snapshot()
    assert snapshot["history"]["h"] == ["halls", "chamber"]

    restored = DeepMemoryOfMoria.restore(snapshot)
    restored.return_deep()
    assert {"moria", "halls", "chamber"} <= {s.id for s in restored.configuration}

    original = DeepMemoryOfMoria()
    original.explore()
    original.escape()
    original.return_deep()
    assert {s.id for s in restored.configuration} == {
        s.id for s in original.configuration
    }


def test_roundtrip_terminated_parallel_machine():
    sm = WarOfTheRing()
    sm.journey()
    sm.destroy_ring()
    sm.coronation()
    sm.ride_to_gondor()
    assert sm.is_terminated
    snapshot = sm.snapshot()
    assert snapshot["terminated"] is True

    restored = WarOfTheRing.restore(snapshot)
    assert restored.is_terminated
    assert {s.id for s in restored.configuration} == {s.id for s in sm.configuration}


def test_roundtrip_terminated_machine():
    sm = OrderMachine(OrderModel())
    sm.submit()
    sm.approve()
    assert sm.is_terminated
    snapshot = sm.snapshot(model_fields=["order_id"])
    assert snapshot["terminated"] is True
    assert snapshot["configuration"] == ["done"]

    restored = OrderMachine.restore(snapshot, OrderModel())
    assert restored.is_terminated
    assert restored.model.order_id == "ORD-1"


def test_restore_from_json_text_and_module_level_function():
    sm = OrderMachine()
    sm.submit()
    text = json.dumps(sm.snapshot())
    restored = restore_snapshot(OrderMachine, text)
    assert "review" in restored.configuration_values


# ---------------------------------------------------------------------------
# Fingerprint detects real definition drift
# ---------------------------------------------------------------------------


def test_fingerprint_detects_definition_change():
    sm = OrderMachine()
    sm.submit()
    snapshot = sm.snapshot()

    class OrderMachineV2(StateMachine):
        draft = State(initial=True)
        review = State()
        rejected = State()
        done = State(final=True)

        submit = draft.to(review)
        reject = review.to(rejected)
        resubmit = rejected.to(review)
        approve = review.to(done)

    # Same class name, different topology -> different fingerprint -> rejected.
    OrderMachineV2.__name__ = "OrderMachine"
    OrderMachineV2.name = "OrderMachine"
    snapshot["machine"] = "OrderMachine"
    with pytest.raises(SnapshotDefinitionMismatch, match="fingerprint"):
        OrderMachineV2.restore(snapshot)


# ---------------------------------------------------------------------------
# Both engines (sync + async) via sm_runner
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestSnapshotOnBothEngines:
    @pytest.mark.parametrize(
        "machine_cls, setup_events, expected_after_restore",
        [
            (NestedCompound, ["go_b"], {"outer", "inner", "b"}),
            (WarOfTheRing, ["journey"], None),
            (GollumPersonality, ["dark_side", "leave"], {"outside"}),
            (DeepMemoryOfMoria, ["explore", "escape"], {"outside"}),
        ],
    )
    async def test_roundtrip_preserves_configuration(
        self, sm_runner, machine_cls, setup_events, expected_after_restore
    ):
        sm = await sm_runner.start(machine_cls)
        for event in setup_events:
            await sm_runner.send(sm, event)

        snapshot = sm.snapshot()
        restored = machine_cls.restore(snapshot)

        if expected_after_restore is not None:
            assert {s.id for s in restored.configuration} == expected_after_restore
        else:
            assert {s.id for s in restored.configuration} == {
                s.id for s in sm.configuration
            }

    async def test_event_sequence_equivalence_after_restore(self, sm_runner):
        sm = await sm_runner.start(NestedCompound)
        await sm_runner.send(sm, "go_b")
        snapshot = sm.snapshot()

        original = await sm_runner.start(NestedCompound)
        await sm_runner.send(original, "go_b")

        restored = NestedCompound.restore(snapshot)
        original_tracer = Tracer()
        restored_tracer = Tracer()
        original.add_listener(original_tracer)
        restored.add_listener(restored_tracer)

        for event in ["leave", "return_via_history", "go_b", "leave", "finish"]:
            await sm_runner.send(original, event)
            await sm_runner.send(restored, event)
            assert {s.id for s in restored.configuration} == {
                s.id for s in original.configuration
            }

        assert restored_tracer.trace == original_tracer.trace
        assert restored.is_terminated

    async def test_restored_async_machine_continues_without_initial_entry(self, sm_runner):
        sm = await sm_runner.start(OrderMachine)
        await sm_runner.send(sm, "submit")
        snapshot = sm.snapshot()

        tracer = Tracer()
        restored = OrderMachine.restore(snapshot, listeners=[tracer])
        # No activation/entry work on restore, regardless of engine.
        assert tracer.trace == []
        assert "review" in restored.configuration_values

        await sm_runner.send(restored, "approve")
        assert "done" in restored.configuration_values
        assert restored.is_terminated


# ---------------------------------------------------------------------------
# Defensive validation branches (tampered documents)
# ---------------------------------------------------------------------------


def test_restore_rejects_non_dict_document():
    with pytest.raises(SnapshotDefinitionMismatch, match="not valid JSON"):
        OrderMachine.restore([1, 2, 3])
    with pytest.raises(SnapshotDefinitionMismatch, match="JSON object"):
        OrderMachine.restore("[1, 2, 3]")


def test_restore_rejects_invalid_json_text():
    with pytest.raises(SnapshotDefinitionMismatch, match="not valid JSON"):
        OrderMachine.restore("{not json")


def test_restore_rejects_unknown_and_missing_keys():
    snapshot = _make_order_snapshot()
    snapshot["extra"] = 1
    with pytest.raises(SnapshotDefinitionMismatch, match="unknown keys"):
        OrderMachine.restore(snapshot)

    snapshot = _make_order_snapshot()
    del snapshot["history"]
    with pytest.raises(SnapshotDefinitionMismatch, match="missing keys"):
        OrderMachine.restore(snapshot)


def test_restore_rejects_bad_field_types():
    with pytest.raises(SnapshotDefinitionMismatch, match="state_field"):
        OrderMachine.restore(_make_order_snapshot(state_field=123))

    with pytest.raises(SnapshotDefinitionMismatch, match="configuration"):
        OrderMachine.restore(_make_order_snapshot(configuration=[]))

    with pytest.raises(SnapshotDefinitionMismatch, match="configuration"):
        OrderMachine.restore(_make_order_snapshot(configuration="review"))

    with pytest.raises(SnapshotDefinitionMismatch, match="history"):
        OrderMachine.restore(_make_order_snapshot(history={"h": "review"}))

    with pytest.raises(SnapshotDefinitionMismatch, match="'model'"):
        OrderMachine.restore(_make_order_snapshot(model=[1, 2]))

    with pytest.raises(SnapshotDefinitionMismatch, match="terminated"):
        OrderMachine.restore(_make_order_snapshot(terminated="yes"))


def test_restore_rejects_non_string_configuration_entry():
    snapshot = _make_order_snapshot(configuration=[42])
    with pytest.raises(SnapshotDefinitionMismatch, match="non-string entry"):
        OrderMachine.restore(snapshot)


def test_restore_rejects_missing_ancestor_in_configuration():
    sm = NestedCompound()
    snapshot = sm.snapshot()
    # 'b' without 'inner' violates ancestor closure.
    snapshot["configuration"] = ["outer", "b"]
    with pytest.raises(SnapshotDefinitionMismatch, match="missing its ancestor"):
        NestedCompound.restore(snapshot)


def test_restore_rejects_compound_without_children():
    sm = NestedCompound()
    snapshot = sm.snapshot()
    snapshot["configuration"] = ["outer"]
    with pytest.raises(SnapshotDefinitionMismatch, match="without any of its children"):
        NestedCompound.restore(snapshot)


def test_restore_rejects_terminated_flag_inconsistency():
    sm = OrderMachine()
    snapshot = sm.snapshot()
    snapshot["terminated"] = True
    with pytest.raises(SnapshotDefinitionMismatch, match="marked terminated"):
        OrderMachine.restore(snapshot)


def test_restore_rejects_managed_state_field_in_model_data():
    sm = OrderMachine(OrderModel())
    with pytest.raises(SnapshotError, match="managed by the state machine"):
        sm.snapshot(model_fields=["state"])


def test_snapshot_export_rejects_nested_non_json_values():
    sm = OrderMachine(OrderModel())
    sm.model.payload = {"ok": [1, {"bad": object()}]}
    with pytest.raises(SnapshotError, match="JSON-serializable"):
        sm.snapshot(model_fields=["payload"])

    sm.model.ints = {1: "x"}
    with pytest.raises(SnapshotError, match="non-string key"):
        sm.snapshot(model_fields=["ints"])


def test_restore_rejects_unknown_history_id_and_history_in_history():
    sm = GollumPersonality()
    sm.dark_side()
    sm.leave()

    snapshot = sm.snapshot()
    snapshot["history"] = {"nope": ["gollum"]}
    with pytest.raises(SnapshotDefinitionMismatch, match="not a declared history"):
        GollumPersonality.restore(snapshot)

    snapshot = sm.snapshot()
    snapshot["history"] = {"h": ["h"]}
    with pytest.raises(SnapshotDefinitionMismatch, match="cannot be stored in history"):
        GollumPersonality.restore(snapshot)

    snapshot = sm.snapshot()
    snapshot["history"] = {"h": [42]}
    with pytest.raises(SnapshotDefinitionMismatch, match="contains a non-string entry"):
        GollumPersonality.restore(snapshot)


def test_restore_rejects_shallow_history_with_grandchild():
    sm = GollumPersonality()
    sm.dark_side()
    sm.leave()
    snapshot = sm.snapshot()
    # 'gollum' is valid for shallow history; tamper list type entry via deep machine.
    from tests.machines.history.deep_memory_of_moria import DeepMemoryOfMoria as Deep

    deep_sm = Deep()
    deep_sm.explore()
    deep_sm.escape()
    deep_snapshot = deep_sm.snapshot()
    # Deep history of moria cannot remember 'outside'.
    deep_snapshot["history"] = {"h": ["outside"]}
    with pytest.raises(SnapshotDefinitionMismatch, match="deep-history child"):
        Deep.restore(deep_snapshot)

    # Shallow history of personality cannot store its own history pseudo-state
    # sibling; use a descendant of a nested child to exercise the shallow branch.
    snapshot["history"] = {"h": ["outside"]}
    with pytest.raises(SnapshotDefinitionMismatch, match="shallow-history child"):
        GollumPersonality.restore(snapshot)


def test_snapshot_rejected_when_internal_queue_nonempty():
    class InternalQueueMachine(StateMachine):
        s1 = State(initial=True)
        s2 = State()
        s3 = State(final=True)
        go = s1.to(s2)
        finish = s2.to(s3)

        def on_go(self):
            # Raised while the macrostep runs: captured via raise_ on the
            # internal queue of a StateChart would be more idiomatic, but for
            # the sync StateMachine we directly assert the quiescence guard.
            with pytest.raises(SnapshotNotStable, match="callback or transition"):
                self.snapshot()

    sm = InternalQueueMachine()
    sm.go()

    # Directly exercise the internal-queue branch using the raw engine API.
    from statemachine.event import BoundEvent

    BoundEvent(id="finish", internal=True, _sm=sm).put()
    assert not sm._engine.internal_queue.is_empty()
    with pytest.raises(SnapshotNotStable, match="internal event queue"):
        sm.snapshot()


def test_snapshot_uses_custom_state_field():
    class StatusModel:
        def __init__(self):
            self.current = None

    class StatusMachine(StateMachine):
        idle = State(initial=True)
        busy = State(final=True)
        run = idle.to(busy)

    model = StatusModel()
    sm = StatusMachine(model, state_field="current")
    snapshot = sm.snapshot()
    assert snapshot["state_field"] == "current"
    restored = StatusMachine.restore(snapshot, StatusModel())
    assert restored.state_field == "current"
    assert restored.current_state_value == "idle"


def test_fingerprint_changes_with_callback_and_is_stable_across_instances():
    from statemachine.snapshot import definition_fingerprint

    class A(StateMachine):
        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2)

    class B(StateMachine):
        s1 = State(initial=True)
        s2 = State(final=True)
        go = s1.to(s2, on="do_go")

        def do_go(self):
            return None

    A.__name__ = "SameName"
    B.__name__ = "SameName"
    assert definition_fingerprint(A) == definition_fingerprint(A)
    assert definition_fingerprint(A) != definition_fingerprint(B)


def test_restore_atomic_parallel_regions_marked_terminated():
    class AtomicParallelFinal(StateChart):
        class board(State.Parallel):
            north = State(final=True)
            south = State(final=True)

        peace = State(final=True)
        truce = board.to(peace)

    snapshot = AtomicParallelFinal().snapshot()
    snapshot["terminated"] = True
    restored = AtomicParallelFinal.restore(snapshot)
    assert restored.is_terminated


def test_restore_parallel_with_atomic_final_regions():
    class AtomicParallelFinal(StateChart):
        class board(State.Parallel):
            north = State(final=True)
            south = State(final=True)

        peace = State(final=True)
        truce = board.to(peace)

    sm = AtomicParallelFinal()
    snapshot = sm.snapshot()
    assert snapshot["terminated"] is False

    restored = AtomicParallelFinal.restore(snapshot)
    assert not restored.is_terminated
    assert {"board", "north", "south"} <= {s.id for s in restored.configuration}

    # The restored machine can still take the declared transition out.
    restored.truce()
    assert restored.is_terminated


def test_tampered_region_rejected_before_any_callback():
    class ExplodingTracer:
        def setup(self, machine):
            raise AssertionError("setup() must not run for an invalid snapshot")

        def on_enter_state(self, state):
            raise AssertionError("entry callback must not run for an invalid snapshot")

    sm = WarOfTheRing()
    snapshot = sm.snapshot()
    snapshot["configuration"] = [
        state_id for state_id in snapshot["configuration"] if state_id != "shire"
    ]
    with pytest.raises(SnapshotDefinitionMismatch, match="parallel state"):
        WarOfTheRing.restore(snapshot, listeners=[ExplodingTracer()])
