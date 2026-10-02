"""Tests for explicit snapshot export and restoration (:mod:`statemachine.persistence`).

The matrix required by the feature contract:

* Snapshots on hierarchical, parallel and shallow/deep history configurations.
* Original and restored instances receive the same event sequence; configurations
  and callback traces must match.
* Tampering with one region or the definition fingerprint fails restoration
  *before any user callback runs*.
* Non-quiescent checkpoints (queued events, active/pending invokes, mid-callback)
  are rejected with an explanatory error.
"""

import threading

import pytest

from statemachine import State
from statemachine import StateChart
from statemachine.event import BoundEvent
from statemachine.exceptions import InvalidSnapshot
from statemachine.exceptions import SnapshotNotQuiescentError
from statemachine.invoke import InvokeContext
from statemachine.invoke import IInvoke
from statemachine.persistence import SNAPSHOT_FORMAT_VERSION
from statemachine.persistence import Snapshot
from statemachine.persistence import definition_fingerprint
from statemachine.persistence import restore_snapshot
from statemachine.persistence import assert_quiescent
from statemachine.persistence import take_snapshot
from tests.machines.history.deep_memory_of_moria import DeepMemoryOfMoria
from tests.machines.history.gollum_personality import GollumPersonality
from tests.machines.history.shallow_moria import ShallowMoria


# -- Test fixtures -----------------------------------------------------------


class CallbackTracer:
    """Listener recording every enter/exit/transition callback as tuples."""

    def __init__(self):
        self.trace: list[tuple] = []

    def before_transition(self, event, state):
        self.trace.append(("before", str(event), state.id))

    def after_transition(self, event, state):
        self.trace.append(("after", str(event), state.id))

    def on_enter_state(self, state):
        self.trace.append(("enter", state.id))

    def on_exit_state(self, state):
        self.trace.append(("exit", state.id))


class DomainModel:
    """Plain model with domain fields; entry callbacks mutate one of them."""

    def __init__(self, title: str = "untitled", score: int = 0):
        self.state = None
        self.title = title
        self.score = score


class HierarchicalOrder(StateChart):
    """Compound nested two levels deep with an outside final branch."""

    class processing(State.Compound):
        class review(State.Compound):
            pending = State(initial=True)
            approved = State()
            rejected = State()

            approve = pending.to(approved)
            reject = pending.to(rejected)
            rework = rejected.to(pending)

        waiting = State(initial=True)
        in_review = waiting.to(review)
        back = review.to(waiting)

    completed = State(final=True)

    start = completed.from_(processing)


class ParallelWar(StateChart):
    """Three parallel regions, each compound with an atomic leaf and a final."""

    class war(State.Parallel):
        class region_one(State.Compound):
            one_active = State(initial=True)
            one_done = State(final=True)
            finish_one = one_active.to(one_done)

        class region_two(State.Compound):
            two_active = State(initial=True)
            two_done = State(final=True)
            finish_two = two_active.to(two_done)

        class region_three(State.Compound):
            three_active = State(initial=True)
            three_done = State(final=True)
            finish_three = three_active.to(three_done)

    idle = State(initial=True)
    begin = idle.to(war)
    leave = war.to(idle)


async def _new(sm_runner, sm_class, tracer, model=None):
    """Create + activate a machine tracked by *tracer* on the runner engine."""
    return await sm_runner.start(sm_class, model=model, listeners=[tracer])


async def _restore(sm_runner, sm_class, snapshot, tracer, model=None):
    """Restore a snapshot with *tracer* attached (adds the async driver listener)."""
    listeners = [tracer]
    if sm_runner.is_async:
        from tests.conftest import _AsyncListener

        listeners.append(_AsyncListener())
    return restore_snapshot(sm_class, snapshot, model=model, listeners=listeners)


async def _send(sm_runner, sm, event):
    return await sm_runner.send(sm, event)


# -- Snapshot payload --------------------------------------------------------


class TestSnapshotPayload:
    def test_payload_contains_required_fields(self):
        sm = HierarchicalOrder()
        snapshot = sm.take_snapshot()
        data = snapshot.to_dict()
        assert data["version"] == SNAPSHOT_FORMAT_VERSION
        assert data["machine_id"] == HierarchicalOrder.id
        assert len(data["fingerprint"]) == 64
        assert data["configuration"] == ["waiting"]
        assert data["history"] == {}
        assert data["model_data"] == {}
        assert data["running"] is True
        assert data["state_field"] == "state"

    def test_json_roundtrip(self):
        sm = HierarchicalOrder()
        sm.in_review()
        snapshot = sm.take_snapshot(include_model_data=[])
        restored = Snapshot.from_json(snapshot.to_json())
        assert restored.configuration == ["pending"]
        assert restored == snapshot

    def test_model_data_whitelist(self):
        model = DomainModel(title="boss", score=42)
        sm = HierarchicalOrder(model=model)
        snapshot = sm.take_snapshot(include_model_data=["title", "score"])
        assert snapshot.model_data == {"title": "boss", "score": 42}

        restored_model = DomainModel()
        restored = HierarchicalOrder.restore_snapshot(snapshot, model=restored_model)
        assert restored.model.title == "boss"
        assert restored.model.score == 42

    def test_non_json_model_data_rejected(self):
        model = DomainModel()
        model.lock = threading.Lock()
        sm = HierarchicalOrder(model=model)
        with pytest.raises(InvalidSnapshot, match="JSON-serializable"):
            sm.take_snapshot(include_model_data=["lock"])

    def test_state_field_cannot_be_exported(self):
        sm = HierarchicalOrder()
        with pytest.raises(InvalidSnapshot, match="managed state field"):
            sm.take_snapshot(include_model_data=["state"])

    def test_missing_model_attribute_rejected(self):
        sm = HierarchicalOrder()
        with pytest.raises(InvalidSnapshot, match="no attribute named 'nope'"):
            sm.take_snapshot(include_model_data=["nope"])

    def test_invalid_include_argument(self):
        sm = HierarchicalOrder()
        with pytest.raises(InvalidSnapshot, match="list of attribute name strings"):
            sm.take_snapshot(include_model_data="title")  # type: ignore[arg-type]

    def test_terminated_machine_snapshot_is_not_running(self):
        sm = HierarchicalOrder()
        sm.start()
        snapshot = sm.take_snapshot()
        assert snapshot.configuration == ["completed"]
        assert snapshot.running is False
        restored = HierarchicalOrder.restore_snapshot(snapshot)
        assert restored.is_terminated


# -- Hierarchical configuration ----------------------------------------------


class TestHierarchicalSnapshot:
    async def test_snapshot_and_equal_event_sequence(self, sm_runner):
        tracer_a, tracer_b = CallbackTracer(), CallbackTracer()
        original = await _new(
            sm_runner, HierarchicalOrder, tracer_a, model=DomainModel(title="a")
        )

        await _send(sm_runner, original, "in_review")
        await _send(sm_runner, original, "reject")
        snapshot = take_snapshot(original)

        assert snapshot.configuration == ["rejected"]

        restored = await _restore(
            sm_runner,
            HierarchicalOrder,
            snapshot,
            tracer_b,
            model=DomainModel(title="b"),
        )

        # Restoration itself produced no entry/exit/transition trace.
        assert tracer_b.trace == []
        assert set(restored.configuration_values) == set(original.configuration_values)
        # The ancestor closure is also present.
        assert {"rejected", "review", "processing"} <= set(restored.configuration_values)

        baseline = len(tracer_a.trace)
        events = ["rework", "approve", "back", "in_review", "approve"]
        for event in events:
            await _send(sm_runner, original, event)
            await _send(sm_runner, restored, event)
            assert set(restored.configuration_values) == set(original.configuration_values)

        # Callback traces after the checkpoint must be identical.
        assert tracer_b.trace == tracer_a.trace[baseline:]

    async def test_restored_machine_reaches_final_identically(self, sm_runner):
        tracer_a, tracer_b = CallbackTracer(), CallbackTracer()

        original = await _new(sm_runner, HierarchicalOrder, tracer_a)
        await _send(sm_runner, original, "in_review")
        restored = await _restore(sm_runner, HierarchicalOrder, original.take_snapshot(), tracer_b)
        baseline = len(tracer_a.trace)

        await _send(sm_runner, original, "approve")
        await _send(sm_runner, restored, "approve")
        await _send(sm_runner, original, "back")
        await _send(sm_runner, restored, "back")
        await _send(sm_runner, original, "start")
        await _send(sm_runner, restored, "start")

        assert restored.is_terminated == original.is_terminated
        assert set(restored.configuration_values) == set(original.configuration_values)
        assert tracer_b.trace == tracer_a.trace[baseline:]


# -- Parallel regions --------------------------------------------------------


class TestParallelSnapshot:
    async def test_parallel_snapshot_and_event_sequence(self, sm_runner):
        tracer_a, tracer_b = CallbackTracer(), CallbackTracer()

        original = await _new(sm_runner, ParallelWar, tracer_a)
        await _send(sm_runner, original, "begin")
        await _send(sm_runner, original, "finish_one")
        await _send(sm_runner, original, "finish_three")
        snapshot = original.take_snapshot()

        assert set(snapshot.configuration) == {"one_done", "two_active", "three_done"}

        restored = await _restore(sm_runner, ParallelWar, snapshot, tracer_b)
        assert tracer_b.trace == []
        assert set(restored.configuration_values) == set(original.configuration_values)
        baseline = len(tracer_a.trace)

        await _send(sm_runner, original, "finish_two")
        await _send(sm_runner, restored, "finish_two")
        assert restored.is_terminated == original.is_terminated
        assert restored.is_terminated
        assert tracer_b.trace == tracer_a.trace[baseline:]

    @pytest.mark.parametrize("tampered", ["two_active", "three_active"])
    def test_tampered_region_rejected_before_callbacks(self, tampered):
        sm = ParallelWar()
        sm.begin()
        sm.finish_one()

        data = sm.take_snapshot().to_dict()
        data["configuration"].remove(tampered)

        tracer = CallbackTracer()
        with pytest.raises(InvalidSnapshot, match="parallel configuration"):
            restore_snapshot(ParallelWar, data, listeners=[tracer])
        assert tracer.trace == []

    def test_extra_leaf_in_two_regions_rejected(self):
        sm = ParallelWar()
        sm.begin()
        data = sm.take_snapshot().to_dict()
        # Two active children of the same compound region violates parent/child rules.
        data["configuration"].append("two_done")
        with pytest.raises(InvalidSnapshot, match="exactly one"):
            restore_snapshot(ParallelWar, data)

    def test_unknown_state_id_rejected(self):
        sm = ParallelWar()
        sm.begin()
        data = sm.take_snapshot().to_dict()
        data["configuration"] = ["one_active", "narnia", "three_active"]
        with pytest.raises(InvalidSnapshot, match="Unknown state id"):
            restore_snapshot(ParallelWar, data)

    def test_compound_id_as_leaf_rejected(self):
        sm = ParallelWar()
        sm.begin()
        data = sm.take_snapshot().to_dict()
        data["configuration"] = ["one_active", "region_two", "three_active"]
        with pytest.raises(InvalidSnapshot, match="only leaf states"):
            restore_snapshot(ParallelWar, data)

    def test_duplicate_leaf_rejected(self):
        sm = ParallelWar()
        sm.begin()
        data = sm.take_snapshot().to_dict()
        data["configuration"].append("one_active")
        with pytest.raises(InvalidSnapshot, match="duplicate"):
            restore_snapshot(ParallelWar, data)


# -- Shallow / deep history --------------------------------------------------


class TestHistorySnapshots:
    async def test_shallow_history_snapshot_replay(self, sm_runner):
        """Snapshot outside a compound still carries its shallow history record."""
        tracer_a, tracer_b = CallbackTracer(), CallbackTracer()
        original = await _new(sm_runner, ShallowMoria, tracer_a)
        await _send(sm_runner, original, "explore")
        await _send(sm_runner, original, "escape")
        snapshot = original.take_snapshot()

        assert snapshot.configuration == ["outside"]
        assert snapshot.history == {"h": ["halls"]}

        restored = await _restore(sm_runner, ShallowMoria, snapshot, tracer_b)
        baseline = len(tracer_a.trace)
        await _send(sm_runner, restored, "return_shallow")
        await _send(sm_runner, original, "return_shallow")
        assert set(restored.configuration_values) == set(original.configuration_values)
        # Shallow history re-enters halls at its INITIAL child (entrance),
        # not the remembered grandchild (chamber).
        assert {"entrance", "halls", "moria"} <= set(restored.configuration_values)
        assert tracer_b.trace == tracer_a.trace[baseline:]

    async def test_deep_history_snapshot_replay(self, sm_runner):
        tracer_a, tracer_b = CallbackTracer(), CallbackTracer()
        original = await _new(sm_runner, DeepMemoryOfMoria, tracer_a)
        await _send(sm_runner, original, "explore")
        await _send(sm_runner, original, "escape")
        snapshot = original.take_snapshot()

        assert snapshot.history == {"h": ["halls", "chamber"]}

        restored = await _restore(sm_runner, DeepMemoryOfMoria, snapshot, tracer_b)
        baseline = len(tracer_a.trace)
        await _send(sm_runner, restored, "return_deep")
        await _send(sm_runner, original, "return_deep")
        assert {"chamber", "halls", "moria"} <= set(restored.configuration_values)
        assert set(restored.configuration_values) == set(original.configuration_values)
        assert tracer_b.trace == tracer_a.trace[baseline:]

    def test_history_updates_after_restore(self):
        tracer_a, tracer_b = CallbackTracer(), CallbackTracer()
        original = GollumPersonality(listeners=[tracer_a])
        original.dark_side()
        original.leave()
        restored = restore_snapshot(
            GollumPersonality, original.take_snapshot(), listeners=[tracer_b]
        )
        baseline = len(tracer_a.trace)

        for event in ("return_via_history", "light_side", "leave", "return_via_history"):
            original.send(event)
            restored.send(event)
            assert set(restored.configuration_values) == set(original.configuration_values)
        assert restored.history_values["h"][0].id == "smeagol"
        assert tracer_b.trace == tracer_a.trace[baseline:]

    def test_shallow_history_rejects_grandchild(self):
        sm = ShallowMoria()
        sm.explore()
        sm.escape()
        data = sm.take_snapshot().to_dict()
        data["history"]["h"] = ["chamber"]
        with pytest.raises(InvalidSnapshot, match="direct children"):
            restore_snapshot(ShallowMoria, data)

    def test_deep_history_accepts_grandchild(self):
        sm = DeepMemoryOfMoria()
        sm.explore()
        sm.escape()
        snapshot = sm.take_snapshot()
        assert snapshot.history["h"] == ["halls", "chamber"]
        restored = restore_snapshot(DeepMemoryOfMoria, snapshot)
        restored.return_deep()
        assert "chamber" in restored.configuration_values

    def test_unknown_history_id_rejected(self):
        sm = GollumPersonality()
        sm.dark_side()
        sm.leave()
        data = sm.take_snapshot().to_dict()
        data["history"]["nope"] = ["gollum"]
        with pytest.raises(InvalidSnapshot, match="Unknown history"):
            restore_snapshot(GollumPersonality, data)

    def test_history_target_outside_owner_rejected(self):
        sm = GollumPersonality()
        sm.dark_side()
        sm.leave()
        data = sm.take_snapshot().to_dict()
        data["history"]["h"] = ["outside"]
        with pytest.raises(InvalidSnapshot, match="direct children"):
            restore_snapshot(GollumPersonality, data)


# -- Definition fingerprint --------------------------------------------------


class TestDefinitionFingerprint:
    def test_fingerprint_stable_across_instances(self):
        assert definition_fingerprint(HierarchicalOrder) == definition_fingerprint(
            HierarchicalOrder
        )
        sm = HierarchicalOrder()
        assert sm.take_snapshot().fingerprint == definition_fingerprint(HierarchicalOrder)

    def test_different_machines_have_different_fingerprints(self):
        assert definition_fingerprint(ParallelWar) != definition_fingerprint(HierarchicalOrder)

    def test_tampered_fingerprint_rejected_before_callbacks(self):
        sm = HierarchicalOrder()
        sm.in_review()
        data = sm.take_snapshot().to_dict()
        data["fingerprint"] = "0" * 64

        tracer = CallbackTracer()
        with pytest.raises(InvalidSnapshot, match="fingerprint does not match"):
            restore_snapshot(HierarchicalOrder, data, listeners=[tracer])
        assert tracer.trace == []

    def test_changed_definition_rejects_old_snapshot(self):
        """A snapshot whose fingerprint differs from the current class is rejected."""

        class_src_v1 = """
from statemachine import State, StateChart
class Versioned(StateChart):
    a = State(initial=True)
    b = State(final=True)
    go = a.to(b)
"""
        class_src_v2 = """
from statemachine import State, StateChart
class Versioned(StateChart):
    a = State(initial=True)
    b = State()
    c = State(final=True)
    go = a.to(b)
    continue_ = b.to(c)
"""
        ns_v1: dict = {}
        exec(class_src_v1, ns_v1)  # noqa: S102 - test fixture
        ns_v2: dict = {}
        exec(class_src_v2, ns_v2)  # noqa: S102 - test fixture
        VersionedV1 = ns_v1["Versioned"]
        VersionedV2 = ns_v2["Versioned"]

        snapshot = VersionedV1().take_snapshot().to_dict()
        assert snapshot["machine_id"] == VersionedV2.id
        with pytest.raises(InvalidSnapshot, match="fingerprint does not match"):
            restore_snapshot(VersionedV2, snapshot)

    def test_wrong_machine_class_rejected(self):
        snapshot = ParallelWar().take_snapshot()
        with pytest.raises(InvalidSnapshot, match="belongs to machine"):
            restore_snapshot(HierarchicalOrder, snapshot)

    def test_unknown_version_rejected(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["version"] = 999
        with pytest.raises(InvalidSnapshot, match="Unsupported snapshot version"):
            Snapshot.from_dict(data)

    def test_missing_field_rejected(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        del data["fingerprint"]
        with pytest.raises(InvalidSnapshot, match="missing required"):
            Snapshot.from_dict(data)

    def test_invalid_json_rejected(self):
        with pytest.raises(InvalidSnapshot, match="not valid JSON"):
            Snapshot.from_json("{not json")

    def test_bad_payload_type_rejected(self):
        with pytest.raises(InvalidSnapshot, match="mapping"):
            Snapshot.from_dict(["not", "a", "dict"])
        with pytest.raises(InvalidSnapshot, match="type"):
            restore_snapshot(HierarchicalOrder, 42)


# -- Quiescence checks -------------------------------------------------------


def _enqueue(sm, event: str, *, internal: bool = False):
    """Put an event directly into a queue without running the processing loop."""
    bound = BoundEvent(event, _sm=sm, internal=internal)
    trigger_data = bound.build_trigger(machine=sm)
    sm._put_nonblocking(trigger_data, internal=internal)


class _BlockingWorker(IInvoke):
    def __init__(self):
        self.started = threading.Event()
        self.finish = threading.Event()

    def run(self, ctx: InvokeContext):
        self.started.set()
        self.finish.wait(timeout=5)

    def on_cancel(self):
        self.finish.set()


class TestQuiescence:
    def test_external_queue_must_be_empty(self):
        sm = HierarchicalOrder()
        _enqueue(sm, "in_review")
        with pytest.raises(SnapshotNotQuiescentError, match="external event queue"):
            sm.take_snapshot()

    def test_internal_queue_must_be_empty(self):
        sm = HierarchicalOrder()
        _enqueue(sm, "in_review", internal=True)
        with pytest.raises(SnapshotNotQuiescentError, match="internal event queue"):
            sm.take_snapshot()

    def test_snapshot_inside_callback_rejected(self):
        captured: list[SnapshotNotQuiescentError] = []

        class Trap(StateChart):
            a = State(initial=True)
            b = State(final=True)
            go = a.to(b)

            def on_exit_a(self):
                try:
                    self.take_snapshot()
                except SnapshotNotQuiescentError as err:
                    captured.append(err)

        Trap().go()
        assert captured and "being executed" in str(captured[0])

    def test_snapshot_before_activation_rejected(self):
        class AsyncSM(StateChart):
            a = State(initial=True)
            b = State(final=True)
            go = a.to(b)

            async def on_enter_state(self, **kwargs):
                pass

        sm = AsyncSM()
        with pytest.raises(SnapshotNotQuiescentError, match="initial state is activated"):
            sm.take_snapshot()

    def test_active_invoke_rejected_and_then_allowed(self):
        worker = _BlockingWorker()

        class InvokingSM(StateChart):
            loading = State(initial=True, invoke=worker)
            ready = State(final=True)
            done_invoke_loading = loading.to(ready)

        sm = InvokingSM()
        assert worker.started.wait(timeout=3)
        with pytest.raises(SnapshotNotQuiescentError, match="active invocation"):
            sm.take_snapshot()

        worker.finish.set()
        # done.invoke arrives on the external queue; draining it reaches 'ready'.
        for _ in range(20):
            if "ready" in sm.configuration_values:
                break
            threading.Event().wait(0.05)
            sm._processing_loop()
        assert "ready" in sm.configuration_values
        snapshot = sm.take_snapshot()
        assert snapshot.configuration == ["ready"]
        assert snapshot.running is False

    def test_assert_quiescent_helper(self):
        sm = HierarchicalOrder()
        assert assert_quiescent(sm) is None
        _enqueue(sm, "in_review")
        with pytest.raises(SnapshotNotQuiescentError, match="external event queue"):
            assert_quiescent(sm)

# -- No replay on restore ----------------------------------------------------


class TestNoReplay:
    def test_entry_and_transition_actions_not_replayed(self):
        calls: list[str] = []

        class ActionSM(StateChart):
            a = State(initial=True, enter=lambda: calls.append("enter_a"))
            b = State(enter=lambda: calls.append("enter_b"))
            c = State(final=True, enter=lambda: calls.append("enter_c"))

            go = a.to(b, on=lambda: calls.append("on_go"))
            finish = b.to(c, on=lambda: calls.append("on_finish"))

        sm = ActionSM()
        calls.clear()  # discard initial entry trace
        sm.go()
        assert calls == ["on_go", "enter_b"]
        calls.clear()

        restored = ActionSM.restore_snapshot(sm.take_snapshot())
        assert calls == []  # nothing replayed on restore
        restored.finish()
        assert calls == ["on_finish", "enter_c"]

    def test_missing_state_is_not_guessed_as_initial(self):
        """A corrupt compound config without its child cannot fall back to initial."""

        class Compound(StateChart):
            class outer(State.Compound):
                first = State(initial=True)
                second = State(final=True)
                move = first.to(second)

            home = State(initial=True)
            enter = home.to(outer)

        sm = Compound()
        sm.enter()
        data = sm.take_snapshot().to_dict()
        assert data["configuration"] == ["first"]
        data["configuration"] = ["outer"]  # the compound, but no leaf child
        with pytest.raises(InvalidSnapshot, match="only leaf states"):
            restore_snapshot(Compound, data)

    def test_empty_configuration_rejected(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["configuration"] = []
        with pytest.raises(InvalidSnapshot, match="empty"):
            restore_snapshot(HierarchicalOrder, data)

    def test_restore_from_dict_and_json_forms(self):
        sm = HierarchicalOrder()
        snapshot = sm.take_snapshot()
        from_dict = restore_snapshot(HierarchicalOrder, snapshot.to_dict())
        from_json = restore_snapshot(HierarchicalOrder, snapshot.to_json())
        assert set(from_dict.configuration_values) == set(sm.configuration_values)
        assert set(from_json.configuration_values) == set(sm.configuration_values)

    def test_model_data_inside_payload_cannot_contain_state_field(self):
        sm = HierarchicalOrder()
        data = sm.take_snapshot().to_dict()
        data["model_data"] = {"state": "waiting"}
        with pytest.raises(InvalidSnapshot, match="state field"):
            restore_snapshot(HierarchicalOrder, data)

    def test_non_json_model_data_in_payload_rejected(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["model_data"] = {"nested": {1, 2, 3}}  # sets are not JSON-native
        with pytest.raises(InvalidSnapshot, match="JSON-serializable"):
            Snapshot.from_dict(data)

    def test_non_object_model_data_rejected(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["model_data"] = ["not", "an", "object"]
        with pytest.raises(InvalidSnapshot, match="must be an object/mapping"):
            Snapshot.from_dict(data)

    def test_state_field_preserved(self):
        sm = HierarchicalOrder(state_field="phase")
        snapshot = sm.take_snapshot()
        assert snapshot.state_field == "phase"
        restored = HierarchicalOrder.restore_snapshot(snapshot)
        assert restored.state_field == "phase"
        assert set(restored.configuration_values) == {"waiting", "processing"}


# -- Additional payload and structural edge cases ----------------------------


class TestPayloadValidationEdges:
    def test_configuration_must_be_list_of_strings(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["configuration"] = ["waiting", 5]
        with pytest.raises(InvalidSnapshot, match="list of state id strings"):
            Snapshot.from_dict(data)

    def test_history_must_be_string_mapping(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["history"] = {"h": "waiting"}
        with pytest.raises(InvalidSnapshot, match="lists of state id strings"):
            Snapshot.from_dict(data)

    def test_history_values_must_be_strings(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["history"] = {"h": [1]}
        with pytest.raises(InvalidSnapshot, match="lists of state id strings"):
            Snapshot.from_dict(data)

    def test_running_must_be_bool(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["running"] = "yes"
        with pytest.raises(InvalidSnapshot, match="'running' must be a boolean"):
            Snapshot.from_dict(data)

    def test_state_field_must_be_string(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["state_field"] = 7
        with pytest.raises(InvalidSnapshot, match="'state_field' must be a non-empty string"):
            Snapshot.from_dict(data)

    def test_state_field_must_be_non_empty(self):
        data = HierarchicalOrder().take_snapshot().to_dict()
        data["state_field"] = ""
        with pytest.raises(InvalidSnapshot, match="'state_field' must be a non-empty string"):
            Snapshot.from_dict(data)

    def test_restore_from_bytes(self):
        snapshot = HierarchicalOrder().take_snapshot()
        restored = HierarchicalOrder.restore_snapshot(snapshot.to_json().encode("utf-8"))
        assert set(restored.configuration_values) == {"waiting", "processing"}

    def test_empty_history_record_rejected(self):
        sm = GollumPersonality()
        sm.dark_side()
        sm.leave()
        data = sm.take_snapshot().to_dict()
        data["history"]["h"] = []
        with pytest.raises(InvalidSnapshot, match="History record for 'h' is empty"):
            restore_snapshot(GollumPersonality, data)

    def test_duplicate_history_entry_rejected(self):
        sm = GollumPersonality()
        sm.dark_side()
        sm.leave()
        data = sm.take_snapshot().to_dict()
        data["history"]["h"] = ["gollum", "gollum"]
        with pytest.raises(InvalidSnapshot, match="duplicates"):
            restore_snapshot(GollumPersonality, data)

    def test_history_pointing_to_history_state_rejected(self):
        sm = GollumPersonality()
        sm.dark_side()
        sm.leave()
        data = sm.take_snapshot().to_dict()
        data["history"]["h"] = ["h"]
        with pytest.raises(InvalidSnapshot, match="unknown state"):
            restore_snapshot(GollumPersonality, data)

    def test_history_state_id_in_configuration_rejected(self):
        sm = GollumPersonality()
        sm.dark_side()
        sm.leave()
        data = sm.take_snapshot().to_dict()
        data["configuration"] = ["h"]
        with pytest.raises(InvalidSnapshot, match="active leaf configuration"):
            restore_snapshot(GollumPersonality, data)

    def test_multiple_top_level_subtrees_rejected(self):
        """Parallel siblings are only valid below a parallel parent, never at root."""
        sm = ParallelWar()
        sm.begin()
        sm.finish_one()
        sm.finish_two()
        sm.finish_three()
        # After war reaches completion it is still one subtree; craft a payload
        # claiming leaves from two unrelated top-level states simultaneously.
        data = sm.take_snapshot().to_dict()
        assert "idle" not in data["configuration"]
        data["configuration"] = ["idle", "one_done", "two_done", "three_done"]
        with pytest.raises(InvalidSnapshot, match="exactly one top-level"):
            restore_snapshot(ParallelWar, data)


class TestPendingInvokeQuiescence:
    def test_pending_invoke_blocks_snapshot(self):
        """An invocation marked by entry but not yet spawned forbids checkpointing.

        ``spawn_pending_sync`` runs at the end of each macrostep, so in normal
        sync operation the window is not user-visible; we exercise the guard
        directly to guarantee its behaviour.
        """
        worker = _BlockingWorker()

        class PendingSM(StateChart):
            loading = State(initial=True, invoke=worker)
            ready = State(final=True)
            done_invoke_loading = loading.to(ready)

        sm = PendingSM()
        assert worker.started.wait(timeout=3)
        # Simulate a state entered with invoke marks but no spawn cycle yet.
        sm._engine._invoke_manager._pending.append((sm.loading._state, {}))
        try:
            with pytest.raises(SnapshotNotQuiescentError, match="invocations pending"):
                sm.take_snapshot()
        finally:
            sm._engine._invoke_manager._pending.clear()
            worker.finish.set()


class TestConstructorSignatureMismatch:
    def test_unsupported_custom_constructor_rejected(self):
        class CustomInit(StateChart):
            a = State(initial=True)
            b = State(final=True)
            go = a.to(b)

            def __init__(self, required):  # noqa: D401 - intentionally incompatible
                super().__init__()
                self.required = required

        # A structurally identical class produces a valid, matching fingerprint;
        # the restore failure then comes from the incompatible __init__ itself.
        probe_ns: dict = {}
        exec(  # noqa: S102 - test fixture with a deliberately colliding class name
            """
from statemachine import State, StateChart
class CustomInit(StateChart):
    a = State(initial=True)
    b = State(final=True)
    go = a.to(b)
""",
            probe_ns,
        )
        snapshot = probe_ns["CustomInit"]().take_snapshot()
        with pytest.raises(InvalidSnapshot, match="restore constructor signature"):
            restore_snapshot(CustomInit, snapshot)


class TestSnapshotConcurrency:
    def test_snapshot_while_another_thread_processes(self):
        """A snapshot must fail if another thread holds the processing lock."""
        entered = threading.Event()
        release = threading.Event()

        class SlowSM(StateChart):
            a = State(initial=True)
            b = State(final=True)
            go = a.to(b)

            def on_exit_a(self):
                entered.set()
                release.wait(timeout=5)

        sm = SlowSM()
        errors: list[Exception] = []

        def transition():
            sm.go()

        worker = threading.Thread(target=transition)
        worker.start()
        try:
            assert entered.wait(timeout=3)
            # The other thread owns the non-reentrant processing lock.
            assert sm._engine._processing.locked()
            try:
                sm.take_snapshot()
            except SnapshotNotQuiescentError as err:
                errors.append(err)
        finally:
            release.set()
            worker.join(timeout=3)

        assert errors and "being executed" in str(errors[0])
        # Once quiescent again, the snapshot reflects the final configuration.
        assert sm.take_snapshot().configuration == ["b"]

    def test_snapshot_failed_lock_acquisition_message(self):
        """The failed non-blocking acquire branch (lock held by another owner)."""
        sm = HierarchicalOrder()
        lock = sm._engine._processing
        lock.acquire()
        try:
            with pytest.raises(SnapshotNotQuiescentError, match="being executed"):
                take_snapshot(sm)
            # The shared checker also reports a locked machine as non-quiescent.
            with pytest.raises(SnapshotNotQuiescentError, match="being executed"):
                assert_quiescent(sm)
        finally:
            lock.release()


# -- Custom (enum) state values ----------------------------------------------


class TestCustomStateValues:
    def test_enum_value_roundtrip(self):
        from enum import Enum

        from statemachine.states import States

        class Color(Enum):
            red = 1
            green = 2

        class Light(StateChart):
            _ = States.from_enum(Color, initial=Color.red, final=Color.green)
            cycle = _.red.to(_.green)

        sm = Light()
        snapshot = sm.take_snapshot()
        assert snapshot.configuration == ["red"]

        restored = restore_snapshot(Light, snapshot)
        assert restored.current_state_value == Color.red
        restored.cycle()
        assert restored.current_state_value == Color.green
        assert restored.is_terminated
