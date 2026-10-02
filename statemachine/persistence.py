"""Explicit snapshot export and restoration for :class:`~statemachine.statemachine.StateChart`.

A workflow may need to be resumed from a stable checkpoint after a process
restart.  This module provides that capability through a versioned, JSON-safe
payload (:class:`Snapshot`) that contains only data — never callables, locks,
futures or arbitrary pickled objects.

Checkpoints are restricted to **quiescent** configurations: the machine must be
activated, idle (no transition in progress), with empty event queues and no
pending or active invocation.  Attempting :func:`take_snapshot` in any other
state raises :class:`SnapshotNotQuiescentError` explaining why.

Restoration (:func:`restore_snapshot`) reuses the class definition to build a
fresh instance and then plants the saved configuration directly.  Entry/exit
callbacks and already-completed transition actions are therefore never
re-executed.  The snapshot is fully validated *before* any user object (model
or listener) is constructed or any user callback could run: format version,
definition fingerprint, state ids, parent/child containment, parallel-region
completeness and history records are all checked against the current
definition.  Missing states are never guessed from initial transitions.
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from .exceptions import InvalidSnapshot
from .exceptions import SnapshotNotQuiescentError
from .graph import iterate_states_and_transitions
from .orderedset import OrderedSet
from .state import HistoryState
from .transition import Transition

if TYPE_CHECKING:
    from .statemachine import StateChart

SNAPSHOT_FORMAT_VERSION = 1
"""Version of the snapshot payload produced by this library."""


# -- Definition fingerprint --------------------------------------------------


def _value_fingerprint(value: Any) -> str:
    """Stable, type-aware representation of a state's custom ``value``."""
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _action_specs(owner: Any) -> list[dict[str, Any]]:
    """Serialize the callback specs attached to a state or transition."""
    specs: list[dict[str, Any]] = []
    for spec in owner._specs:
        specs.append(
            {
                "group": str(spec.group),
                "reference": int(spec.reference),
                "name": str(spec),
                "convention": bool(spec.is_convention),
                "priority": int(spec.priority),
                "expected_value": spec.expected_value,
            }
        )
    return specs


def definition_payload(sm_class: "type[StateChart]") -> dict[str, Any]:
    """Build the canonical, JSON-serializable description of a machine definition.

    States, history pseudo-states, transitions (including initial and history
    default transitions) and every registered callback spec are included.  The
    callback bodies are not inspected — method/property *names* and ordering are
    part of the contract instead, keeping the fingerprint stable across
    processes.
    """
    states: list[dict[str, Any]] = []
    for state in iterate_states_and_transitions(sm_class.states):
        if isinstance(state, Transition):
            continue

        states.append(
            {
                "id": state.id,
                "parent": state.parent.id if state.parent else None,
                "name": state.name,
                "type": (
                    "history"
                    if isinstance(state, HistoryState)
                    else "parallel"
                    if state.parallel
                    else "compound"
                    if state.states
                    else "atomic"
                ),
                "history_kind": state.type.value if isinstance(state, HistoryState) else None,
                "initial": bool(state.initial),
                "final": bool(state.final),
                "order": state.document_order,
                "value": _value_fingerprint(state.value),
            }
        )

    transitions: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for owner in iterate_states_and_transitions(sm_class.states):
        if isinstance(owner, Transition):
            for target in owner.targets:
                transitions.append(
                    {
                        "source": owner.source.id,
                        "target": target.id,
                        "events": [str(event.id) for event in owner.events],
                        "internal": bool(owner.internal),
                        "initial": bool(owner.initial),
                    }
                )
        for spec in _action_specs(owner):
            owner_id = owner.id if hasattr(owner, "id") else owner.source.id
            actions.append({"owner": owner_id, **spec})

    return {
        "machine": sm_class.id,
        "states": states,
        "transitions": transitions,
        "actions": actions,
    }


def definition_fingerprint(sm_class: "type[StateChart]") -> str:
    """Return the SHA-256 fingerprint of a state machine definition.

    The payload is canonicalized with sorted keys so the hash only depends on
    the semantic content, not on dict insertion order.
    """
    canonical = json.dumps(
        definition_payload(sm_class), sort_keys=True, separators=(",", ":"), default=repr
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- Snapshot data structure -------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """An immutable, JSON-safe checkpoint of a quiescent state machine.

    Attributes:
        version: Payload format version.
        machine_id: The :attr:`StateChart.id` of the machine class.
        fingerprint: Definition fingerprint captured at export time.
        configuration: Active **leaf** (atomic) state ids.
        history: Mapping of history pseudo-state ids to remembered state ids.
        model_data: Caller-whitelisted model attributes (JSON-native values).
        running: ``False`` when the machine had reached a top-level final state.
        state_field: Name of the model attribute holding the raw state value.
    """

    version: int
    machine_id: str
    fingerprint: str
    configuration: list[str]
    history: dict[str, list[str]]
    model_data: dict[str, Any]
    running: bool = True
    state_field: str = "state"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` representation (deep-copied)."""
        data = {
            "version": self.version,
            "machine_id": self.machine_id,
            "fingerprint": self.fingerprint,
            "configuration": list(self.configuration),
            "history": {key: list(value) for key, value in self.history.items()},
            "model_data": copy.deepcopy(self.model_data),
            "running": self.running,
            "state_field": self.state_field,
        }
        # Fail fast if the payload cannot cross a JSON boundary: snapshots are
        # meant to be persisted, so accepting non-serializable data would only
        # surface the error later, far away from the caller.
        json.dumps(data)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Snapshot":
        """Build a :class:`Snapshot` from a plain mapping, rejecting malformed payloads."""
        if not isinstance(data, dict):
            raise InvalidSnapshot("Snapshot payload must be a mapping.")

        required = ("version", "machine_id", "fingerprint", "configuration", "history")
        missing = [key for key in required if key not in data]
        if missing:
            raise InvalidSnapshot(f"Snapshot is missing required field(s): {missing!r}.")

        if data["version"] != SNAPSHOT_FORMAT_VERSION:
            raise InvalidSnapshot(
                f"Unsupported snapshot version {data['version']!r}; "
                f"expected {SNAPSHOT_FORMAT_VERSION}."
            )

        configuration = data["configuration"]
        history = data["history"]
        if not isinstance(configuration, list) or not all(
            isinstance(x, str) for x in configuration
        ):
            raise InvalidSnapshot("'configuration' must be a list of state id strings.")
        if not isinstance(history, dict) or not all(
            isinstance(key, str)
            and isinstance(value, list)
            and all(isinstance(x, str) for x in value)
            for key, value in history.items()
        ):
            raise InvalidSnapshot("'history' must map history ids to lists of state id strings.")

        model_data = data.get("model_data", {})
        if not isinstance(model_data, dict):
            raise InvalidSnapshot("'model_data' must be an object/mapping.")
        try:
            json.dumps(model_data)
        except TypeError as err:
            raise InvalidSnapshot(f"'model_data' is not JSON-serializable: {err}") from err

        running = data.get("running", True)
        if not isinstance(running, bool):
            raise InvalidSnapshot("'running' must be a boolean.")

        state_field = data.get("state_field", "state")
        if not isinstance(state_field, str) or not state_field:
            raise InvalidSnapshot("'state_field' must be a non-empty string.")

        return cls(
            version=SNAPSHOT_FORMAT_VERSION,
            machine_id=str(data["machine_id"]),
            fingerprint=str(data["fingerprint"]),
            configuration=list(configuration),
            history={key: list(value) for key, value in history.items()},
            model_data=dict(model_data),
            running=running,
            state_field=state_field,
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the snapshot to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str | bytes) -> "Snapshot":
        """Parse a JSON string into a :class:`Snapshot`."""
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as err:
            raise InvalidSnapshot(f"Snapshot is not valid JSON: {err}") from err


# -- Quiescence --------------------------------------------------------------


def _quiescence_error(sm: "StateChart", *, lock_held_by_caller: bool = False):
    """Return an error explaining why *sm* cannot be checkpointed, or ``None``.

    Args:
        lock_held_by_caller: When ``True`` the caller already holds the engine
            processing lock (so a locked lock means a callback on the *same*
            thread/task is requesting the snapshot, which is forbidden).  When
            ``False`` a locked lock means another worker owns the machine.
    """
    engine = sm._engine

    if sm.current_state_value is None:
        return SnapshotNotQuiescentError(
            "Cannot take a snapshot before the initial state is activated."
        )

    # A callback taking a snapshot runs inside the processing loop; the lock is
    # then held by the current thread/task, so a non-blocking acquire detects it.
    if engine._processing.locked() and not lock_held_by_caller:
        return SnapshotNotQuiescentError(
            "Cannot take a snapshot while a transition or callback is being executed; "
            "checkpoints are only allowed at stable (quiescent) configurations."
        )

    if not engine.external_queue.is_empty():
        return SnapshotNotQuiescentError(
            "Cannot take a snapshot with non-empty external event queue; "
            "process or cancel pending events first."
        )

    if not engine.internal_queue.is_empty():
        return SnapshotNotQuiescentError(
            "Cannot take a snapshot with non-empty internal event queue; "
            "let the current macrostep drain the internal queue first."
        )

    invoke_manager = engine._invoke_manager
    if invoke_manager._pending:
        return SnapshotNotQuiescentError(
            "Cannot take a snapshot with invocations pending creation; "
            "let the processing loop spawn them or exit the owning states."
        )

    # Mirror InvokeManager._cleanup_terminated: a terminated-and-cancelled
    # invocation is historical bookkeeping, but anything still alive (or
    # terminated while its state remains active, e.g. a child machine that
    # must still receive events) forbids checkpointing.
    active = [
        inv
        for inv in invoke_manager._active.values()
        if not (inv.terminated and inv.ctx.cancelled.is_set())
    ]
    if active:
        ids = ", ".join(sorted({inv.invokeid for inv in active}))
        return SnapshotNotQuiescentError(
            f"Cannot take a snapshot with active invocation(s): {ids}; "
            "cancel or wait for invoked work to finish first."
        )

    return None


def assert_quiescent(sm: "StateChart") -> None:
    """Raise :class:`SnapshotNotQuiescentError` unless *sm* is checkpointable."""
    error = _quiescence_error(sm)
    if error is not None:
        raise error


def _assert_quiescent_unlocked(sm: "StateChart") -> None:
    """Quiescence checks that still apply once the processing lock is held."""
    error = _quiescence_error(sm, lock_held_by_caller=True)
    if error is not None:
        raise error


# -- Export ------------------------------------------------------------------


def _extract_model_data(sm: "StateChart", include: "list[str] | tuple[str, ...] | None"):
    if include is None:
        return {}
    if not isinstance(include, (list, tuple)) or not all(
        isinstance(name, str) for name in include
    ):
        raise InvalidSnapshot("'include_model_data' must be a list of attribute name strings.")

    state_field = sm.state_field
    data: dict[str, Any] = {}
    for name in include:
        if name == state_field:
            raise InvalidSnapshot(
                f"Cannot export {state_field!r} via 'include_model_data': it is the managed "
                "state field and is restored from 'configuration'."
            )
        if not hasattr(sm.model, name):
            raise InvalidSnapshot(f"Model has no attribute named {name!r}.")
        data[name] = getattr(sm.model, name)
    return data


def _state_id_index(sm_class: "type[StateChart]") -> dict[str, Any]:
    """Map every state/history id to its definition object."""
    return {
        state.id: state
        for state in iterate_states_and_transitions(sm_class.states)
        if not isinstance(state, Transition)
    }


def take_snapshot(
    sm: "StateChart",
    include_model_data: "list[str] | tuple[str, ...] | None" = None,
) -> Snapshot:
    """Export a :class:`Snapshot` of a quiescent *sm*.

    Args:
        sm: The state machine to checkpoint.
        include_model_data: Explicit whitelist of model attribute names whose
            values are copied into the snapshot.  Values must be JSON-native
            (``str``, ``int``, ``float``, ``bool``, ``None``, and lists/tuples
            or dicts composed of those).  Callables and arbitrary objects are
            rejected.  Omit to persist only the machine configuration.
    """
    # Hold the processing lock while reading so a concurrent thread/task cannot
    # start a transition between the quiescence check and the configuration read.
    lock = sm._engine._processing
    if not lock.acquire(blocking=False):
        raise SnapshotNotQuiescentError(
            "Cannot take a snapshot while a transition or callback is being executed; "
            "checkpoints are only allowed at stable (quiescent) configurations."
        )
    try:
        _assert_quiescent_unlocked(sm)

        sm_class = type(sm)
        leaf_ids = [
            state.id
            for state in sorted(
                (state for state in sm.configuration if state.is_atomic),
                key=lambda state: state.document_order,
            )
        ]

        history = {
            history_id: [state.id for state in recorded_states]
            for history_id, recorded_states in sm.history_values.items()
        }

        snapshot = Snapshot(
            version=SNAPSHOT_FORMAT_VERSION,
            machine_id=sm_class.id,
            fingerprint=definition_fingerprint(sm_class),
            configuration=leaf_ids,
            history=history,
            model_data=_extract_model_data(sm, include_model_data),
            running=bool(sm._engine.running),
            state_field=sm.state_field,
        )
        # Validates JSON-serializability of the whole payload up front.
        try:
            snapshot.to_dict()
        except TypeError as err:
            raise InvalidSnapshot(
                f"Snapshot payload is not JSON-serializable: {err}"
            ) from err
        return snapshot
    finally:
        lock.release()


# -- Restoration validation --------------------------------------------------


def _validate_structure(snapshot: Snapshot, sm_class: "type[StateChart]") -> dict[str, Any]:
    """Validate the payload against a definition without touching any instance.

    Returns the id -> definition index on success.  All structural checks run
    here, before any model, listener or machine instance is constructed, so a
    corrupt snapshot can never reach user code.
    """
    id_index = _state_id_index(sm_class)

    if snapshot.machine_id != sm_class.id:
        raise InvalidSnapshot(
            f"Snapshot belongs to machine {snapshot.machine_id!r} but was restored "
            f"using {sm_class.id!r}."
        )

    current_fingerprint = definition_fingerprint(sm_class)
    if snapshot.fingerprint != current_fingerprint:
        raise InvalidSnapshot(
            "Snapshot definition fingerprint does not match the current class definition; "
            "the state machine changed since the snapshot was taken."
        )

    leaves = snapshot.configuration
    if not leaves:
        raise InvalidSnapshot("Snapshot configuration is empty; no leaf states to restore.")
    if len(set(leaves)) != len(leaves):
        raise InvalidSnapshot("Snapshot configuration contains duplicate leaf states.")

    leaf_states: list[Any] = []
    for state_id in leaves:
        state = id_index.get(state_id)
        if state is None:
            raise InvalidSnapshot(f"Unknown state id {state_id!r} in snapshot configuration.")
        if isinstance(state, HistoryState):
            raise InvalidSnapshot(
                f"History pseudo-state {state_id!r} cannot be an active leaf configuration."
            )
        if not state.is_atomic:
            raise InvalidSnapshot(
                f"Snapshot configuration must list only leaf states, but {state_id!r} is "
                f"a {'parallel' if state.parallel else 'compound'} state."
            )
        leaf_states.append(state)

    leaf_set = set(leaf_states)

    def check_subtree(state: Any) -> None:
        """Enforce the single-child / full-region rules for every active container."""
        active_children = [
            child
            for child in state.states
            if child in leaf_set
            or any(leaf.is_descendant(child) for leaf in leaf_states)
        ]
        if state.parallel:
            if len(active_children) != len(state.states):
                missing = [child.id for child in state.states if child not in active_children]
                raise InvalidSnapshot(
                    f"Incomplete parallel configuration for {state.id!r}: "
                    f"missing region(s) {missing!r}."
                )
        elif len(active_children) != 1:
            raise InvalidSnapshot(
                f"Compound state {state.id!r} has {len(active_children)} active child regions "
                f"{[child.id for child in active_children]!r}; a non-parallel compound can hold "
                "exactly one."
            )
        for child in active_children:
            if not child.is_atomic:
                check_subtree(child)

    roots_with_leaves = [
        top
        for top in sm_class.states
        if top in leaf_set or any(leaf.is_descendant(top) for leaf in leaf_states)
    ]
    if len(roots_with_leaves) != 1:
        raise InvalidSnapshot(
            "Snapshot must activate exactly one top-level state subtree, found "
            f"{[top.id for top in roots_with_leaves]!r}."
        )
    root = roots_with_leaves[0]
    if not root.is_atomic:
        check_subtree(root)

    # History records: ids must exist, point to the owning compound's subtree,
    # and respect the shallow (direct children) vs deep (any descendant) kind.
    for history_id, recorded_ids in snapshot.history.items():
        history_state = id_index.get(history_id)
        if history_state is None or not isinstance(history_state, HistoryState):
            raise InvalidSnapshot(f"Unknown history pseudo-state id {history_id!r}.")
        parent = history_state.parent
        if parent is None:  # pragma: no cover - a history state always nests in a compound
            raise InvalidSnapshot(f"History state {history_id!r} has no owning compound.")
        if not recorded_ids:
            raise InvalidSnapshot(f"History record for {history_id!r} is empty.")
        if len(set(recorded_ids)) != len(recorded_ids):
            raise InvalidSnapshot(f"History record for {history_id!r} contains duplicates.")
        for recorded_id in recorded_ids:
            recorded = id_index.get(recorded_id)
            if recorded is None or isinstance(recorded, HistoryState):
                raise InvalidSnapshot(
                    f"History {history_id!r} references unknown state {recorded_id!r}."
                )
            if history_state.type.is_deep:
                valid = recorded.is_descendant(parent)
            else:
                valid = recorded.parent is parent
            if not valid:
                kind = "deep" if history_state.type.is_deep else "shallow"
                relation = "descendants" if history_state.type.is_deep else "direct children"
                raise InvalidSnapshot(
                    f"{kind.capitalize()} history {history_id!r} may only remember "
                    f"{parent.id!r} {relation}, but {recorded_id!r} is not."
                )

    return id_index


# -- Restoration -------------------------------------------------------------


def _raw_configuration_value(configuration: list[str], id_index: dict[str, Any]):
    """Convert leaf state ids to the raw value stored on the model.

    The runtime configuration holds every active state — both leaves and their
    compound/parallel ancestors — so the ancestor closure is reconstructed here.
    """
    states: OrderedSet[Any] = OrderedSet()
    for state_id in configuration:
        leaf = id_index[state_id]
        states.add(leaf)
        for ancestor in leaf.ancestors():
            states.add(ancestor)
    values = OrderedSet(state.value for state in states)
    if len(values) == 1:
        return next(iter(values))
    return values


def restore_snapshot(
    sm_class: "type[StateChart]",
    snapshot: "Snapshot | dict[str, Any] | str | bytes",
    model: Any = None,
    *,
    listeners: "list[object] | None" = None,
    **kwargs: Any,
) -> "StateChart":
    """Build a new machine instance from *snapshot* without replaying actions.

    The snapshot is fully validated against *sm_class* before any user object
    is constructed; an invalid or tampered snapshot therefore fails before a
    single entry/exit or transition callback can run.  On success the fresh
    instance starts directly in the saved configuration — no ``__initial__``
    activation, entry/exit handlers or completed transition actions run.

    Args:
        sm_class: The :class:`StateChart` subclass matching the snapshot.
        snapshot: A :class:`Snapshot`, its ``dict`` form, or a JSON string.
        model: Model instance to attach.  When omitted a fresh internal
            :class:`~statemachine.model.Model` is created.
        listeners: Runtime listeners attached to the restored instance.
        **kwargs: Forwarded to the class constructor (listener ``setup()``
            dependency injection).
    """
    if isinstance(snapshot, (str, bytes)):
        snapshot = Snapshot.from_json(snapshot)
    elif isinstance(snapshot, dict):
        snapshot = Snapshot.from_dict(snapshot)
    elif not isinstance(snapshot, Snapshot):
        raise InvalidSnapshot(
            f"Cannot restore snapshot of type {type(snapshot).__name__!r}; expected Snapshot, "
            "dict or JSON string."
        )

    # All structural/definition checks happen before any instantiation.
    id_index = _validate_structure(snapshot, sm_class)

    if snapshot.state_field in snapshot.model_data:
        raise InvalidSnapshot(
            f"'model_data' must not contain the managed state field {snapshot.state_field!r}."
        )

    # Plant the raw configuration on the model BEFORE constructing the machine.
    # Both engines' start() is a no-op when a current state exists, so the
    # __initial__ activation — and hence every entry callback — is skipped.
    if model is None:
        from .model import Model

        model = Model()

    raw_value = _raw_configuration_value(snapshot.configuration, id_index)
    setattr(model, snapshot.state_field, raw_value)

    try:
        sm = sm_class(
            model=model,
            state_field=snapshot.state_field,
            listeners=listeners or [],
            **kwargs,
        )
    except TypeError as err:
        raise InvalidSnapshot(
            f"{sm_class.__name__} does not accept the standard restore constructor "
            f"signature (model, state_field, listeners): {err}"
        ) from err

    # Restore history records (id -> State objects in the new instance).
    sm.history_values = {
        history_id: [id_index[state_id] for state_id in recorded_ids]
        for history_id, recorded_ids in snapshot.history.items()
    }

    # The constructor cleared/restored the model anyway, but re-apply whitelisted
    # data explicitly in case a custom __init__ replaced attributes.
    for name, value in snapshot.model_data.items():
        setattr(sm.model, name, copy.deepcopy(value))

    if not snapshot.running:
        sm._engine.running = False

    return sm
