"""Explicit export/restore of quiescent state machine configurations ("snapshots").

A snapshot is a plain JSON document capturing everything needed to continue a
workflow after a process restart — and nothing that cannot cross a process
boundary: functions, locks, futures, threads and arbitrary pickled objects are
never serialized.

Snapshot document::

    {
        "version": 1,
        "machine": "<class name>",
        "fingerprint": "<sha256 of the statechart definition>",
        "state_field": "state",
        "terminated": false,
        "configuration": ["<state id>", ...],
        "history": {"<history pseudo-state id>": ["<state id>", ...]},
        "model": {"<allowed attribute>": <json value>, ...}
    }

Snapshots may only be taken at a **stable configuration**:

* no callback or transition is currently executing;
* the external and internal event queues are empty;
* no invoked child session or handler is still active.

Restore validates the document (version, definition fingerprint, state ids,
parent/child constraints, parallel-region completeness) *before* instantiating
the machine, so a tampered or stale snapshot fails before a single user
callback can run. Restore never executes entry/exit actions or already
completed transition actions, and a missing state is never guessed as an
initial state.
"""

import hashlib
import json
from typing import TYPE_CHECKING
from typing import Any

from .callbacks import CallbacksRegistry
from .exceptions import StateMachineError
from .graph import iterate_states
from .model import Model
from .orderedset import OrderedSet
from .state import HistoryState
from .state import State

if TYPE_CHECKING:
    from .statemachine import StateChart

SNAPSHOT_VERSION = 1
"""Version of the snapshot document format."""

_DOCUMENT_KEYS = (
    "version",
    "machine",
    "fingerprint",
    "state_field",
    "terminated",
    "configuration",
    "history",
    "model",
)
_JSON_TYPES = (str, int, float, bool, type(None), list, dict)


class SnapshotError(StateMachineError):
    """Base class for snapshot export/restore errors."""


class SnapshotNotStable(SnapshotError):
    """The machine is not at a stable configuration and cannot be snapshotted."""


class SnapshotDefinitionMismatch(SnapshotError):
    """The snapshot does not match the current state machine definition."""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def take_snapshot(
    sm: "StateChart",
    model_fields: "list[str] | tuple[str, ...] | None" = None,
) -> dict[str, Any]:
    """Export a stable configuration of *sm* as a JSON-safe dict.

    Args:
        sm: The state machine to snapshot.
        model_fields: Optional allowlist of ``sm.model`` attribute names whose
            current values are stored. Values must be JSON-native. Only the
            allowlisted attributes are restored; callbacks and other model
            methods are never serialized.

    Raises:
        SnapshotNotStable: If a transition/callback is running, an event queue
            is non-empty, or an invocation is still active.
        SnapshotError: If an allowlisted model value is not JSON-serializable.
    """
    _assert_quiescent(sm)

    configuration = sorted((state.id for state in sm.configuration))
    if not configuration:
        raise SnapshotError("Cannot snapshot an empty configuration.")

    history = {
        history_id: [
            state.id
            for state in sorted(states, key=lambda state: state.document_order)
        ]
        for history_id, states in sm.history_values.items()
    }

    model_data = _export_model_data(sm, model_fields)

    snapshot = {
        "version": SNAPSHOT_VERSION,
        "machine": type(sm).__name__,
        "fingerprint": definition_fingerprint(type(sm)),
        "state_field": sm.state_field,
        "terminated": sm.is_terminated,
        "configuration": configuration,
        "history": history,
        "model": model_data,
    }
    # Defensive: the document must always be JSON-encodable as-is.
    _json_dump(snapshot)
    return snapshot


def _assert_quiescent(sm: "StateChart") -> None:
    engine = sm._engine
    if not engine._processing.acquire(blocking=False):
        raise SnapshotNotStable(
            "Cannot take a snapshot while a callback or transition is executing."
        )
    try:
        if not engine.internal_queue.is_empty():
            raise SnapshotNotStable(
                "Cannot take a snapshot while the internal event queue is not empty."
            )
        if not engine.external_queue.is_empty():
            raise SnapshotNotStable(
                "Cannot take a snapshot while the external event queue is not empty."
            )
        invoke_manager = engine._invoke_manager
        if invoke_manager._pending:
            raise SnapshotNotStable(
                "Cannot take a snapshot while an invocation is pending for "
                f"{[state.id for state, _ in invoke_manager._pending]}."
            )
        active = [
            invocation
            for invocation in invoke_manager._active.values()
            if not invocation.terminated
        ]
        if active:
            raise SnapshotNotStable(
                "Cannot take a snapshot while an active invocation is running for "
                f"{sorted({invocation.state_id for invocation in active})}."
            )
    finally:
        engine._processing.release()


def _export_model_data(sm: "StateChart", model_fields: "list[str] | tuple[str, ...] | None"):
    if model_fields is None:
        return {}
    model_data: dict[str, Any] = {}
    for field_name in model_fields:
        if not hasattr(sm.model, field_name):
            raise SnapshotError(
                f"Model {type(sm.model).__name__!r} has no attribute {field_name!r} allowed "
                "by model_fields."
            )
        value = getattr(sm.model, field_name)
        if field_name == sm.state_field:
            raise SnapshotError(
                f"Model field {field_name!r} is managed by the state machine and is restored "
                "from 'configuration'; remove it from model_fields."
            )
        _ensure_json_value(value, field_name)
        model_data[field_name] = value
    return model_data


def _ensure_json_value(value: Any, location: str) -> None:
    if not isinstance(value, _JSON_TYPES):
        raise SnapshotError(
            f"Model value for {location!r} is of type {type(value).__name__}, which is not "
            "JSON-serializable."
        )
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SnapshotError(
                    f"Model mapping for {location!r} has a non-string key {key!r}."
                )
            _ensure_json_value(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_value(item, f"{location}[{index}]")


# ---------------------------------------------------------------------------
# Definition fingerprint
# ---------------------------------------------------------------------------


def definition_fingerprint(machine_cls: "type[StateChart]") -> str:
    """Return a stable sha256 fingerprint of a state machine *definition*.

    The fingerprint covers the state hierarchy (ids, names, parallel/final
    flags), declared transitions (events, sources, targets, internal/initial
    flags) and the names of every declared callback/guard. Two classes with the
    same name but different definitions therefore produce different
    fingerprints and snapshots cannot be restored against a changed
    definition.
    """
    payload = {
        "name": machine_cls.__name__,
        "states": [_state_payload(state) for state in machine_cls.states],
        "transitions": [
            _transition_payload(transition)
            for state in iterate_states(machine_cls.states)
            for transition in state.transitions
            if not transition.initial
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_payload(state: State) -> dict[str, Any]:
    return {
        "id": state.id,
        "name": state.name,
        "parallel": state.parallel,
        "final": state.final,
        "initial": state.initial,
        "history": [
            {"id": history.id, "deep": history.type.is_deep} for history in state.history
        ],
        "states": [_state_payload(child) for child in state.states],
    }


def _transition_payload(transition: Any) -> dict[str, Any]:
    return {
        "events": sorted(event.id for event in transition._events),
        "source": transition.source.id,
        "targets": [target.id for target in transition.targets],
        "internal": transition.internal,
        "callbacks": sorted(
            f"{spec.group.name}:{spec.attr_name}"
            for spec in transition._specs
            if not spec.is_convention
        ),
    }


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def restore(
    machine_cls: "type[StateChart]",
    snapshot: "dict[str, Any] | str | bytes",
    model: Any = None,
    listeners: "list[object] | None" = None,
) -> "StateChart":
    """Restore a state machine from a snapshot document.

    All structural validation runs before the machine is instantiated, so an
    invalid snapshot cannot trigger user callbacks (including ``setup()``
    listeners or entry actions). No entry/exit actions run during restore.

    Args:
        machine_cls: The state machine class whose definition produced the
            snapshot.
        snapshot: A snapshot dict, or a JSON string/bytes produced by
            ``json.dumps(sm.snapshot())``.
        model: Model instance to attach. A fresh internal model is created
            when omitted.
        listeners: Runtime listeners attached after construction. Class-level
            listeners declared on the class are also resolved.

    Raises:
        SnapshotDefinitionMismatch: Version, machine id, fingerprint or
            configuration/history validation failure.
    """
    data = _load_snapshot(snapshot)
    states_by_id = _states_by_id(machine_cls)
    _validate_document(machine_cls, data, states_by_id)

    active_states = [states_by_id[state_id] for state_id in data["configuration"]]
    restored_history = {
        history_id: [states_by_id[state_id] for state_id in state_ids]
        for history_id, state_ids in data["history"].items()
    }

    instance = machine_cls.__new__(machine_cls)
    target_model = model if model is not None else Model()
    instance.model = target_model
    instance.state_field = data["state_field"]
    instance.history_values = {}
    instance.start_configuration_values = []
    instance._callbacks = CallbacksRegistry()
    instance._config = instance._build_configuration()
    instance._listeners = {}
    instance._engine = instance._get_engine()

    instance._register_callbacks(list(listeners or []))

    # Restore runtime state directly — no entry/exit, no initial transition.
    instance._config.states = _ordered_configuration(active_states)
    instance.history_values = restored_history
    instance._engine.running = not data["terminated"]

    for field_name, value in data["model"].items():
        setattr(target_model, field_name, value)

    return instance


def _ordered_configuration(states: "list[State]"):
    return OrderedSet(sorted(states, key=lambda state: state.document_order))


def _load_snapshot(snapshot: "dict[str, Any] | str | bytes") -> dict[str, Any]:
    if isinstance(snapshot, dict):
        data = snapshot
    else:
        try:
            data = json.loads(snapshot)
        except (TypeError, ValueError) as err:
            raise SnapshotDefinitionMismatch(f"Snapshot is not valid JSON: {err}") from err
    if not isinstance(data, dict):
        raise SnapshotDefinitionMismatch("Snapshot document must be a JSON object.")
    return data


def _validate_document(
    machine_cls: "type[StateChart]", data: dict[str, Any], states_by_id: dict[str, State]
) -> None:
    extra = set(data) - set(_DOCUMENT_KEYS)
    if extra:
        raise SnapshotDefinitionMismatch(f"Snapshot has unknown keys: {sorted(extra)}.")
    missing = set(_DOCUMENT_KEYS) - set(data)
    if missing:
        raise SnapshotDefinitionMismatch(f"Snapshot is missing keys: {sorted(missing)}.")

    if data["version"] != SNAPSHOT_VERSION:
        raise SnapshotDefinitionMismatch(
            f"Unsupported snapshot version {data['version']!r}; expected {SNAPSHOT_VERSION}."
        )
    if data["machine"] != machine_cls.__name__:
        raise SnapshotDefinitionMismatch(
            f"Snapshot was created for machine {data['machine']!r}, but "
            f"{machine_cls.__name__!r} was provided for restore."
        )
    if data["fingerprint"] != definition_fingerprint(machine_cls):
        raise SnapshotDefinitionMismatch(
            "Snapshot fingerprint does not match the current machine definition; "
            "the definition has changed since the snapshot was taken."
        )

    state_field = data["state_field"]
    if not isinstance(state_field, str) or not state_field:
        raise SnapshotDefinitionMismatch("Snapshot 'state_field' must be a non-empty string.")

    configuration = data["configuration"]
    history = data["history"]
    model_data = data["model"]
    if not isinstance(configuration, list) or not configuration:
        raise SnapshotDefinitionMismatch(
            "Snapshot 'configuration' must be a non-empty list of state ids."
        )
    if not isinstance(history, dict) or not all(
        isinstance(key, str) and isinstance(value, list) for key, value in history.items()
    ):
        raise SnapshotDefinitionMismatch(
            "Snapshot 'history' must map history pseudo-state ids to lists of state ids."
        )
    if not isinstance(model_data, dict) or not all(isinstance(key, str) for key in model_data):
        raise SnapshotDefinitionMismatch("Snapshot 'model' must be an object with string keys.")
    if state_field in model_data:
        raise SnapshotDefinitionMismatch(
            f"Snapshot 'model' must not contain the managed state field {state_field!r}."
        )
    if not isinstance(data["terminated"], bool):
        raise SnapshotDefinitionMismatch("Snapshot 'terminated' must be a boolean.")

    active = _resolve_configuration(states_by_id, configuration)
    _validate_configuration(active, data["terminated"])
    _validate_history(states_by_id, history)


def _states_by_id(machine_cls: "type[StateChart]") -> dict[str, State]:
    # ``states_map`` is keyed by value and excludes history pseudo-states, so
    # walk the declared hierarchy to collect every state (incl. history) by id.
    return {state.id: state for state in iterate_states(machine_cls.states)}


def _resolve_configuration(states_by_id: dict[str, State], configuration: list) -> list[State]:
    active: list[State] = []
    for state_id in configuration:
        if not isinstance(state_id, str):
            raise SnapshotDefinitionMismatch(
                f"Snapshot 'configuration' contains a non-string entry {state_id!r}."
            )
        state = states_by_id.get(state_id)
        if state is None:
            raise SnapshotDefinitionMismatch(
                f"Unknown state id {state_id!r} in snapshot configuration."
            )
        if isinstance(state, HistoryState):
            raise SnapshotDefinitionMismatch(
                f"History pseudo-state {state_id!r} cannot be part of the active configuration."
            )
        active.append(state)
    return active


def _validate_configuration(active: "list[State]", terminated: bool) -> None:
    """Validate parent/child and parallel-region constraints."""
    active_set = set(active)

    for state in active:
        # Ancestor closure: being in a child means being in every ancestor.
        for ancestor in state.ancestors():
            if ancestor not in active_set:
                raise SnapshotDefinitionMismatch(
                    f"Snapshot configuration contains {state.id!r} but is missing its ancestor "
                    f"{ancestor.id!r}."
                )

    # Group *direct* active children by their parent.
    children_by_parent: dict[State, list[State]] = {}
    for state in active:
        if state.parent is not None:
            children_by_parent.setdefault(state.parent, []).append(state)

    for parent in sorted(children_by_parent, key=lambda state: state.document_order):
        children = children_by_parent[parent]
        if parent.parallel:
            # Every region of an active parallel state must be represented.
            missing_regions = [
                region.id
                for region in parent.states
                if not _has_active_state_in_region(region, active_set)
            ]
            if missing_regions:
                raise SnapshotDefinitionMismatch(
                    "Snapshot configuration does not cover every region of parallel state "
                    f"{parent.id!r}; missing {missing_regions}."
                )
        elif len(children) > 1:
            raise SnapshotDefinitionMismatch(
                "Snapshot configuration activates sibling states "
                f"{sorted(state.id for state in children)} of non-parallel state "
                f"{parent.id!r}."
            )

    # Exactly one top-level state.
    top_level = [state for state in active if state.parent is None]
    if len(top_level) != 1:  # pragma: no branch - ancestor resolution keeps the root
        raise SnapshotDefinitionMismatch(
            "Snapshot configuration must contain exactly one top-level state, got "
            f"{sorted(state.id for state in top_level)}."
        )
    root = top_level[0]
    if terminated:
        if not _root_is_in_final_state(root, active_set):
            raise SnapshotDefinitionMismatch(
                "Snapshot is marked terminated but the top-level state "
                f"{root.id!r} is not in a final configuration."
            )
    elif root.final:
        raise SnapshotDefinitionMismatch(
            f"Snapshot configuration contains final top-level state {root.id!r} but "
            "'terminated' is false."
        )

    # Every active non-atomic state must have at least one active child.
    for state in active:
        if state.states and not children_by_parent.get(state):
            raise SnapshotDefinitionMismatch(
                f"Snapshot configuration contains {state.id!r} without any of its children."
            )


def _has_active_state_in_region(region: State, active_set: set[State]) -> bool:
    # An atomic region is represented by itself; a compound region requires an
    # active descendant state.
    if not region.states:
        return region in active_set
    return any(active.is_descendant(region) for active in active_set)


def _root_is_in_final_state(root: State, active_set: set[State]) -> bool:
    """Final configuration of the unique top-level state.

    A compound/atomic root is final when the root itself is a final state; a
    parallel root is final when every one of its regions is final.
    """
    if root.final:
        return True
    if not root.parallel:
        return False
    return all(_region_is_final(region, active_set) for region in root.states)


def _region_is_final(region: State, active_set: set[State]) -> bool:
    """A parallel region is final when its active chain reaches a final state."""
    if not region.states:
        return region in active_set and region.final
    return any(active.final and active.is_descendant(region) for active in active_set)


def _validate_history(states_by_id: dict[str, State], history: dict[str, list]) -> None:
    for history_id, state_ids in history.items():
        history_state = states_by_id.get(history_id)
        if history_state is None or not isinstance(history_state, HistoryState):
            raise SnapshotDefinitionMismatch(
                f"Snapshot history entry {history_id!r} is not a declared history pseudo-state."
            )
        owner = history_state.parent
        for state_id in state_ids:
            if not isinstance(state_id, str):
                raise SnapshotDefinitionMismatch(
                    f"Snapshot history {history_id!r} contains a non-string entry {state_id!r}."
                )
            target = states_by_id.get(state_id)
            if target is None:
                raise SnapshotDefinitionMismatch(
                    f"Unknown state id {state_id!r} in snapshot history {history_id!r}."
                )
            if isinstance(target, HistoryState):
                raise SnapshotDefinitionMismatch(
                    f"History pseudo-state {state_id!r} cannot be stored in history "
                    f"{history_id!r}."
                )
            if history_state.type.is_deep:
                valid = target.is_descendant(owner)
            else:
                valid = target.parent is owner
            if not valid:
                kind = "deep" if history_state.type.is_deep else "shallow"
                raise SnapshotDefinitionMismatch(
                    f"History entry {state_id!r} is not a {kind}-history child of {owner.id!r}."
                )


def _json_dump(payload: dict[str, Any]) -> None:
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as err:  # pragma: no cover - values pre-validated
        raise SnapshotError(f"Snapshot payload is not JSON-serializable: {err}") from err


# Public functional alias mirroring the ``StateChart.restore`` classmethod.
def restore_snapshot(machine_cls, snapshot, model=None, listeners=None):
    """Restore a state machine from *snapshot*; alias of :func:`restore`."""
    return restore(machine_cls, snapshot, model=model, listeners=listeners)
