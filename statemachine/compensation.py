"""Compensable microstep support (saga-style rollback).

When ``compensable_microstep = True`` on a state machine, the engine treats
each microstep as an atomic transaction over the ``before`` → ``exit`` →
``on`` → ``enter`` action groups.

Every action in those groups must be explicitly declared **side-effect-free**
(:func:`side_effect_free`) or registered together with a **compensation
function** (:func:`compensable`). If an action raises, only the actions that
already completed are compensated, in reverse order; each receives the
credential returned by its original action. The active configuration and
history are restored to what they were before the microstep.

Microsteps involving ``invoke`` handlers or undeclared actions are rejected
*before* anything executes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from inspect import isawaitable
from inspect import Parameter
from typing import TYPE_CHECKING
from typing import Any

from .callbacks import ACTION_META_ATTR
from .exceptions import UnsupportedCompensableAction
from .i18n import _
from .orderedset import OrderedSet
from .signature import SignatureAdapter

if TYPE_CHECKING:
    from .state import State


class ActionKind(Enum):
    """How an action participates in a compensable microstep."""

    SIDE_EFFECT_FREE = "side_effect_free"
    """The action has no external side effects; nothing has to be undone."""

    COMPENSABLE = "compensable"
    """The action has a compensation function that undoes its side effects."""


@dataclass(frozen=True)
class ActionMeta:
    """Metadata describing how an action participates in compensation.

    Attached to the user function under :data:`~statemachine.callbacks.ACTION_META_ATTR`.
    """

    kind: ActionKind
    undo: "Callable[..., Any] | str | None" = None
    """Compensation function, or the name of a method on the machine/model/listeners."""

    label: str = ""
    """Human-readable label used in diagnostics."""


def side_effect_free(func: Callable) -> Callable:
    """Declare an action as having no external side effects.

    Side-effect-free actions are allowed inside a compensable microstep without
    a compensation function. Use this for pure computations and other effects
    that need no undo.
    """
    setattr(
        func,
        ACTION_META_ATTR,
        ActionMeta(kind=ActionKind.SIDE_EFFECT_FREE, label=func.__name__),
    )
    return func


def compensable(
    undo: "Callable[..., Any] | str",
    *,
    label: str = "",
) -> Callable:
    """Register an action together with its compensation function.

    Args:
        undo: Called to undo the action if a later action in the same microstep
            fails. It receives the same dependency-injected arguments as the
            original action plus ``credential`` — the value returned by the
            original action. May be a callable or the name of a method on the
            state machine, its model, or a listener.
        label: Optional human-readable label used in diagnostics.

    The decorator preserves the function's signature and coroutine status, so it
    composes freely with the ``@<event>.on`` / ``@state.enter`` /
    ``@state.exit`` decorators.
    """

    def decorator(func: Callable) -> Callable:
        setattr(
            func,
            ACTION_META_ATTR,
            ActionMeta(
                kind=ActionKind.COMPENSABLE,
                undo=undo,
                label=label or func.__name__,
            ),
        )
        return func

    return decorator


def get_action_meta(func: Any) -> "ActionMeta | None":
    """Return the :class:`ActionMeta` attached to *func*, or ``None``."""
    return getattr(func, ACTION_META_ATTR, None)


@dataclass
class CompensationEntry:
    """A successfully completed action waiting to be compensated."""

    label: str
    undo: "Callable[..., Any] | str"
    credential: Any
    args: tuple
    kwargs: dict


@dataclass
class ConfigurationSnapshot:
    """Captured microstep state restored when compensation runs."""

    configuration: OrderedSet[State]
    history_values: dict[str, list[State]]
    internal_events: list[Any]
    running: bool
    root_parallel_final_pending: "State | None"


class CompensationManager:
    """Tracks completed actions during one compensable microstep and undoes them.

    A fresh instance is pushed on the engine for the duration of a single
    microstep (``before`` → ``exit`` → ``on`` → ``enter``).
    """

    def __init__(self, engine: Any):
        self.engine = engine
        self.stack: list[CompensationEntry] = []

    @property
    def sm(self):
        return self.engine.sm

    def snapshot(self) -> ConfigurationSnapshot:
        return ConfigurationSnapshot(
            configuration=OrderedSet(self.sm.configuration),
            history_values={key: list(states) for key, states in self.sm.history_values.items()},
            internal_events=self.engine.internal_queue.snapshot(),
            running=self.engine.running,
            root_parallel_final_pending=self.engine._root_parallel_final_pending,
        )

    def restore(self, snapshot: ConfigurationSnapshot) -> None:
        self.sm.configuration = snapshot.configuration
        self.sm.history_values = snapshot.history_values
        self.engine.internal_queue.restore(snapshot.internal_events)
        self.engine.running = snapshot.running
        self.engine._root_parallel_final_pending = snapshot.root_parallel_final_pending

    def resolve_undo(self, undo: "Callable[..., Any] | str") -> Callable:
        """Resolve an undo reference into a bound callable."""
        if not isinstance(undo, str):
            return undo
        for obj in (self.sm, self.sm.model, *self.sm.active_listeners):
            candidate = getattr(obj, undo, None)
            if callable(candidate):
                return candidate
        raise UnsupportedCompensableAction(
            _("Compensation function '{}' was not found.").format(undo)
        )

    def track_sync(self, wrapper: Any, args: tuple, kwargs: dict) -> Any:
        """Invoke a sync action wrapper and, on success, record its compensation."""
        result = wrapper.call(*args, **kwargs)
        self._record(wrapper, args, kwargs, result)
        return result

    async def track_async(self, wrapper: Any, args: tuple, kwargs: dict) -> Any:
        """Invoke an async action wrapper and, on success, record its compensation."""
        result = await wrapper(*args, **kwargs)
        self._record(wrapper, args, kwargs, result)
        return result

    def _record(self, wrapper: Any, args: tuple, kwargs: dict, result: Any) -> None:
        # ``wrapper._callback`` is the signature-adapter closure; compensation
        # metadata is copied onto it by ``dispatcher.callable_method``.
        meta = get_action_meta(wrapper._callback)
        if meta is None:
            raise UnsupportedCompensableAction(
                _(
                    "Action '{name}' in a compensable microstep must be declared "
                    "side-effect-free or registered with a compensation function."
                ).format(name=getattr(wrapper._callback, "__name__", wrapper._callback))
            )
        if meta.kind is ActionKind.SIDE_EFFECT_FREE:
            return
        assert meta.undo is not None
        self.resolve_undo(meta.undo)  # fail fast on unresolvable name references
        self.stack.append(
            CompensationEntry(
                label=meta.label,
                undo=meta.undo,
                credential=result,
                args=args,
                kwargs=kwargs,
            )
        )

    def run_compensations(self) -> list[BaseException]:
        """Run all recorded compensations in reverse order; collect their errors."""
        errors: list[BaseException] = []
        while self.stack:
            entry = self.stack.pop()
            try:
                self._invoke_undo(entry)
            except BaseException as error:  # noqa: BLE001 - keep undoing; report all errors
                errors.append(error)
        return errors

    async def run_compensations_async(self) -> list[BaseException]:
        """Async counterpart of :meth:`run_compensations`."""
        errors: list[BaseException] = []
        while self.stack:
            entry = self.stack.pop()
            try:
                result = self._invoke_undo(entry)
                if isawaitable(result):
                    await result
            except BaseException as error:  # noqa: BLE001 - keep undoing; report all errors
                errors.append(error)
        return errors

    def _invoke_undo(self, entry: CompensationEntry) -> Any:
        undo = self.resolve_undo(entry.undo)
        adapter = SignatureAdapter.from_callable(undo)
        params = adapter.parameters

        # Decide how the undo wants the credential before binding.
        positional_kinds = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        positional_params = [
            (name, param)
            for name, param in params.items()
            if param.kind in positional_kinds
        ]
        var_positional = next(
            (name for name, p in params.items() if p.kind is Parameter.VAR_POSITIONAL),
            None,
        )

        if "credential" in params:
            kwargs = {**entry.kwargs, "credential": entry.credential}
            bound = adapter.bind_expected(*entry.args, **kwargs)
            return undo(*bound.args, **bound.kwargs)

        # Otherwise the credential is positional with a user-chosen name (or
        # absorbed by *args). Bind the injected context first, then place the
        # credential in the next free positional slot.
        bound = adapter.bind_expected(*entry.args, **entry.kwargs)
        call_args = list(bound.args)
        call_kwargs = dict(bound.kwargs)
        if len(call_args) < len(positional_params):
            call_args.append(entry.credential)
        elif var_positional is not None:
            call_args.append(entry.credential)
        else:
            raise UnsupportedCompensableAction(
                _(
                    "Compensation function has no parameter to receive the action's "
                    "credential (declare a 'credential' parameter or use **kwargs)."
                )
            )
        return undo(*call_args, **call_kwargs)
