"""Compensatable microstep support (saga-style rollback).

When :attr:`StateChart.compensate_on_failure` is enabled, each microstep runs in
*compensatable mode*: actions are registered together with a compensation
function via :func:`compensatable`, and callbacks that have no side effects must
be explicitly declared with :func:`pure`. If any action of the microstep fails,
the compensations of the actions that completed successfully within that
microstep are executed in reverse order, and the active configuration and
history are restored to their pre-microstep values.

The compensation receives the *credential* returned by the original action, so
it never has to guess which external side effects were performed:

>>> from statemachine.compensation import compensatable, pure

>>> log = []
>>> def reserve():
...     log.append("reserve")
...     return "reservation-42"  # credential handed to the compensation
>>> def release(credential):
...     log.append(f"release:{credential}")

>>> compensatable(reserve, release)
Compensatable(action=<function reserve ...>, compensation=<function release ...>)

"""

import contextvars
from collections.abc import Callable
from inspect import isawaitable
from typing import Any

from .exceptions import StateMachineError
from .i18n import _
from .utils import run_async_from_sync

_active_scope: contextvars.ContextVar["CompensationScope | None"] = contextvars.ContextVar(
    "compensation_scope", default=None
)


class CompensationError(StateMachineError):
    """A compensated microstep failed and one or more compensations also failed.

    Preserves the original exception that aborted the microstep together with
    every exception raised by the compensation functions.
    """

    def __init__(self, original: BaseException, compensation_errors: list[Exception]):
        self.original = original
        """The exception that aborted the microstep."""
        self.compensation_errors = list(compensation_errors)
        """Every exception raised by the compensation functions, in execution order."""
        msg = _("Microstep failed with {!r} and {} compensation(s) also failed: {}").format(
            original,
            len(self.compensation_errors),
            ", ".join(repr(e) for e in self.compensation_errors),
        )
        super().__init__(msg)


class Compensatable:
    """Pairs an action with its compensation function.

    Args:
        action: The action callback (a callable or a name resolved on the
            machine/model/listeners, like any other action).
        compensation: The compensation callback (callable or name). It is
            invoked with the credential returned by ``action`` when a later
            action of the same microstep fails.
    """

    def __init__(self, action: Any, compensation: Any):
        self.action = action
        self.compensation = compensation

    def __repr__(self):
        return (
            f"{type(self).__name__}(action={self.action!r}, "
            f"compensation={self.compensation!r})"
        )


def compensatable(action: Any, compensation: Any) -> Compensatable:
    """Register ``action`` together with its ``compensation`` function.

    Can be used anywhere an action is accepted (``on=``, ``enter=``, ``exit=``).
    Only meaningful when :attr:`StateChart.compensate_on_failure` is enabled.
    """
    return Compensatable(action, compensation)


def pure(func: Callable) -> Callable:
    """Declare a callback as side-effect-free.

    In compensatable mode, user ``exit``/``on``/``enter`` callbacks must either
    be registered with :func:`compensatable` or marked with ``@pure``,
    otherwise the microstep is rejected before execution.
    """
    func.is_pure_action = True  # type: ignore[attr-defined]
    return func


class CompensationScope:
    """Collects the compensations of actions completed within one microstep.

    Each entry pairs a compensation callable with the credential returned by
    its action. Compensations run in reverse (LIFO) order, and a failing
    compensation does not prevent the remaining ones from running.
    """

    def __init__(self):
        self._stack: list[tuple[Callable, Any]] = []

    def push(self, compensation: Callable, credential: Any):
        self._stack.append((compensation, credential))

    def compensate(self) -> list[Exception]:
        """Run all compensations in reverse order, collecting every error."""
        errors = []
        while self._stack:
            compensation, credential = self._stack.pop()
            try:
                result = compensation(credential)
                if isawaitable(result):
                    result = run_async_from_sync(result)
                    if isawaitable(result):
                        # A loop is already running in this thread: a sync
                        # engine cannot drive the coroutine to completion.
                        result.close()
                        raise RuntimeError(
                            "Cannot run async compensation from a synchronous "
                            "engine while an event loop is already running. "
                            "Use the async engine or a synchronous compensation."
                        )
            except Exception as e:
                errors.append(e)
        return errors

    async def acompensate(self) -> list[Exception]:
        """Async variant of :meth:`compensate`, awaiting async compensations."""
        errors = []
        while self._stack:
            compensation, credential = self._stack.pop()
            try:
                result = compensation(credential)
                if isawaitable(result):
                    await result
            except Exception as e:
                errors.append(e)
        return errors


def activate_scope(scope: CompensationScope) -> contextvars.Token:
    """Make ``scope`` the active scope for the current context."""
    return _active_scope.set(scope)


def reset_scope(token: contextvars.Token):
    """Restore the scope that was active before :func:`activate_scope`."""
    _active_scope.reset(token)


def current_scope() -> "CompensationScope | None":
    """Return the scope active for the current context, if any."""
    return _active_scope.get()
