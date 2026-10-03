from collections.abc import MutableSet
from typing import TYPE_CHECKING

from .i18n import _

if TYPE_CHECKING:
    from .event import Event
    from .state import State


class StateMachineError(Exception):
    "Base exception for this project, all exceptions that can be raised inherit from this class."


class InvalidDefinition(StateMachineError):
    "The state machine has a definition error"


class InvalidStateValue(InvalidDefinition):
    "The current model state value is not mapped to a state definition."

    def __init__(self, value, msg=None):
        self.value = value
        if msg is None:
            msg = _("{!r} is not a valid state value.").format(value)
        super().__init__(msg)


class AttrNotFound(InvalidDefinition):
    "There's no method or property with the given name"


class TransitionNotAllowed(StateMachineError):
    "Raised when there's no transition that can run from the current :ref:`configuration`."

    def __init__(self, event: "Event | None", configuration: MutableSet["State"]):
        self.event = event
        self.configuration = configuration
        name = ", ".join(str(s.name) for s in configuration)
        msg = _("Can't {} when in {}.").format(
            self.event and self.event.name or "transition", name
        )
        super().__init__(msg)


class UnsupportedCompensableAction(InvalidDefinition):
    "A compensable microstep contains an action that cannot be safely compensated."


class CompensationError(StateMachineError):
    """A compensable microstep failed and one or more compensations also failed.

    The original failure is preserved as :attr:`original_error`, and every
    exception raised by compensation callbacks is collected in
    :attr:`compensation_errors` (in compensation order). Compensation of the
    remaining successful actions always runs to completion, so this single
    exception reports the full picture.
    """

    def __init__(self, original_error: Exception, compensation_errors: list[BaseException]):
        self.original_error = original_error
        self.compensation_errors = list(compensation_errors)
        detail = ", ".join(type(error).__name__ for error in compensation_errors)
        super().__init__(
            f"Microstep failed with {original_error!r}; compensation also failed "
            f"({len(compensation_errors)} error(s): {detail})."
        )
