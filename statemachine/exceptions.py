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


class SnapshotError(StateMachineError):
    "Base class for errors while exporting or restoring a :ref:`persistence` snapshot."


class SnapshotNotQuiescentError(SnapshotError):
    """A snapshot was requested while the state machine cannot be checkpointed.

    The machine is quiescent only when it is activated, no event is being
    processed, the external/internal event queues are empty, and there is no
    pending or active invocation.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class InvalidSnapshot(SnapshotError):
    "The snapshot payload is malformed or incompatible with the current definition."


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
