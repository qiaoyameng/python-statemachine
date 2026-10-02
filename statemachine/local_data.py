from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from typing import TYPE_CHECKING
from typing import Any

from .exceptions import InvalidDefinition
from .exceptions import StateMachineError
from .i18n import _

if TYPE_CHECKING:
    from .state import State

LocalDataValueFactory = Callable[[], Any]
LocalDataFactory = Callable[[], MutableMapping[str, Any]]


class ReleasedLocalDataError(StateMachineError):
    """Raised when callback code writes through a released local-data frame."""


def normalize_local_data(
    local_data: "LocalDataValueFactory | Mapping[str, LocalDataValueFactory] | None",
) -> "LocalDataValueFactory | None":
    if local_data is None:
        return None

    if callable(local_data):
        return local_data

    if isinstance(local_data, Mapping):
        for value in local_data.values():
            if not callable(value):
                raise InvalidDefinition(
                    _("Each value in 'local_data' must be a zero-argument factory.")
                )

        def factory():
            return {key: factory_() for key, factory_ in local_data.items()}

        return factory

    raise InvalidDefinition(
        _("'local_data' must be a zero-argument factory or a mapping of factories.")
    )


def state_definition(state: "State") -> "State":
    return getattr(state, "_state", state)


class LocalDataFrame(MutableMapping[str, Any]):
    """Read-through view of one active state's local-data layer."""

    def __init__(
        self,
        state: "State",
        values: MutableMapping[str, Any],
        frames: "dict[State, LocalDataFrame]",
        released_frames: "dict[State, LocalDataFrame] | None" = None,
    ):
        self._state = state_definition(state)
        self._values = values
        self._frames = frames
        self._released_frames = released_frames or {}
        self._released = False

    def _is_active(self) -> bool:
        return self._frames.get(self._state) is self

    def _require_active(self):
        if not self._is_active():
            raise ReleasedLocalDataError(
                _("Cannot write to released state local data frame.")
            )

    def _frames_for_read(self) -> "list[LocalDataFrame]":
        frames = [self]
        state = self._state.parent
        while state is not None:
            frame = (self._released_frames if self._released else self._frames).get(state)
            if frame is not None:
                frames.append(frame)
            state = state.parent
        return frames

    def __getitem__(self, key: str) -> Any:
        for frame in self._frames_for_read():
            value = frame._values
            if key in value:
                return value[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._require_active()
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        self._require_active()
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        keys: dict[str, None] = {}
        for frame in self._frames_for_read():
            for key in frame._values:
                keys[key] = None
        return iter(keys)

    def __len__(self) -> int:
        return len({key for frame in self._frames_for_read() for key in frame._values})

    def release(self) -> None:
        if self._released:
            return
        snapshot = {self._state: self}
        state = self._state.parent
        while state is not None:
            frame = self._frames.get(state)
            if frame is not None:
                snapshot[state] = LocalDataFrame(
                    state,
                    dict(frame._values),
                    self._frames,
                )
            state = state.parent
        for frame in snapshot.values():
            frame._released_frames = snapshot
            frame._released = True
        self._released = True


StateLocalData = LocalDataFrame
"""Public mapping type injected as the ``local_data`` callback parameter."""


class LocalDataManager:
    """Owns active state frames and creates context views for callbacks."""

    def __init__(self):
        self._frames: dict[State, LocalDataFrame] = {}
        self._released: dict[State, LocalDataFrame] = {}

    def build_frames(self, states: "list[State]") -> "dict[State, LocalDataFrame]":
        built: dict[State, LocalDataFrame] = {}
        for state in states:
            definition = state_definition(state)
            if definition in self._frames:
                continue
            factory = getattr(state, "local_data_factory", None) or dict
            try:
                values = factory()
            except Exception:
                for frame in built.values():
                    frame.release()
                raise
            if not isinstance(values, MutableMapping):
                for frame in built.values():
                    frame.release()
                raise InvalidDefinition(_("A 'local_data' factory must return a mapping."))
            built[definition] = LocalDataFrame(definition, values, self._frames)
        return built

    def activate(self, frames: "dict[State, LocalDataFrame]") -> None:
        for state in frames:
            self._released.pop(state, None)
        self._frames.update(frames)

    def release(self, states: "list[State]") -> None:
        for state in states:
            definition = state_definition(state)
            frame = self._frames.pop(definition, None)
            if frame is not None:
                self._released[definition] = frame
                frame.release()

    def clear_released(self) -> None:
        self._released.clear()

    def rollback(self, frames: "dict[State, LocalDataFrame]") -> None:
        for state, frame in frames.items():
            active = self._frames.pop(state, None)
            if active is frame:
                active.release()

    def context_for(self, state: "State") -> "LocalDataFrame | None":
        current = state_definition(state)
        while current is not None:
            frame = self._frames.get(current)
            if frame is not None:
                return frame
            frame = self._released.get(current)
            if frame is not None:
                return frame
            current = current.parent
        return None
