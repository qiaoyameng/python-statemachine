from __future__ import annotations

from inspect import iscoroutinefunction
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from .exceptions import InvalidDefinition
from .exceptions import StateMachineError
from .i18n import _

if TYPE_CHECKING:
    from .state import State


class StateLocalData:
    def __init__(self, **factories: Callable[[], Any]):
        for name, factory in factories.items():
            if not callable(factory):
                message = _("State local data {!r} must use a factory function.")
                raise InvalidDefinition(message.format(name))
        self.factories = dict(factories)

    def add(self, other: "StateLocalData") -> "StateLocalData":
        merged = StateLocalData(**self.factories)
        merged.factories.update(other.factories)
        return merged

    def has_async_factory(self) -> bool:
        return any(iscoroutinefunction(factory) for factory in self.factories.values())


class StateDataFrame:
    __slots__ = ("state", "values", "parent", "active")

    def __init__(
        self,
        state: State | None,
        values: dict[str, Any],
        parent: "StateDataFrame | None",
    ):
        self.state = state
        self.values = values
        self.parent = parent
        self.active = True

    def close(self) -> None:
        self.active = False

    def _lookup(self, key: str) -> Any:
        frame: StateDataFrame | None = self
        while frame is not None:
            if key in frame.values:
                return frame.values[key]
            frame = frame.parent
        raise KeyError(key)

    def __getitem__(self, key: str) -> Any:
        return self._lookup(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def __delitem__(self, key: str) -> None:
        if key not in self.values:
            raise KeyError(key)
        del self.values[key]

    def __contains__(self, key: object) -> bool:
        try:
            self._lookup(str(key))
        except KeyError:
            return False
        return True

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> set[str]:
        keys: set[str] = set()
        frame: StateDataFrame | None = self
        while frame is not None:
            keys.update(frame.values)
            frame = frame.parent
        return keys

    def items(self):
        return ((key, self[key]) for key in self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())


class StateDataContext:
    __slots__ = ("manager", "frame")

    def __init__(self, manager: "StateDataManager", frame: StateDataFrame):
        self.manager = manager
        self.frame = frame

    def _writable_frame(self) -> StateDataFrame:
        if not self.frame.active:
            raise StateMachineError(
                _("Cannot write to a state frame that is no longer active.")
            )
        return self.frame

    def __getitem__(self, key: str) -> Any:
        return self.frame[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._writable_frame().values[key] = value

    def __delitem__(self, key: str) -> None:
        frame = self._writable_frame()
        if key not in frame.values:
            raise KeyError(key)
        del frame.values[key]

    def __contains__(self, key: object) -> bool:
        return key in self.frame

    def get(self, key: str, default: Any = None) -> Any:
        return self.frame.get(key, default)

    def keys(self) -> set[str]:
        return self.frame.keys()

    def items(self):
        return self.frame.items()

    def __iter__(self):
        return iter(self.frame)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def state(self) -> State | None:
        return self.frame.state

    @property
    def active(self) -> bool:
        if not self.frame.active:
            return False
        if self.frame.state is None:
            return True
        return self.manager.frame_for(self.frame.state) is self.frame


class StateDataManager:
    def __init__(self, machine: Any):
        self.machine = machine
        self.frames: dict[State, StateDataFrame] = {}
        self._contexts: dict[StateDataFrame, StateDataContext] = {}
        self._ephemeral: dict[State, StateDataContext] = {}
        self._transient_frames: list[StateDataFrame] = []

    def resolve_state(self, state: State | str | Any) -> State:
        if isinstance(state, str):
            state = self.machine.states_map[state]
        return getattr(state, "_state", state)

    def frame_for(self, state: State | str | None) -> StateDataFrame | None:
        if state is None:
            return None
        current: State | None = self.resolve_state(state)
        while current is not None:
            frame = self.frames.get(current)
            if frame is not None:
                return frame
            current = current.parent
        return None

    def frame_context(self, frame: StateDataFrame) -> StateDataContext:
        context = self._contexts.get(frame)
        if context is None:
            context = StateDataContext(self, frame)
            self._contexts[frame] = context
        return context

    def context_for(self, state: State | str | None) -> StateDataContext:
        resolved = self.resolve_state(state)
        frame = self.frame_for(resolved)
        if frame is not None:
            return self.frame_context(frame)
        context = self._ephemeral.get(resolved)
        if context is None:
            context = StateDataContext(self, StateDataFrame(resolved, {}, None))
            self._ephemeral[resolved] = context
        return context

    def begin_transient_frames(self) -> None:
        self._transient_frames = []

    def discard_transient_frames(self) -> None:
        for frame in self._transient_frames:
            self.frames.pop(frame.state, None)
            frame.close()
        self._transient_frames.clear()

    def commit_transient_frames(self) -> None:
        self._transient_frames.clear()

    def parent_frame(self, state: State) -> StateDataFrame | None:
        if state.parent is None:
            return None
        return self.frame_for(state.parent)

    def _store_frame(self, state: State, values: dict[str, Any]) -> StateDataFrame:
        frame = StateDataFrame(state, values, self.parent_frame(state))
        self.frames[state] = frame
        self._transient_frames.append(frame)
        return frame

    def build(self, state: State | str) -> StateDataFrame:
        state = self.resolve_state(state)
        existing = self.frames.get(state)
        if existing is not None:
            existing.close()
            self.frames.pop(state, None)
        values = {
            name: factory()
            for name, factory in dict(state.local_data.factories).items()
        }
        return self._store_frame(state, values)

    async def build_async(
        self,
        state: State | str,
        replace: bool = False,
    ) -> StateDataFrame:
        state = self.resolve_state(state)
        existing = self.frames.get(state)
        if existing is not None:
            if not replace:
                return existing
            existing.close()
            self.frames.pop(state, None)
        values: dict[str, Any] = {}
        for name, factory in dict(state.local_data.factories).items():
            value = factory()
            if iscoroutinefunction(factory):
                value = await value
            values[name] = value
        return self._store_frame(state, values)

    def release(self, state: State | str) -> None:
        canonical = self.resolve_state(state)
        frame = self.frames.pop(canonical, None)
        if frame is None:
            for key in list(self.frames):
                if key.id == canonical.id:
                    frame = self.frames.pop(key, None)
                    break
        if frame is not None:
            frame.close()

    def has_async_declarations(self) -> bool:
        return any(
            state.local_data.has_async_factory()
            for state in self.machine.states_map.values()
        )
