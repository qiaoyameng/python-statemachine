(states)=
(state)=

# States

```{seealso}
New to statecharts? See [](concepts.md) for an overview of how states,
transitions, events, and actions fit together.
```

A **state** represents a distinct mode or condition of the system at a given
point in time. States are the building blocks of a statechart — you define them
as class attributes, and the library handles initialization, validation, and
lifecycle management.

```py
>>> from statemachine import State, StateChart

>>> class TrafficLight(StateChart):
...     green = State(initial=True)
...     yellow = State()
...     red = State()
...
...     cycle = green.to(yellow) | yellow.to(red) | red.to(green)

>>> sm = TrafficLight()
>>> "green" in sm.configuration_values
True

```


## State parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | `""` | Human-readable display name. Defaults to the attribute name, capitalized. Accepts any object castable to `str` (e.g. a lazy translation proxy), resolved via `str()` at display time. |
| `value` | `None` | Custom value for this state, accessible via `configuration_values`. |
| `initial` | `False` | Marks this as the initial state. Exactly one per machine (or per compound). |
| `final` | `False` | Marks this as a final (accepting) state. No outgoing transitions allowed. |
| `enter` | `None` | Callback(s) to run when entering this state. See {ref}`state-actions`. |
| `exit` | `None` | Callback(s) to run when leaving this state. See {ref}`state-actions`. |
| `invoke` | `None` | Background work spawned on entry, cancelled on exit. See {ref}`invoke-actions`. |
| `local_data` | `None` | Zero-argument factory that creates data scoped to one state entry. |

```py
>>> class CampaignMachine(StateChart):
...     draft = State("Draft", value=1, initial=True)
...     producing = State("Being produced", value=2)
...     closed = State("Closed", value=3, final=True)
...
...     produce = draft.to(producing)
...     deliver = producing.to(closed)

>>> sm = CampaignMachine()
>>> sm.send("produce")
>>> list(sm.configuration_values)
[2]

```


## Initial state

A {ref}`StateChart` must have exactly one `initial` state. The initial state is
entered when the machine starts, and the corresponding {ref}`enter actions
<state-actions>` are called.


(final-state)=

## Final state

A **final** state signals that the machine has completed its work. No outgoing
transitions are allowed from a final state.

```py
>>> sm = CampaignMachine()
>>> sm.send("produce")
>>> sm.send("deliver")
>>> sm.is_terminated
True

```

You can query the list of all declared final states:

```py
>>> sm.final_states
[State('Closed', id='closed', value=3, initial=False, final=True, parallel=False)]

```

```{seealso}
See {ref}`validations` for the checks the library performs at class definition
time — including final state reachability, unreachable states, and trap states.
```


(state-local-data)=

## State-local data

State-local data belongs to one activation of a state. The library creates it
immediately before entry callbacks run and releases it when the state exits:

```py
>>> class Review(StateChart):
...     waiting = State(initial=True, local_data=lambda: {"notes": []})
...     approved = State(final=True, local_data=lambda: {"notes": []})
...     approve = waiting.to(approved)
...
...     def on_enter_waiting(self, local_data):
...         local_data["notes"].append("created for this review")

>>> sm = Review()
>>> sm.send("approve")

```

Declare a mapping of factories when a state needs several local values:

```py
>>> waiting = State(local_data={"items": list, "attempts": int})

```

Each machine and each re-entry calls the factory again. This avoids mutable
defaults shared by multiple machines or by previous activations.

Inside a child state, `local_data` reads through active ancestor states. A
write creates or replaces the key only in the current state's frame; it does
not mutate the parent:

```py
>>> class CompoundReview(StateChart):
...     class review(State.Compound, local_data=lambda: {"decision": "pending"}):
...         waiting = State(initial=True)
...         done = State(final=True)
...         decide = waiting.to(done)
...     idle = State(initial=True)
...     start = idle.to(review)
...
...     def on_enter_waiting(self, local_data):
...         assert local_data["decision"] == "pending"
...         local_data["decision"] = "approved"
...         assert local_data["decision"] == "approved"
...
...     def on_exit_review(self, local_data):
...         assert local_data["decision"] == "pending"

>>> sm = CompoundReview()
>>> sm.send("start")
>>> sm.send("decide")

```

Parallel regions each receive their own frame. A callback may retain a
{class}`~statemachine.StateLocalData` reference, but after exit that reference
is a read-only snapshot. Later writes raise a
{class}`~statemachine.exceptions.StateMachineError`. Internal transitions reuse
the current frame; external self-transitions and history re-entry create a new
one.


(compound-states)=

## Compound states

```{versionadded} 3.0.0
```

Compound states contain inner child states, enabling hierarchical state machines.
Define them using the `State.Compound` inner class syntax:

```py
>>> from statemachine import State, StateChart

>>> class Journey(StateChart):
...     class shire(State.Compound):
...         bag_end = State(initial=True)
...         green_dragon = State()
...         visit_pub = bag_end.to(green_dragon)
...     road = State(final=True)
...     depart = shire.to(road)

>>> sm = Journey()
>>> set(sm.configuration_values) == {"shire", "bag_end"}
True

```

Entering a compound activates both the parent and its `initial` child. You can query
whether a state is compound using the `is_compound` property.

```{seealso}
See {ref}`done-state-events` for completion events when a compound state's
final child is reached.
```


(parallel-states)=

## Parallel states

```{versionadded} 3.0.0
```

Parallel states activate all child regions simultaneously. Each region operates
independently. Define them using `State.Parallel`:

```py
>>> from statemachine import State, StateChart

>>> class WarOfTheRing(StateChart):
...     class war(State.Parallel):
...         class quest(State.Compound):
...             start = State(initial=True)
...             end = State(final=True)
...             go = start.to(end)
...         class battle(State.Compound):
...             fighting = State(initial=True)
...             won = State(final=True)
...             victory = fighting.to(won)

>>> sm = WarOfTheRing()
>>> "start" in sm.configuration_values and "fighting" in sm.configuration_values
True

```

```{seealso}
See {ref}`done-state-events` for how `done.state` events work with parallel
states (all regions must reach a final state).
```


(history-states)=

## History pseudo-states

```{versionadded} 3.0.0
```

A history pseudo-state records the active child of a compound state when it is exited.
Re-entering via the history state restores the previously active child. Import and use
`HistoryState` inside a `State.Compound`:

```py
>>> from statemachine import HistoryState, State, StateChart

>>> class WithHistory(StateChart):
...     class mode(State.Compound):
...         a = State(initial=True)
...         b = State()
...         h = HistoryState()
...         switch = a.to(b)
...     outside = State()
...     leave = mode.to(outside)
...     resume = outside.to(mode.h)

>>> sm = WithHistory()
>>> sm.send("switch")
>>> sm.send("leave")
>>> sm.send("resume")
>>> "b" in sm.configuration_values
True

```

Use `HistoryState(type="deep")` for deep history that remembers the exact leaf state
in nested compounds.


```{seealso}
See {ref}`querying-configuration` for how to inspect which states are currently
active at runtime.
```


(states from enum types)=

## States from Enum types

{ref}`States` can also be declared from standard `Enum` classes.

For this, use {ref}`States (class)` to convert your `Enum` type to a list of {ref}`State` objects.


```{eval-rst}
.. automethod:: statemachine.states.States.from_enum
  :noindex:
```

```{seealso}
See the example {ref}`sphx_glr_auto_examples_enum_campaign_machine.py`.
```
