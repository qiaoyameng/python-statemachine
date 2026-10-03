(compensable-microsteps)=

# Compensable microsteps (saga)

Some transitions orchestrate external side effects in sequence — reserve
resources, deduct a quota, create a record. When a later step fails, the
earlier steps must be undone. **Compensable microsteps** turn the action
groups of a single {ref}`microstep <macrostep-microstep>` into an atomic,
rollback-capable transaction.

The mode is **opt-in** per state machine via the `compensable_microstep`
class flag. With it off (the default), exception semantics are exactly as
described in {ref}`error-handling` — nothing changes for existing machines.
In compensable mode the original failure always **propagates to the caller**
after compensation (it is not converted to an internal `error.execution`
event), because a rolled-back transaction must be reported rather than
silently redirected.

## Declaring actions

Every user action in the `before`, `exit`, `on`, and `enter` groups of a
compensable microstep must be declared as one of:

- **side-effect-free** with `@side_effect_free` — pure computations and other
  effects that need no undo;
- **compensable** with `@compensable(undo=...)` — an action paired with the
  function that undoes it.

The decorators attach metadata to the action function, so they compose with
the usual `@<event>.on`, `@state.enter`, and `@state.exit` decorators (put
`@compensable` / `@side_effect_free` closest to the function).

```py
>>> from statemachine import StateChart, State, compensable, side_effect_free

>>> class Wallet:
...     def __init__(self):
...         self.log = []
...     def reserve(self):
...         self.log.append("reserve")
...         return "reservation-1"
...     def release(self, credential):
...         self.log.append(("release", credential))

>>> wallet = Wallet()

```

### The compensation receives the original credential

The value returned by the action is the **credential** that identifies the
external effect (a reservation id, a deduction id). It is handed to the
compensation, so undoing never has to guess what the original action did.

The compensation may declare a `credential` keyword parameter or take it as a
positional parameter with any name (e.g. `reservation`). It also receives the
same {ref}`dependency-injection` arguments as the original action. `undo`
accepts either a callable or the name of a method on the state machine, its
model, or a listener.

```py
>>> class Saga(StateChart):
...     compensable_microstep = True
...
...     pending = State(initial=True)
...     done = State(final=True)
...     confirm = pending.to(done)
...
...     @compensable(undo="release")
...     def on_confirm(self):
...         return wallet.reserve()
...
...     def release(self, credential):
...         wallet.release(credential)

```

## What happens on failure

Only actions that **completed successfully** are pushed onto the compensation
stack. When any later action raises, compensations run in **reverse order**,
each receiving its credential; the active **configuration and history** are
restored to their pre-microstep state (internal events queued during the
microstep are discarded); and the original exception propagates.

```py
>>> class FailingSaga(Saga):
...     @side_effect_free
...     def on_enter_done(self):
...         raise RuntimeError("record creation failed")

>>> sm = FailingSaga()
>>> try:
...     sm.send("confirm")
... except RuntimeError as error:
...     print(error)
record creation failed

>>> sm.configuration_values == {"pending"}
True

>>> wallet.log
['reserve', ('release', 'reservation-1')]

```

The reservation was released using the original credential, and the machine
is back in `pending`, ready to accept events again.

### Compensation failures

A failing compensation does not stop the remaining compensations. The
original failure and **all** compensation failures are reported together via
`CompensationError` — see `error.original_error` and
`error.compensation_errors`.

### Cancellation

On the async engine, cancelling the task running a compensable microstep
(including during an entry action) triggers the same reverse compensation and
configuration rollback, then re-raises `asyncio.CancelledError`.

## Upfront rejection

Because external side effects cannot be guessed, an unsafe microstep is
refused **before any action runs** with `UnsupportedCompensableAction` when:

- a `before`/`exit`/`on`/`enter` action is neither side-effect-free nor
  compensable;
- the microstep enters or exits a state that declares an `invoke` handler
  (background work that cannot be automatically undone);
- a named compensation function cannot be resolved.

## Scope and limitations

- Guards (`cond`/`unless`), validators, and `prepare_event` must be pure —
  they run during transition selection, outside the transaction.
- The `after` group is a *finalize* hook that runs after the microstep's state
  changes; it is not part of the compensable transaction.
- `invoke` is not supported inside compensable microsteps; keep invoked
  states in separate, non-compensable transitions.
- The feature covers both the sync and async engines.
