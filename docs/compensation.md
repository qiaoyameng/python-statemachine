(compensation)=
# Compensation

Some transitions perform a sequence of external side effects — reserve a
resource, debit a quota, create a record. When a later step fails, the earlier
ones must be undone. Setting `compensate_on_failure = True` enables the
**compensatable microstep** mode: each microstep becomes an atomic saga.

In this mode:

- Actions are registered together with a compensation function using
  {func}`compensatable <statemachine.compensation.compensatable>`. The
  compensation receives the *credential* returned by its action, so it never
  has to guess which side effects were performed.
- User `exit`/`on`/`enter` callbacks that have no side effects must be
  explicitly declared with {func}`@pure <statemachine.compensation.pure>`.
  States using `invoke` are not supported. Violations are rejected with
  `InvalidDefinition` **before** anything executes.
- On failure, only the actions that **completed successfully** within the
  microstep are compensated, in reverse order, and the active configuration
  and history are restored to their pre-microstep values.
- A failing compensation does not stop the remaining ones. The original
  exception and every compensation exception are preserved in a
  {class}`CompensationError <statemachine.compensation.CompensationError>`.
- Cancellation (e.g. `asyncio.CancelledError`) also triggers compensation
  before propagating.

```py
>>> from statemachine import State, StateChart
>>> from statemachine.compensation import compensatable, pure

>>> class OrderSaga(StateChart):
...     compensate_on_failure = True
...
...     idle = State(initial=True)
...     booked = State(final=True)
...     failed = State(final=True)
...
...     book = idle.to(booked, on=[
...         compensatable("reserve_stock", "release_stock"),
...         compensatable("charge", "refund"),
...     ])
...     error_execution = idle.to(failed)
...
...     def __init__(self, *args, **kwargs):
...         self.log = []
...         self.fail_charge = False
...         super().__init__(*args, **kwargs)
...
...     def reserve_stock(self):
...         self.log.append("reserve_stock")
...         return "reservation-1"  # credential handed to the compensation
...
...     def release_stock(self, credential):
...         self.log.append(f"release_stock:{credential}")
...
...     def charge(self):
...         self.log.append("charge")
...         if self.fail_charge:
...             raise RuntimeError("payment declined")
...         return "receipt-1"
...
...     def refund(self, credential):
...         self.log.append(f"refund:{credential}")

When every action succeeds, no compensation runs:

>>> sm = OrderSaga()
>>> sm.send("book")
['reservation-1', 'receipt-1']
>>> sm.log
['reserve_stock', 'charge']
>>> sm.current_state_value
'booked'

When `charge` fails, `reserve_stock` — the only action that completed — is
compensated with the credential it returned, the configuration is restored,
and the failure follows the usual error semantics (here, `error.execution`
moves the machine to `failed`):

>>> sm = OrderSaga()
>>> sm.fail_charge = True
>>> sm.send("book")
>>> sm.log
['reserve_stock', 'charge', 'release_stock:reservation-1']
>>> sm.current_state_value
'failed'

```

```{note}
With `compensate_on_failure = False` (the default), `compensatable()`
registrations are inert and the engine keeps the existing exception
semantics described in {ref}`error-handling`.
```
