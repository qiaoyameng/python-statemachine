# Persistence: snapshots and restoration

When a workflow needs to survive a process restart, you can export an explicit
**snapshot** of a stable configuration and later restore a fresh state machine
instance from it. Snapshots are versioned, JSON-safe data — no functions,
locks, futures, or arbitrary pickled objects are serialized.

## Quiescent checkpoints only

A snapshot captures a **quiescent** configuration. The machine must be:

- activated (past its initial entry),
- idle — no transition or callback currently executing,
- drained — both the external and internal event queues empty,
- settled — no pending or active invocation.

Taking a snapshot in any other situation raises
`SnapshotNotQuiescentError`, whose message explains what is preventing the
checkpoint. Let in-flight events finish, cancel delayed events, or wait for
invoked work to complete first.

## Taking and restoring a snapshot

```{py}
>>> from statemachine import State, StateChart

>>> class Order(StateChart):
...     draft = State(initial=True)
...     review = State()
...     done = State(final=True)
...     submit = draft.to(review)
...     finish = review.to(done)

>>> order = Order()
>>> order.submit()

>>> snapshot = order.take_snapshot()
>>> snapshot.configuration
['review']

```

Restoring builds a new instance directly in the saved configuration. Entry and
exit handlers, as well as already-completed transition actions, are **not**
re-executed:

```{py}
>>> resumed = Order.restore_snapshot(snapshot)
>>> "review" in resumed.configuration_values
True
>>> resumed.finish()
>>> resumed.is_terminated
True

```

The snapshot is a plain data object with JSON helpers:

```{py}
>>> text = snapshot.to_json()
>>> restored = Order.restore_snapshot(text)
>>> list(restored.configuration_values)
['review']

```

## Persisting domain data

Only configuration is saved by default. Use `include_model_data` with an
explicit whitelist of model attributes whose values are JSON-native
(strings, numbers, booleans, `None`, lists/tuples and dicts of those):

```{py}
>>> class OrderModel:
...     def __init__(self):
...         self.state = None
...         self.amount = 100

>>> model = OrderModel()
>>> order = Order(model=model)
>>> snapshot = order.take_snapshot(include_model_data=["amount"])
>>> snapshot.model_data
{'amount': 100}

>>> new_model = OrderModel()
>>> resumed = Order.restore_snapshot(snapshot, model=new_model)
>>> resumed.model.amount
100

```

The managed state field itself cannot be exported through `include_model_data`
— it is always restored from the validated `configuration`.

## Definition fingerprint

Every snapshot records the machine id and a SHA-256 **fingerprint** of the
class definition (states, transitions, history nodes and the registered
callback contract). On restore, the snapshot is rejected if:

- the payload format version is unsupported,
- it is fed to a different state machine class,
- the definition changed since the snapshot was taken,
- a state id is unknown,
- a parallel region is incomplete,
- parent/child containment is violated,
- a shallow/deep history record points outside its owner.

All checks run **before** any model, listener, or machine instance is
constructed, so a corrupt or tampered snapshot fails before a single user
callback can run. Missing states are never guessed from initial transitions.

## What is captured

| Field | Contents |
| --- | --- |
| `version` | Snapshot payload format version |
| `machine_id` | The state machine class id |
| `fingerprint` | SHA-256 of the current definition |
| `configuration` | Active **leaf** state ids (ancestors are reconstructed) |
| `history` | Shallow/deep history records, by history node id |
| `model_data` | Caller-whitelisted JSON-native model attributes |
| `running` | `False` once a top-level final state was reached |
| `state_field` | Name of the model field holding the raw state value |
