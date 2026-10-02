# Snapshots (export and restore)

A workflow sometimes needs to continue on a different process: a restart, a
deployment, a worker taking over. python-statemachine supports this through an
**explicit snapshot API** that exports the current configuration as a plain
JSON document and restores it later without replaying any side effects.

A snapshot can only be taken at a **stable configuration** (a quiescent
checkpoint): no callback or transition is running, both event queues are
empty, and no invoked session is active. Attempting to take a snapshot in any
other state raises `SnapshotNotStable` explaining the reason.

## What a snapshot contains

```py
>>> from statemachine import State, StateMachine
>>> class Order(StateMachine):
...     draft = State(initial=True)
...     review = State()
...     done = State(final=True)
...     submit = draft.to(review)
...     approve = review.to(done)
>>> sm = Order()
>>> sm.submit()
>>> snapshot = sm.snapshot()
>>> sorted(snapshot)
['configuration', 'fingerprint', 'history', 'machine', 'model', 'state_field', 'terminated', 'version']
>>> snapshot["configuration"]
['review']
>>> snapshot["terminated"]
False

```

The document contains:

- `version` — snapshot format version;
- `machine` — the state machine class name;
- `fingerprint` — a hash of the statechart definition (states, hierarchy,
  transitions and declared callback names). Restore refuses a snapshot whose
  fingerprint does not match the current class, so a definition change can
  never be silently applied to an old workflow;
- `state_field` — the model attribute used to persist the active state values;
- `terminated` — whether a top-level final configuration was reached;
- `configuration` — ids of all active states (leaves **and** their compound
  ancestors, including parallel regions);
- `history` — remembered entries for every shallow/deep history pseudo-state;
- `model` — optional, JSON-native model data from an explicit allowlist.

Functions, bound methods, locks, futures, threads or arbitrary pickled
objects are **never** serialized.

## Including model data

Pass an allowlist of model attributes with `model_fields`; values must be
JSON-native (`str`, numbers, booleans, `None`, lists, dicts with string keys).

```py
>>> class OrderModel:
...     def __init__(self):
...         self.state = None
...         self.order_id = "ORD-42"
>>> sm = Order(OrderModel())
>>> snapshot = sm.snapshot(model_fields=["order_id"])
>>> snapshot["model"]
{'order_id': 'ORD-42'}

```

Persist the document with the standard `json` module. Here we move the
machine to ``review`` and persist that configuration:

```py
>>> import json
>>> sm.submit()
>>> text = json.dumps(sm.snapshot(model_fields=["order_id"]))

```

## Restoring

Restore validates the version, the machine id, the fingerprint, every state
id, parent/child closure and parallel-region completeness **before** the
machine instance (and therefore any user callback) is created. Restore then
sets the configuration directly: no entry or exit actions run, and a missing
state is never guessed as an initial state.

```py
>>> restored = Order.restore(text, OrderModel())
>>> "review" in restored.configuration_values
True
>>> restored.model.order_id
'ORD-42'
>>> restored.approve()
>>> restored.is_terminated
True

```

Runtime listeners can be attached at restore time. Note that restore does not
execute any entry actions:

```py
>>> class Tracer:
...     def on_enter_state(self, state):
...         print("enter", state.id)
>>> sm2 = Order(OrderModel())
>>> again = Order.restore(sm2.snapshot(), OrderModel(), listeners=[Tracer()])

```

Restoring a tampered document fails before any callback runs:

```py
>>> bad = json.loads(text)
>>> bad["configuration"] = ["nope"]
>>> from statemachine import SnapshotDefinitionMismatch
>>> try:
...     Order.restore(bad, OrderModel())
... except SnapshotDefinitionMismatch:
...     print("rejected")
rejected

```

## Shallow and deep history

History pseudo-state records are part of the snapshot, so re-entering a
compound through history after a restart restores the same configuration the
machine remembered before it left.

## Exceptions

- `SnapshotNotStable` — a callback is running, an event queue is non-empty, or
  an invocation is still active;
- `SnapshotDefinitionMismatch` — invalid version, machine id, fingerprint or
  configuration/history content;
- `SnapshotError` — base class for snapshot problems (e.g. non-JSON model
  data).
