# Structural analysis

As statecharts grow, looking at the outgoing transitions of a single `State` is no
longer enough to spot configurations the machine can never leave — especially with
parallel regions, where the reachable *configuration* space is a product of the
regions. The structural analysis API explores the configuration graph of a
statechart **without executing any user code** and reports:

- states that no configuration exploration can reach;
- non-final configurations with no successors (dead ends);
- cycles formed by eventless transitions, where the machine never goes idle.

Guards (`cond`/`unless`), validators, and actions are **never called**. Unguarded
transitions are treated as certain edges, while guarded transitions have an unknown
outcome and are reported conservatively. Every result is therefore labeled as
`Certainty.DEFINITE` (only unguarded edges involved) or `Certainty.POSSIBLE`
(depends on unknown guard outcomes or runtime-only data such as recorded history).

```{hint}
The analysis is a structural approximation, **not** a proof of business-level
deadlock: a "possible" dead configuration may never occur because the guards
involved always pass at runtime, and a "definitely reachable" configuration may
never be observed because the required events are never sent.
```

## Running the analysis

Call {func}`~statemachine.analysis.analyze_statechart` or the
`StateChart.analyze_structure()` classmethod:

```py
>>> from statemachine import State, StateChart

>>> class TrafficLight(StateChart):
...     green = State(initial=True)
...     yellow = State()
...     red = State()
...     cycle = green.to(yellow) | yellow.to(red) | red.to(green)

>>> report = TrafficLight.analyze_structure()
>>> report.complete
True
>>> report.ok  # no findings
True
>>> sorted(report.definitely_reachable_states)
['green', 'red', 'yellow']

```

The report also exposes the explored stable configurations, split by certainty:
`definitely_reachable_configurations` and `possibly_reachable_configurations`.

## Unreachable states

States that no explored configuration contains are reported as
`FindingKind.UNREACHABLE_STATE` findings:

```py
>>> class Disconnected(StateChart):
...     validate_disconnected_states = False
...     a = State(initial=True)
...     b = State()
...     orphan = State()
...     go = a.to(b) | b.to(a)
...     spin = orphan.to.itself()

>>> report = Disconnected.analyze_structure()
>>> for finding in report.unreachable_states:
...     print(finding.certainty.value, finding.states, finding.message)
definite ('orphan',) State 'orphan' is not reachable from the initial configuration.

```

## Dead configurations

A non-final configuration with no outgoing transitions at all is a definite dead
end. When only guarded transitions leave the configuration, it is reported as a
*possible* dead end — it is stuck only if every guard fails:

```py
>>> class Stuck(StateChart):
...     validate_trap_states = False
...     a = State(initial=True)
...     b = State()
...     go = a.to(b)

>>> report = Stuck.analyze_structure()
>>> for finding in report.dead_configurations:
...     print(finding.certainty.value, finding.configuration, finding.message)
definite ('b',) Configuration [b] is not final and has no outgoing transitions.

```

Each finding carries a witness `path` from the initial configuration. When the
exploration completed, the path is a shortest one (`path_is_shortest`):

```py
>>> finding = report.dead_configurations[0]
>>> finding.path_is_shortest
True
>>> [(step.kind, step.event, step.source, step.targets) for step in finding.path]
[('initial', None, '', ('a',)), ('event', 'go', 'a', ('b',))]

```

## Eventless cycles

Unguarded eventless transitions forming a loop mean the run-to-completion loop
never terminates — a definite finding. If the loop involves guarded eventless
transitions, it is reported as possible:

```py
>>> class Blinker(StateChart):
...     on = State(initial=True)
...     off = State()
...     on.to(off)
...     off.to(on)

>>> report = Blinker.analyze_structure()
>>> for finding in report.eventless_cycles:
...     print(finding.certainty.value, finding.states)
definite ('off', 'on')

```

## Guards are never executed

The analyzer treats guarded transitions as unknown edges. This machine has a
guard that raises if called — the analysis runs cleanly and reports the guarded
targets as only *possibly* reachable:

```py
>>> class Guarded(StateChart):
...     a = State(initial=True)
...     b = State()
...     c = State(final=True)
...     go = a.to(b, cond="exploding_guard")
...     finish = b.to(c, cond="exploding_guard")
...
...     def exploding_guard(self):
...         raise AssertionError("must not be executed by the analysis")

>>> report = Guarded.analyze_structure()
>>> sorted(report.definitely_reachable_states)
['a']
>>> sorted(report.possibly_reachable_states)
['a', 'b', 'c']
>>> [f.configuration for f in report.dead_configurations]
[('a',), ('b',)]

```

## Limiting the search space

The number of explored configurations is bounded by `max_configurations`
(default: 1000). When the limit is reached, the report is marked as incomplete
and contains the partial results known so far — including reachability claims
being downgraded to `Certainty.POSSIBLE`:

```py
>>> report = Disconnected.analyze_structure(max_configurations=1)
>>> report.complete
False
>>> report.configurations_explored <= 1
True
>>> report.unreachable_states[0].certainty.value
'possible'

```
