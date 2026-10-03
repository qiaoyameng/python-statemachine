(structural-analysis)=

# Structural analysis

As statecharts grow with compound states, parallel regions and history,
inspecting the incoming transitions of a single state is no longer enough to
spot unreachable parallel configurations. {func}`statemachine.analyze_structure`
(and the {meth}`~statemachine.StateChart.analyze_structure` classmethod) builds
an abstract graph of *configurations* and reports structural issues **without
running the machine**.

## What is analyzed

The report contains up to three kinds of findings:

- **unreachable states** -- states that are never part of any explored
  configuration;
- **dead configurations** -- stable, non-final configurations with no
  possible outgoing transition;
- **eventless cycles** -- loops of automatic (eventless) transitions that can
  keep the processing loop busy without progress.

Every finding carries a witness {class}`~statemachine.analysis.Path`. Reachability
and cycle findings get a **shortest** path (in microsteps); cycle reports also
describe one repeating loop, which is explicitly flagged as *not* claimed to be
the shortest loop.

## Guards are never executed

The analysis is purely structural. The machine is **not instantiated** and no
guard, validator, action, entry or exit callback is ever called. Unguarded
edges are treated as definite. A guarded edge has an unknown condition, so it
only contributes a *possible* result:

- {data}`~statemachine.analysis.Certainty.DEFINITE` -- reachable through
  unconditional edges only;
- {data}`~statemachine.analysis.Certainty.POSSIBLE` -- reachable in at least
  one structurally possible execution.

```{warning}
A *possible* dead configuration or eventless cycle is an over-approximation,
**not** a proof of a business deadlock or a live-lock. A guard the analyzer
cannot evaluate may prevent that execution in practice. The report always
carries this caveat in its ``note``.
```

## Quick start

```py
>>> from statemachine import State, StateChart, analyze_structure

>>> class OrderFlow(StateChart):
...     draft = State(initial=True)
...     submitted = State()
...     done = State(final=True)
...
...     submit = draft.to(submitted)
...     approve = submitted.to(done)

>>> report = analyze_structure(OrderFlow)
>>> sorted(report.reachable_states)
['done', 'draft', 'submitted']
>>> sorted(report.definite_reachable_states)
['done', 'draft', 'submitted']
>>> report.findings
[]

```

A guarded transition keeps its target in the *possible* set only:

```py
>>> class MaybeFlow(StateChart):
...     start = State(initial=True)
...     target = State(final=True)
...
...     jump = start.to(target, cond="unknown_condition")
...
...     def unknown_condition(self):
...         raise AssertionError("never called by the analyzer")

>>> report = analyze_structure(MaybeFlow)
>>> sorted(report.reachable_states)
['start', 'target']
>>> report.definite_reachable_states
{'start'}

```

Structural traps are reported with a shortest witness path:

```py
>>> class TrappedFlow(StateChart):
...     validate_trap_states = False
...     validate_final_reachability = False
...     entry = State(initial=True)
...     stuck = State()
...     finish = State(final=True)
...
...     move = entry.to(stuck)
...     leave = entry.to(finish)

>>> report = analyze_structure(TrappedFlow)
>>> dead = report.dead_configurations[0]
>>> dead.state_ids
('stuck',)
>>> str(dead.path)
'move[entry -> stuck]'

```

## Bound on the search

Compound entry, parallel regions and history can produce exponentially many
configurations. The search is bounded by ``max_configurations`` (default
``10000``). When the bound is reached the report is marked ``truncated`` and
contains everything discovered so far -- truncation never introduces false
positives, it only makes the report potentially incomplete:

```py
>>> report = analyze_structure(OrderFlow, max_configurations=1)
>>> report.truncated
True

```

## API

See {class}`~statemachine.analysis.StructureReport`,
{class}`~statemachine.analysis.Finding` and
{class}`~statemachine.analysis.Path` for the full result model.
