# Agent Workflows

Optimizing a GPU kernel often takes many experiments. After producing a
correct candidate, an agent may still need to investigate a bottleneck and
try another implementation. Keeping this search moving requires deciding
what to try next, carrying results into later turns, and choosing when to
stop.

An agent workflow coordinates this work across experiments and turns. It
gives the search continuity while agents use the compiler harness to
implement ideas and obtain feedback.

In this chapter, we will describe how different agent workflows organize a
kernel optimization search and compose with a compiler harness.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/agent-workflows.html" title="Agent workflows composed with a compiler harness" class="diagram-frame" height="400"></iframe>
  <figcaption>Select an example workflow to see how it organizes the search and uses the compiler harness.</figcaption>
</figure>
```

## Examples of Agent Workflows

The same compiler harness can support different workflows. The examples
below show how each organizes the search across experiments and agent turns.

### Fixed Pipeline

A workflow with predefined stages gives each part of the search a defined
role. For example, a fixed pipeline can repeat the sequence
research → write kernel → profile → propose direction.

The pipeline can assign different stages to different subagents, each with a
focused task. Results pass from one stage to the next, and the proposed
direction guides another iteration.

### Goal Mode

A workflow can also give an agent an objective and let it decide how to
proceed. For example,
[goal mode](https://learn.chatgpt.com/use-cases/follow-goals) keeps an
objective active across turns while the agent chooses what to do next.
There is no predefined division of roles: the same agent handles research,
implementation, and evaluation, choosing its next action as the task
progresses.

For kernel optimization, the goal can specify a target speedup together with
the required correctness checks.

### Flame Chase

An agent pursuing a goal may repeatedly overlook the same problems.
Alternating agents can give the search a fresh perspective, particularly
when the agents use different models with different blind spots.
[Flame Chase](https://docs.humanfia.ai/humanize/flows/flame-chase) follows this
approach by having two agents take turns on the same task.

Each turn starts with the task and the current repository in a fresh
session. Code, results, and notes saved in the repository carry the work
between agents. Each agent decides how to continue from what the previous
one left behind, with a shared budget bounding the run.

## Composing with a Compiler Harness

Agents connect a workflow to the compiler harness by calling its tools and
interpreting the results. For example, a profiling request can come from a
subagent assigned to a pipeline stage or from an agent pursuing a goal.
Both can call the same tool and receive the same kind of report. The
workflow determines which agent calls the tool and how work continues
afterward.

This shared interface lets different workflows reuse the same harness.
Changing how agents divide the work can leave their tool calls unchanged.
Conversely, if the profiling tool returns a more informative report through
the same interface, agents in each workflow can use that additional feedback.

[Running Agent Workflows](running-agent-workflows.md) puts this composition into practice
with TIRx Harness and shows how to use goal mode or Flame Chase
for a kernel optimization task.
