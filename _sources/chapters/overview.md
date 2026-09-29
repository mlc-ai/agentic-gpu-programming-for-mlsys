# Overview

Machine learning systems rely on GPU kernels for training and serving, so
making these kernels faster can reduce the time and resources needed to run
a model. Achieving these gains, however, requires an iterative engineering
process: exploring implementations, drawing on hardware knowledge and
existing code, and refining the result through debugging and measurement.
As coding agents become more capable, they can take on more of this work.

Yet much of an agent's effort can go into work around the optimization
itself: predicting how kernel code will be lowered to hardware behavior,
diagnosing timing-dependent failures or measurements distorted by concurrent
GPU activity, and finding an implementation or optimization strategy
relevant to the task. This reduces token efficiency: the agent spends its
budget resolving these uncertainties, leaving less room to explore
optimizations. Otherwise viable optimizations may never be reached within
a practical search budget.

These are not simply search problems that can be solved by asking the agent
to try more variants. They point to missing pieces in the development
environment: a predictable programming foundation, tools for reasoning
beyond a single execution, reusable implementation knowledge, and controlled
evaluation. Taken together, they amount to a need for a compiler-based
environment built for agents -- we will use the term **compiler harness** to
name it.

As agentic programming shifts more of the work from writing code to shaping
the environment in which agents operate, we expect this harness to become
increasingly important. This framing is inspired by concurrent efforts in
AI-oriented system design. [CAKE](https://arxiv.org/abs/2608.12629) makes
compiler–agent co-design explicit, including a self-evolving program
representation that supports verification and diagnostics.
[OpenAI's Jalapeño](https://x.com/cdleary/article/2094878051238887834)
reflects a related full-stack principle: designing the programming target
itself to be clear and predictable enough for AI to optimize effectively.

```{raw} html
<figure id="agentic-harness-diagram" class="interactive-figure">
  <iframe src="../_static/diagrams/agentic-harness.html" title="Components of an agentic compiler harness" class="diagram-frame" height="430"></iframe>
  <figcaption>Click a component to see how the agent uses it and what it returns.</figcaption>
</figure>
```

In this chapter, we will describe the components of agentic GPU programming
and how a compiler harness and agent workflows fit together.

## Components of Agentic GPU Programming

Agentic GPU programming brings together a compiler foundation,
domain-specific compiler analysis, a knowledge base, benchmarking and
profiling, and agent workflows. The first four form the compiler harness;
agent workflows coordinate how agents use it over an optimization run.

### Compiler Foundation

An agent needs a way to express and compile kernels with a clear connection
between source code and GPU operations. A compiler harness can build on a
minimal, stable intermediate representation (IR) that makes operations,
memory accesses, and synchronization explicit. Keeping the IR small gives
agents fewer concepts to reason about, while stable semantics preserve the
meaning of programs across compiler updates. The IR should also be extensible
to hardware primitives, so new instructions and synchronization mechanisms
can be added as hardware evolves.

### Domain-specific Compiler Analysis

A failed test may tell an agent that a kernel is wrong without explaining
which operation caused the failure. Some errors may appear only under
particular inputs or execution orders. Domain-specific compiler analysis uses
the program's structure and GPU execution semantics to reason about its
behavior beyond an individual test run. The resulting diagnostics should
explain what went wrong and identify the relevant program operations.

### Knowledge Base

Knowing that tiling or pipelining can improve performance does not tell an
agent how to implement it for a particular operator and GPU. A knowledge base
supplies complete kernels that show how algorithms, data layouts, and
schedules fit together. Hardware and instruction-set documentation explain
the operations available and their conditions of use. Together, these
sources let the agent find a concrete strategy and understand how to adapt it.

### Benchmarking and Profiling

A timing improvement may reflect a better kernel, but it may also reflect
interference from other GPU work or state left by an earlier experiment. A
reliable benchmark needs controlled execution conditions and consistent
rules for correctness checks and timing measurements so that candidates can
be compared. Profiling complements these measurements by showing where time
is spent and how hardware resources are used. This helps the agent identify
performance bottlenecks and decide which parts of the kernel to optimize.

### Agent Workflows

An optimization task often takes many experiments and agent turns. An
agent workflow coordinates this work: it determines how tasks are
assigned, how work continues across turns, and when a run stops. Within that
workflow, agents use the compiler harness to find relevant knowledge,
implement kernels, and obtain correctness and performance feedback.

The compiler harness can compose with different agent workflows. Each can
use the same knowledge, tools, and evaluation interfaces.

## Connecting the Components

The workflow coordinates the optimization process, and the compiler harness
supplies the capabilities agents use throughout it.

Within the workflow, an agent retrieves knowledge to form a strategy,
expresses it as a candidate kernel, and uses program analysis to investigate
defects. It also checks correctness and measures performance on the target
GPU, using profiling to understand the results. These interactions give the
agent the information needed to repair a candidate, investigate a bottleneck,
or try another approach. The workflow carries the task forward across
experiments and agent turns.
Across runs, validated kernels can enrich the knowledge base, while problems
exposed during optimization guide improvements to the analysis tools and harness.

The following chapters develop these elements in turn:

- [Domain-specific Compiler Analysis](analysis.md) shows how an analysis
  reasons about a kernel beyond a single execution and reports which
  operations are responsible for a defect.
- [Knowledge Base for Agents](knowledge.md) describes how reusable kernels and
  hardware documentation are organized so that an agent can find a strategy
  and adapt it to the operator and GPU at hand.
- [Benchmarking and Profiling](benchmarking.md) covers the controlled
  execution and measurement rules that make candidates comparable, and the
  profiling that shows where a kernel spends its time.
- [Agent Workflows](agent-workflows.md) explains how agent workflows compose
  with the compiler harness.

The compiler foundation makes a kernel analyzable, keeps curated kernels in
a form an agent can reuse, and compiles each candidate just in time. Part II
introduces one instance of such a foundation, which supplies the examples
used throughout the rest of the book.
