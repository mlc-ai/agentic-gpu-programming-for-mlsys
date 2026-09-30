# Agentic GPU Programming for MLSys

Machine learning systems depend on fast GPU kernels for training and serving.
Attention, matrix multiplication, and fused operators account for substantial
work in these systems. Improving their implementations can reduce the time
and resources needed to run a model.

Making these kernels fast requires in-depth knowledge of algorithms, GPU
hardware, and programming models. Coding agents can increasingly carry out
this work: finding relevant implementations, writing kernels, interpreting
diagnostics, and choosing what to try next. Using these capabilities
effectively requires a workflow that coordinates the work and an environment
that gives agents access to the knowledge and feedback they need.

This book introduces the main elements of agentic GPU programming. You will
learn how a compiler-driven harness equips an agent to express optimization
ideas, diagnose errors, measure performance, and retain the results worth
keeping. Together, these capabilities form an effective optimization flow.

The book develops a compiler foundation, domain-specific program analysis, a
knowledge base, and GPU benchmarking and profiling. It then shows how agent
workflows compose with this compiler harness to guide an optimization run.

The material grows out of real-world experience building and using agentic GPU
programming systems. It is a companion book to
[Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/),
and we plan to integrate it into the
[Machine Learning Systems](https://mlsyscourse.org/) course series at Carnegie
Mellon University.

This book is open source. Contributions, corrections, and examples are welcome
through the [GitHub repository](https://github.com/mlc-ai/agentic-gpu-programming-for-mlsys).

## How This Book Is Organized

- **Part I, Elements of Agentic GPU Programming.** This part introduces the
  fundamental elements of agentic GPU programming. It begins with an overview
  of how compiler infrastructure and agent workflows fit together in an
  iterative kernel-development process. The following chapters develop the
  core elements—domain-specific program analysis, reusable knowledge,
  benchmarking, and profiling—through examples and interactive diagrams. The
  part concludes by comparing different agent workflows and showing how these
  elements can be composed into effective GPU optimization strategies.
- **Part II, TIRx Harness Overview.** This part introduces TIRx, one concrete
  instance of a compiler harness and the environment used for the examples in
  the rest of the book. It walks through the TIRx compiler foundation and the
  KCoral benchmark server with its execution interface.
- **Part III, Agentic GPU Programming in Action.** This part is a hands-on
  tutorial that puts the preceding chapters to work and carries agentic GPU
  programming out end to end. It looks closely at the feedback the harness
  returns over the course of a run and at the interaction patterns between the
  agent and each element of the harness, and it closes with a few advanced
  tips. Grouped GEMM introduces the workflow; recorded Kimi Delta Attention
  runs supply the diagnostic and review examples.

```{toctree}
:caption: Part I, Elements of Agentic GPU Programming
:maxdepth: 1

chapters/overview
chapters/analysis
chapters/knowledge
chapters/benchmarking
chapters/agent-workflows
```

```{toctree}
:caption: Part II, TIRx Harness Overview
:maxdepth: 1

chapters/harness
chapters/tirx-compiler-foundation
chapters/kcoral
```

```{toctree}
:caption: Part III, Agentic GPU Programming in Action
:maxdepth: 1

chapters/launching-the-agent
chapters/compiler-analysis-deepdive
chapters/benchmark-server-deepdive
chapters/self-improvement
chapters/advanced-tips
```
