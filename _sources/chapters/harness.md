# Introduction to TIRx Harness

Part I established the motivation for an agentic compiler harness and its
components. TIRx Harness brings those components into a concrete system:
the TIRx compiler foundation, a knowledge base, domain-specific program
analysis, and benchmarking and profiling through KCoral.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/tirx-harness.html" title="Components of TIRx Harness" class="diagram-frame" height="430"></iframe>
  <figcaption>The compiler harness from Part I, with its concrete TIRx components. The knowledge base and program analysis sit above the compiler foundation; KCoral connects GPU evaluation and profiling to the target devices. Select a component to inspect its role.</figcaption>
</figure>
```

The figure keeps the component organization from Part I. The TIRx foundation
represents the kernels that the agent writes and the analysis tools inspect.
The knowledge base supplies implementations and hardware references. Program
analysis explains kernel behavior, while KCoral provides GPU access for
correctness tests, timings, and profiles.

In this chapter, we will describe the responsibilities of each TIRx Harness
component and how the components work together.

## A Minimal, Predictable Compiler Foundation

The compiler foundation determines how directly an agent can connect what
it writes to what the GPU executes. An agent may know the execution strategy
it wants without knowing which source program will produce it. Compiler
transformations can also complicate debugging, since a failure may originate
in either the authored kernel or its compilation.

TIRx is a domain-specific language and compiler for machine learning kernels.
It serves as the compiler foundation for the harness. We write kernels through
TIRx-lite, a low-level Python interface to TIRx that exposes hardware
instructions, memory addresses, warp roles, and synchronization directly.
Keeping this interface close to NVIDIA's PTX instruction set reduces the additional
semantics an agent must learn; PTX documentation supplies the meaning and
requirements of the exposed instructions.

The foundation's intermediate representation (IR) retains loops, conditionals,
and buffers so that compilation and analysis can operate on the same structured
program. The hardware instruction interface also makes new instructions easier
to expose through small extensions. Support for an instruction includes both
its compiler lowering and, where available, a model that the analysis tools
can inspect.

## A Knowledge Base of Implementations and Hardware References

A compiler foundation makes hardware strategies expressible, but an agent
also needs implementation knowledge. TIRx Harness includes a knowledge base
of kernel implementations and hardware documentation. Its kernel zoo contains
concrete TIRx implementations that show how algorithms, layouts, and schedules
fit together, including ports from established GPU libraries:

| Kernel in the zoo | Ported from |
|---|---|
| {kernels}`FlashAttention-4 forward <flashattention/flash_attention4.py>` | FlashAttention |
| {kernels}`Dense FP8/FP4 matrix multiplication <deepgemm/fp8_gemm_1d1d.py>` | DeepGEMM |
| {kernels}`Gated DeltaNet prefill <flashinfer/gdn_prefill/gdn_prefill_sm100.py>` | FlashInfer |

Preserving complete implementations keeps their dataflow, synchronization,
and workload assumptions available for inspection. This context is easy to
lose in a short summary of optimization lessons. Kernels in the zoo can
therefore serve as references for techniques such as pipelined data movement
or specialized warp roles, even when an agent is implementing a different
computation.

TIRx Harness's knowledge base also includes hardware documentation to
complement the zoo. The PTX ISA describes instructions and their conditions
of use, including capabilities that may
not yet appear in the collection. The zoo provides known implementation
strategies; the ISA expands the set of hardware mechanisms available for
new ones.

## Domain-Specific Program Analysis

A kernel may pass many GPU tests and still contain a timing-dependent
synchronization bug or data race. TIRx Harness provides CPU-based simulation
and program analysis to investigate behavior beyond a pass/fail result:

| Tool | Purpose |
|---|---|
| NumSim | Simulate supported operations and expose outputs for comparison with an independent numerical reference. |
| Synccheck | Check synchronization behavior, including deadlocks and barrier protocol violations. |
| Racecheck | Detect conflicting memory accesses and missing ordering dependencies. |

These tools inspect the TIRx program with concrete inputs. Their conclusions
are scoped to the supplied kernel configuration, inputs, and supported
operations; reports identify unsupported behavior and coverage limits.
They complement correctness tests on the target GPU.

## Benchmarking and Profiling with KCoral

Correctness tests, timings, and profiles need a controlled execution
environment. If several agents share a GPU without coordination, a timing
change may reflect concurrent activity rather than an improvement in the
kernel. TIRx Harness uses KCoral, a remote GPU evaluation service, to
coordinate access to the target devices.

KCoral provides the execution environment for numerical checks, benchmarks,
and profiling captures, together with results and diagnostic artifacts.
For each kernel being optimized, an evaluation program specifies its input
cases, reference outputs, and timing rules. KCoral supplies the managed GPU
on which that program runs. This separates the meaning of a measurement
from the machinery that executes it.

For performance diagnosis, the harness integrates In-Kernel Event Tracing
(IKET) and NVIDIA Nsight Compute (NCU). IKET provides timelines of named
stages across warps and thread blocks, exposing waits and overlap. NCU
provides hardware measurements such as throughput, occupancy, memory traffic,
and stalls. These views connect performance problems to specific parts of
the kernel.

Separating development from evaluation also allows the target GPU to reside
on a server or edge device that does not host the coding agent. The same
harness can provide programming and analysis tools on a workstation while
using KCoral for measurements on the target hardware.

## Continuing Through Part II

The following chapters develop the programming and execution interfaces of
this harness:

- [TIRx Compiler Foundation](tirx-compiler-foundation.md) shows how an agent
  expresses optimization choices in kernel code and inspects the generated
  program, and explains how the instruction interface can be extended.
- [KCoral Benchmark Server](kcoral.md) explains how GPU experiments are
  submitted and executed, how the service coordinates access and isolates
  experiments, and how it returns benchmark and profiling results.

Part III then uses these components together in a Kimi Delta Attention
optimization task, following the agent from implementation through analysis
and GPU evaluation.
