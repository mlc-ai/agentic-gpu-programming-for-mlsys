# Knowledge Base for Agents

Optimizing a GPU kernel draws on implementation experience and knowledge
of the target hardware. This knowledge is often spread across code
repositories and hardware manuals, so an agent may spend substantial effort
finding relevant material before it can try an optimization.

A knowledge base organizes reusable kernels and hardware documentation
for the agent to search and learn from. It provides established strategies
as starting points while helping the agent discover hardware capabilities
for new optimizations.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/knowledge-base.html" title="A knowledge base for an agent" class="diagram-frame" height="410"></iframe>
  <figcaption>Select a knowledge source to see how the agent uses it. The highlighted connections show the retrieval and return of knowledge.</figcaption>
</figure>
```

In this chapter, we will describe the contents of a knowledge base and how
agents use it to find and adapt optimization strategies.

(knowledge-base)=
## Kernel Zoo

An agent may know that vectorization, tiling, or software pipelining can improve
a GPU kernel, yet still need a concrete example of how to apply these techniques.
A kernel zoo supplies complete programs showing how an algorithm, a data
layout, and an execution schedule fit together. The agent can inspect these
implementations and adapt their strategies to its own kernel.

Reference kernels can come from two main sources:

- **Ports of established implementations.** Kernels from libraries such as
  FlashAttention, FlashInfer, and DeepGEMM give the agent concrete algorithms,
  layouts, and synchronization patterns as a starting point for porting.
  Diversity in algorithms, layouts, and schedules broadens the strategies
  available to the agent.
- **Kernels developed by agents.** Useful results of optimization runs can
  become reference kernels, with short notes on their key optimizations and
  where they apply, so later agents can reuse them. On new hardware with few
  kernels to port, agents can create initial implementations from operator
  definitions and hardware documentation. Once validated and benchmarked,
  these implementations become the zoo's first reference kernels for later
  agents to adapt and improve.

### Knowledge Reuse Across Kernels

A kernel zoo may not contain the operator being optimized, so the agent needs
to recognize useful mechanisms in implementations of other operators. The
two examples below show that a reference can contribute at different scales:
the first supplies an algebraic decomposition, the second a cache policy on
a single load instruction.

For an example at the algorithm level, Gated DeltaNet and Kimi Delta
Attention are two variants of linear attention. Both decay a recurrent state
and apply a delta-rule correction. Gated DeltaNet uses one decay factor per
token and head, and Kimi Delta Attention uses one per key channel. Their
[chunkwise formulations](https://arxiv.org/html/2510.26692v1#S3.SS1) produce
systems `M X = R` with different coefficients but the same
unit-lower-triangular structure, so an algorithm that inverts one applies to
the other.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/kernel-knowledge-transfer.html" title="Reusing a triangular inversion strategy across kernels" class="diagram-frame" height="490"></iframe>
  <figcaption>The two operators produce different coefficients but share a unit-lower-triangular system. This structure lets an agent transfer the block inversion strategy from Gated DeltaNet to Kimi Delta Attention.</figcaption>
</figure>
```

The {kernels}`Gated DeltaNet prefill kernel <flashinfer/gdn_prefill/gdn_prefill_sm100.py>`
inverts this system hierarchically, merging 8×8 diagonal inverses into 16×16,
32×32, and 64×64 blocks with warp-level MMA. These merge steps depend on the
triangular structure, so they also apply to the system in Kimi Delta
Attention. An agent can reuse the hierarchical inversion strategy while
adapting block sizes and operand types to its target kernel.

A reference can also contribute a single instruction choice, even when its
operator has different mathematics.
For example, the {kernels}`DeepGEMM TF32 prenorm GEMM port <deepgemm/tf32_hc_prenorm_gemm.py>`
passes different L2 cache policies to its TMA loads: `evict_first` for the
left matrix operand and `evict_last` for the right. These load sites show how
to give different data different retention priorities in L2. An agent can
adapt this mechanism to its own reuse pattern: if a kernel reads raw inputs
once but exchanges intermediate tiles with another kernel, it can use
`evict_first` for the raw inputs and favor retaining the intermediate tiles.
This leaves more L2 capacity for data that another kernel will read soon.

In these examples, an agent can learn how to break a matrix inversion into
smaller problems from Gated DeltaNet, and how to choose which data to retain in L2
from DeepGEMM, then adapt both ideas to its own kernel. A kernel zoo can
therefore provide useful optimizations even when the reference kernels
compute different operators.

## Hardware and ISA Documentation

A kernel zoo contains only techniques that someone has already implemented,
and on new hardware those are few. Hardware manuals and ISA documentation
describe what the hardware itself provides: its instructions, memory
hierarchy, cache controls, and synchronization primitives, along with the
conditions for using each one correctly. When a profile exposes a bottleneck,
the agent can search these documents for a mechanism that addresses it and
turn that mechanism into an optimization strategy. The knowledge base therefore
needs searchable documentation for the version that matches the target hardware.

For example, suppose an agent is optimizing two concurrent kernels that
exchange temporary tiles through global memory. It wants to reduce HBM
traffic: once the consumer has finished using a tile, its contents are no
longer needed, so writing them back wastes bandwidth. The eviction hints
borrowed from DeepGEMM influence retention priority, but dirty tiles are still
written back when evicted.

Searching the PTX ISA for a way to avoid that writeback, the agent can find
[`discard.global.L2`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard),
which allows tiles still in L2 to be dropped without writing them back,
avoiding that unnecessary traffic. The manual also specifies that discarded
data becomes indeterminate. The agent must therefore ensure that the consumer
has finished reading each tile before discarding it. The documentation supplies
both the mechanism and the condition for using it.
