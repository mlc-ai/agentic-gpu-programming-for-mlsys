# TIRx Compiler Foundation

An agent may want to move a wait, change a memory layout, or use a new GPU
instruction. To try these ideas, it needs to express these changes directly
in kernel code. The previous chapter introduced the compiler
foundation that supports this work. We now turn to TIRx-lite, the low-level
Python interface to TIRx used in this book.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/tirx-lite-layers.html" title="From TIRx-lite source to generated code" class="diagram-frame" height="430"></iframe>
  <figcaption>The source maps almost directly to the code that runs. Click a source line to see the code it becomes.</figcaption>
</figure>
```

This interface leaves the agent responsible for layouts, instruction choices,
and synchronization. The other harness components support that work: the
knowledge base supplies implementation examples, program analysis helps
investigate mistakes, and GPU measurements show whether a change improves
performance.

In this chapter, we will describe how an agent expresses an optimization in
kernel code, adds support for an instruction, and inspects the generated
program. Small examples introduce each capability; a GEMM kernel
at the end brings them together.

## Expressing Programs in TIRx-lite

To turn an optimization idea into a kernel, the agent needs to control both
the operations and the dependencies between them. We begin with a scheduling
change, then follow an instruction from its source call to generated code.

### Express the optimization in source

Overlapping a load with computation requires control over where the consumer
waits for its data. The agent can place the wait directly in kernel code. The
[TIRx architecture](https://tvm.apache.org/docs/tirx/overview.html#authoring-layers-and-ir)
represents expressions, control flow, buffers, and backend operations in a
shared object model. Higher-level tile operations can expand into lower-level
statements and calls within that model, so the program level can be raised
where a task benefits from it without losing access to the level below.

TIRx-lite's
{kernels}`authoring interface <tirx_lite/__init__.py>` exposes layout,
instruction, and pipeline choices in source. `txl.smem_pool()` allocates
shared-memory tiles with explicit swizzles, and view helpers compute addresses
and instruction descriptors. `txl.ptx` supplies explicit PTX calls for data
operations, with CUDA-level control flow. `txl.specialize()` assigns warp
roles, while explicit synchronization primitives let the agent place waits
and completion signals.

This control leaves the agent responsible for data readiness and buffer
lifetimes. A compiler-managed schedule can reduce that burden, but also
limits which scheduling changes the agent can express directly.

With the operand load already issued, for example, the agent can move
independent work before the wait:

```python
# Before
pipe.full.wait(stage, phase)
compute_independent_work()
consume_loaded_tile()

# After
compute_independent_work()
pipe.full.wait(stage, phase)
consume_loaded_tile()
```

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/wait-placement.html" title="Move the wait, keep the dependency" class="diagram-frame" height="640"></iframe>
  <figcaption>Independent work must not access the incoming tile. The wait must observe the matching load's completion; the producer may reuse the buffer only after consumption finishes.</figcaption>
</figure>
```

### Add support for new instructions

An optimization may need a hardware instruction that is not yet available
through `txl.ptx`. TIRx supports extending the instruction interface by
registering operand types, effects, and PTX emission. Kernel code can then
call the registered instruction through `txl.ptx`; analysis support also
requires a model of its behavior where applicable.

Keeping extensions local reduces the amount of compiler code that must
change for a new instruction. Its meaning still needs to agree across code
generation and analysis: both must interpret the same operands and effects.

To see how a registered instruction is used, consider `add.f32`. The vector
addition kernel from the opening figure calls it between a pair of loads and
a store:

```python
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=4, grid=lambda p: (p["n"] + 127) // 128)
def vector_add(
    a: txl.gptr(txl.f32), b: txl.gptr(txl.f32),
    out: txl.gptr(txl.f32), n: txl.i32,
):
    i = txl.cta_id() * 128 + txl.thread_id()
    with txl.If(i < n), txl.Then():
        x = txl.local_scalar(txl.f32)
        y = txl.local_scalar(txl.f32)
        txl.ptx.ld.global_.f32(x, a.ptr_to([i]))
        txl.ptx.ld.global_.f32(y, b.ptr_to([i]))
        txl.ptx.add.f32(x, x, y)
        txl.ptx.st.global_.f32(out.ptr_to([i]), x)
```

The instruction calls take explicit operands, with destinations first.
`txl.If` and `ptr_to` construct the surrounding control flow and addressing.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/instruction-registration.html" title="add.f32: from registration to PTX" class="diagram-frame" height="380"></iframe>
  <figcaption>The <code>add.f32</code> registration specifies read/write access and PTX emission. The call <code>txl.ptx.add.f32(x, x, y)</code> supplies the operands; the written operand becomes an output of the generated instruction.</figcaption>
</figure>
```

### Inspect the generated program

To judge an optimization, the agent needs to connect its source edit to the
operations the GPU will execute. The explicit instruction calls lower to
inline PTX in generated CUDA. The downstream `ptxas` compiler handles
machine-level optimization. The kernel object exposes its intermediate
representation (IR), generated source, and compiled module:

```python
vector_add.func        # the TIRx IR used for compilation and analysis
vector_add.source()    # the generated CUDA source
vector_add.compile()   # a runnable module
```

In the generated source, each instruction call becomes one inline PTX
instruction, such as `add.f32 %0, %1, %2;`.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/inspect-levels.html" title="One kernel, three views" class="diagram-frame" height="380"></iframe>
  <figcaption>The same addition in source, in the IR that analyses consume, and in the generated CUDA.</figcaption>
</figure>
```

For example, a layout conversion is written directly as PTX operations:
shuffles, or shared-memory loads and stores with synchronization. The source
shows which threads exchange values and which addresses they use, so the
agent can edit those operations directly when it changes the producer or
consumer layout. This visibility requires more hardware knowledge from the
agent. In a higher-level interface, one layout conversion may stand for many
such operations; inspecting generated code is then needed to understand the
work behind that single source operation.

## Example: A GEMM Kernel

A complete kernel must combine instruction choices, memory layouts, and
synchronization into one program. We can see these choices together in the
following GEMM kernel, which computes a 128×128 tile of `D = A Bᵀ` with FP16
inputs and output on one CTA (thread block).
All threads copy `A` and `B` into swizzled shared memory; one thread issues
the Tensor Core MMAs into Tensor Memory and signals an mbarrier when they
complete; then each warp reads 32 rows of the result back, converts them to
FP16, and stores them.

```{code-block} python
:class: fold-code

import tirx_kernels.tirx_lite as txl

M, N, K = 128, 128, 64
MMA = "tcgen05.mma.cta_group::1.kind::f16"
IDESC = 0x08200010          # f16 x f16 -> f32, M=128, N=128, A and B K-major
TMEM_LD = "tcgen05.ld.sync.aligned.32x32b.x32.b32"


@txl.kernel(warps=4, grid=1)
def gemm(A: txl.gptr[txl.f16, (M, K)], B: txl.gptr[txl.f16, (N, K)],
         D: txl.gptr[txl.f16, (M, N)]):
    warp, lane, tid = txl.warp_id(), txl.lane_id(), txl.thread_id()

    # Shared memory: TMEM address slot, one mbarrier, swizzled A and B tiles.
    smem = txl.smem_pool()
    tmem_slot = smem.alloc((1,), "uint32")
    bar = smem.alloc((1,), "uint64", align=8)
    As = smem.alloc((M, K), "float16", swizzle=txl.SW128B)
    Bs = smem.alloc((N, K), "float16", swizzle=txl.SW128B)

    with txl.If(warp == 0), txl.Then():
        txl.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            tmem_slot.ptr_to([0]), txl.uint32(N))
    with txl.If(tid == 0), txl.Then():
        txl.ptx.mbarrier.init.shared.b64(bar.ptr_to([0]), txl.uint32(1))

    # Load: thread t copies row t of A and of B into shared memory, 16 bytes at a time.
    r = txl.alloc_local([4], "uint32")
    for j in range(K // 8):
        txl.ptx.ld.global_.v4.u32(r[0], r[1], r[2], r[3], A.ptr_to([tid, j * 8]))
        txl.ptx.st.shared.v4.u32(As.ptr_to(tid, j * 8), r[0], r[1], r[2], r[3])
        txl.ptx.ld.global_.v4.u32(r[0], r[1], r[2], r[3], B.ptr_to([tid, j * 8]))
        txl.ptx.st.shared.v4.u32(Bs.ptr_to(tid, j * 8), r[0], r[1], r[2], r[3])
    txl.ptx.fence.proxy.async_.shared__cta()
    txl.ptx.fence.mbarrier_init.release.cluster()
    txl.ptx.tcgen05.fence__before_thread_sync()
    txl.cuda.cta_sync()
    txl.ptx.tcgen05.fence__after_thread_sync()
    tmem = txl.local_scalar(txl.u32)
    txl.ptx.ld.shared.u32(tmem, tmem_slot.ptr_to([0]))

    # Compute: one thread issues the MMAs into Tensor Memory and signals the barrier.
    with txl.If(tid == 0), txl.Then():
        txl.idioms.mma_chain(MMA, tmem, a=As, b=Bs, idesc=IDESC,
                             pred=None, accumulate=False, guard="pred")
        txl.ptx["tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"](
            bar.ptr_to([0]))
    txl.cuda.mbarrier_wait(bar.ptr_to([0]), 0)
    txl.ptx.tcgen05.fence__after_thread_sync()

    # Write back: warp w reads rows 32w..32w+31 from Tensor Memory, converts, stores.
    acc = txl.alloc_local([32], "float32")
    out = txl.alloc_local([16], "uint32")
    row = warp * 32 + lane
    for c in range(N // 32):
        txl.ptx[TMEM_LD](*[acc[i] for i in range(32)],
                         tmem + (txl.cast(warp * 32, "uint32") << 16) + c * 32)
        txl.ptx.tcgen05.wait__ld.sync.aligned()
        for i in range(16):
            txl.ptx["cvt.rn.f16x2.f32"](out[i], acc[2 * i + 1], acc[2 * i])
            txl.ptx.st.global_.u32(D.ptr_to([row, c * 32 + 2 * i]), out[i])

    txl.ptx.tcgen05.fence__before_thread_sync()
    txl.cuda.cta_sync()
    with txl.If(warp == 0), txl.Then():
        txl.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
        txl.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem, txl.uint32(N))
```

Read the kernel in three stages. The load stage chooses the shared-memory
layout and copies the operands into it. The compute stage issues the matrix
multiply-accumulate operations and arranges a completion signal. The writeback
stage waits for that signal, reads the accumulator, and stores the result.
Allocation and release surround these stages to manage Tensor Memory's
lifetime.

These choices give the agent concrete places to edit a kernel: it can change
the layout, the instructions, or the schedule and inspect the resulting
program. Analysis can help diagnose errors in a candidate, but evaluating its
performance requires measurements on the target GPU. The
[next chapter](kcoral.md) describes how KCoral supplies that execution
environment.
