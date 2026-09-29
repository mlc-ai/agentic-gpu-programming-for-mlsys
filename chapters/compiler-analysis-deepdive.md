# Compiler Analysis Deep Dive

The last chapter showed how to carry out agentic GPU programming with the
TIRx Harness. In this chapter, we will take a deep dive and review how an
agent interacts with Synccheck and Racecheck over one specific run.

A hang or a race can survive many GPU runs before it shows up, so the agent
needs feedback that does not depend on timing. Synccheck and Racecheck supply
it: they analyze a kernel on the CPU, on concrete inputs, and report
synchronization and data-race findings against its source lines. Both tools
build on NumSim, the CPU simulation foundation that models the kernel's
operations and memory behavior.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/analysis-feedback.html" title="Program in, report out" class="diagram-frame" height="640"></iframe>
  <figcaption>Synccheck and Racecheck analyze kernels with concrete inputs and return reports that point to source lines. Pick a checker, then switch between the buggy and the fixed program.</figcaption>
</figure>
```

## Calling the Analyses

To check its own kernels as often as it edits them, the agent calls the
checkers directly from its scripts. Both checkers are Python functions in
`tirx_harness` that take a TIRx `PrimFunc` and concrete values for its arguments.
A TIRx-lite kernel exposes this representation as `kernel.func`. The
producer-overtaking kernel in the Synccheck figure has a single argument,
an output array of `ROUNDS` integers, and calls Synccheck like this:

```python
from tirx_harness import racecheck, synccheck

report = synccheck(kernel.func, inputs={"out": np.zeros(ROUNDS, np.int32)})
report.print()           # findings with source excerpts
report.verdict           # "clean", "review", "incomplete", or "error"
report.to_dict()         # structured evidence
report.require_clean()   # raise unless the verdict is "clean"
```

In the Kimi Delta Attention run, the agents met all four verdicts:

| Verdict | Examples from the run |
|---|---|
| `error` | `deadlock`, `mbarrier_arrive_before_consumption`, `setmaxnreg_incomplete_warpgroup`, `data_race` |
| `incomplete` | `missing_input_bindings`: the adapter did not bind every kernel argument |
| `review` | `uninitialized_read`: a TMEM read of bytes never written |
| `clean` | No finding for this kernel and these inputs |

For a real kernel, the agent writes a small adapter script that builds such
inputs. The script below, `check_sync.py`, is condensed from the one the agent
wrote for its first Kimi Delta Attention candidate. It loads the candidate, creates random
tensors at a tiny sequence length, builds the TMA descriptors, and runs either
checker:

```python
"""Pre-GPU Synccheck / Racecheck for the fused KDA kernel on a small launch."""
from tirx_harness import synccheck, racecheck
from tirx_harness.numsim import TensorMap

which = sys.argv[1] if len(sys.argv) > 1 else "sync"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 128
H = int(sys.argv[3]) if len(sys.argv) > 3 else 1

q = bf16_bits(rng.standard_normal((T, H, D)) * 0.5)
...
s0 = (rng.standard_normal((H, D, D)) * 0.25).astype(np.float32)
o = np.zeros((T, H, D), np.uint16)

def tmap(base):
    return TensorMap(
        base=base.reshape(-1), global_shape=(D, T, H), global_strides=(H * D * 2, D * 2),
        box_shape=(64, 64, 1), element_strides=(1, 1, 1), dtype="bfloat16", swizzle="128B",
    ).numpy()

inputs = dict(q=q.reshape(-1), ..., q_map=tmap(q), k_map=tmap(k), v_map=tmap(v), o_map=tmap(o),
              scale=float(1.0 / math.sqrt(D)))
kern = mod.build_kernel(T, H)
report = (synccheck if which == "sync" else racecheck)(kern.func, inputs)
print(f"{which} T={T} H={H} verdict={report.verdict} in {time.time()-t0:.1f}s")
report.print()
```

Each check took about half a minute on the CPU and needed no GPU, so the agent
ran both in the background on every change to synchronization, alongside its
GPU tests. Later, the agents copied the same script for each new candidate
and added `require_clean()`, so that a non-clean verdict failed the command.

The launch has to be long enough to exercise every stage of the pipeline,
including reuse of each ring slot. The agent recorded its rule in its notes
early in the run:

```text
T=320 (5 chunks) runs in ~30 s each and exercised ring wrap-around; both must be clean before benchmarking.
```

When a later candidate added arguments that the adapter did not bind, both
checkers returned `incomplete`, and the next agent concluded that "the
earlier H=96 checker attempts were incomplete because their adapters omitted
the new scratch arguments, so they are not evidence."

## Case Study: Feedback in the Kimi Delta Attention Run

The examples above showed how the agent prepares inputs, calls the checkers,
and reads their verdicts. We now follow what happens after a finding arrives:
the agent traces it back to the kernel, makes a repair, and runs the checkers
again. In this stretch of the Kimi Delta Attention run, each repair allows
the analysis to proceed further, until both checkers return clean reports.
The trace includes the agent's commands, the checkers' output, its notes,
and its edits. Output excerpts keep only the key lines; `...` marks omitted
text.

The kernel is the agent's first fused candidate. Preprocessing warps compute
gates and operand tiles, an MMA warp issues the matrix instructions,
state-keeper warps carry the recurrence, and a store warp writes the output;
they hand data to one another through barriers and ring buffers. The kernel
has passed a small GPU correctness test, and its official benchmark is
running on the server.

The agent starts both checkers through `check_sync.py` in the background, on a
launch of two chunks:

```bash
python check_sync.py sync 128 1
python check_sync.py race 128 1
```

Both checkers report the same finding:

```text
sync T=128 H=1 verdict=error in 24.7s
  [ERROR] mbarrier_prior_generation_consumption_not_happens_before: generation 0 consumption
          does not happen before mbarrier.arrive on ... generation 1
    Source: warp 8, source op #33346, loop iteration 1
       897 |                 m_o.arrive(0)
    Related: warp 14, source op #45975, loop iteration 0
      1055 |                 m_o.wait(0, st_o.phase)
race T=128 H=1 verdict=error in 30.3s
  [ERROR] mbarrier_prior_generation_consumption_not_happens_before: ...
```

> **Agent:** Both checkers confirm the O-staged barrier `m_o` lacks
> back-pressure, risking a parity miss if the state keeper outruns the store
> warp. I'll fix it by making it a 2-stage barrier indexed by the V/O slot,
> then re-run both checkers including a 5-chunk launch to test ring
> wrap-around.

It edits the kernel:

```diff
-        m_o = txl.MBarrier(smem, 1)
+        m_o = txl.MBarrier(smem, 2)  # one generation per V/O slot; p_v.empty back-pressures it
         m_o.init(128)
 ...
                 txl.ptx[FENCE_ASYNC]()
-                m_o.arrive(0)
+                m_o.arrive(st_v.stage)
                 st_v.advance()
 ...
-            st_o = txl.PipelineState(1, phase=0)
             st_vo = txl.PipelineState(2, phase=0)
 ...
-                m_o.wait(0, st_o.phase)
-                st_o.advance()
+                m_o.wait(st_vo.stage, st_vo.phase)
```

The agent reruns both checkers on five chunks, enough to wrap around every
ring:

```bash
python check_sync.py sync 320 1
python check_sync.py race 320 1
```

Synccheck is clean, but Racecheck reports a race between the MMA warp and a
preprocessing warp:

```text
sync T=320 H=1 verdict=clean in 25.1s
synccheck CLEAN - 0 finding(s); ... 81 verifier states / 1245 verifier transitions
race T=320 H=1 verdict=error in 31.7s
  [ERROR] data_race: unordered read/write on allocation#0:[68608, 68612) ...
          memory ordering failure: no async->generic proxy bridge from shared::cta to shared::cta
          orders the accesses
    Prior: warp 13, source op #37098, loop iteration 0: ... [tirx.ptx.tcgen05_mma_ts]
    Current: warp 2, source op #4287, loop iteration 2
       697 |                     txl.ptx.st.shared.b32(kev.ptr_to(t, col0), ow)
```

> **Agent:** Racecheck flags one remaining issue: no async-to-generic proxy
> bridge between the MMA's async-proxy read of the Kend slot and a later
> generic write, even though synccheck's mbarrier handshake is fine. I'll
> check the "proxy bridge" rule in the NumSim semantics docs and compare it
> against how the GDN kernel orders its analogous `s_ainv`
> generic-write-after-MMA-read path, along with the benchmark log.

It searches the checker's documentation for the rule and reads the section it
points to:

```text
- observed TCGEN commit/wait completion closes the tracked asynchronous work
  and publishes its ordinary completion clock, but does **not** publish an
  implicit generic-async proxy bridge;
```

> **Agent:** I'll add the required `fence.proxy.async` after the Kend slot and
> T/Bt stage waits to satisfy Racecheck's rule on reused shared slots, along
> with IKET range annotations for profiling right after the benchmark result.

It edits the kernel:

```diff
                 b_kend_empty.wait(st_kend.stage, st_kend.phase)
+                txl.ptx[FENCE_ASYNC]()  # STATE MMA (async proxy) read this Kend slot; order before generic overwrite
                 kv = s_k[st_k.stage]
 ...
                 p_prep.empty.wait(st_prep.stage, st_prep.phase)
+                txl.ptx[FENCE_ASYNC]()  # NV/QKV MMAs read this T/Bt stage; order before generic overwrite
                 b_beta_full.wait(st_tok.stage, st_tok.phase)
```

The agent reruns both checkers:

```bash
python check_sync.py race 320 1
python check_sync.py sync 320 1
```

Synccheck is still clean, and Racecheck reports a second race, this time
between a state-keeper warp and a preprocessing warp:

```text
sync T=320 H=1 verdict=clean in 25.3s
race T=320 H=1 verdict=error in 30.3s
  [ERROR] data_race: unordered read/write on allocation#0:[219648, 219652) ...
          execution ordering failure: no synchronization edge orders the actors
    Prior: warp 8, source op #22936, loop iteration 0
       859 |                         txl.ptx["ld.shared.v4.f32"](
    Current: warp 0, source op #3886, loop iteration 2
       661 |                     txl.ptx["st.shared.v2.f32"](txl.address_of(s_gam[pstage, col0]), gm0, gm1)
```

> **Agent:** Racecheck caught a real ordering bug: prep writes chunk c+2's
> decay vector into a ring slot before waiting for its release, letting a
> prep running two chunks ahead clobber data the state keeper still needs.
> I'll fix this by moving the stage-empty wait and proxy fence to the top of
> prep's chunk loop, then re-run both checkers.

It edits the kernel:

```diff
             with txl.serial(NC) as c:
                 pstage = st_prep.stage
+                # The prep stage (s_gam, s_t, s_bt, s_e) is free once QKV(c-2) committed.
+                p_prep.empty.wait(st_prep.stage, st_prep.phase)
+                txl.ptx[FENCE_ASYNC]()  # NV/QKV MMAs read this T/Bt stage; order before generic overwrite
 ...
-                p_prep.empty.wait(st_prep.stage, st_prep.phase)
-                txl.ptx[FENCE_ASYNC]()  # NV/QKV MMAs read this T/Bt stage; order before generic overwrite
                 b_beta_full.wait(st_tok.stage, st_tok.phase)
```

The agent reruns both checkers once more:

```bash
python check_sync.py race 320 1
python check_sync.py sync 320 1
```

Both are clean:

```text
race T=320 H=1 verdict=clean in 31.4s
=== racecheck CLEAN ===
race conflicts checked in global, shared, tmem
native phase kda_fwd: 514516 semantic memory access(es)
sync T=320 H=1 verdict=clean in 25.1s
synccheck CLEAN - 0 finding(s); ... 81 verifier states / 1245 verifier transitions
```

> **Agent:** Both checkers are now clean on the 5-chunk launch. The official
> benchmark of the first version just finished too — let me read it.

Four runs of each checker and three edits took under ten minutes, all on the
CPU, while the official benchmark of the first version was still running.
