# Benchmark Server Deep Dive

The last chapter took a deep dive into compiler analysis. In this
chapter, we will do the same for the KCoral benchmark server: how an agent
runs GPU experiments through it, and how the timing and profiling feedback it
returns led to new best kernels over the same run.

Whether a correct kernel is also fast, and where its time goes, can only be
measured on the GPU. In the Kimi Delta Attention run, every such experiment
went through the KCoral benchmark server.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/benchmark-feedback.html" title="One command, one remote experiment" class="diagram-frame" height="620"></iframe>
  <figcaption>Every GPU experiment in the Kimi Delta Attention run was one command on the development machine. Pick an experiment to see the command, the script that runs on the GPU host, and what comes back.</figcaption>
</figure>
```

## From an Agent Command to a Remote Experiment

Using KCoral's Python API for each check or profile involves constructing a
program, submitting it, and handling the returned results. Because coding
agents already use shell commands to work with files and invoke development
tools, TIRx Harness packages these steps into reusable command-line tools.

For example, an agent can invoke
{harness}`kcoral_python.py <evolution/remote/kcoral_python.py>`
for a Python experiment or {harness}`kcoral_ncu.py <evolution/remote/kcoral_ncu.py>`
for profiling, using familiar tool arguments together with the server address
and local input and output paths. The task adds its own command,
`kcoral_remote.py`, which submits a candidate to the official benchmark.

### Submit the experiment

In the recorded run, a new best kernel has just been admitted, and the agent
decides to profile it:

> **Agent:** I'm now taking an NCU snapshot of the new global best itself to identify
> whether its remaining limit is instruction issue, shared-memory/TMA traffic,
> or tensor-core dependency; that will steer the next structurally different
> experiment.

Its `capture_ncu.py` imports the candidate, constructs inputs, and launches
the kernel. The profiling command below omits surrounding shell bookkeeping.

```bash
python evolution/remote/kcoral_ncu.py --remote http://10.0.2.2:8901 \
    --send candidates/kda/forward_b1_t8192_h96/scratch/profile-overlap-diag-inverse \
    -o artifacts/overlap-diag-inverse-gram-tail/kda-full.ncu-rep \
    --set full --launch-count 1 --kernel-name kda_fwd_kernel -- python capture_ncu.py
```

`--remote` selects the service endpoint, and `--send` supplies the candidate
and capture script. `-o` names the local destination for the returned report.
NCU's capture options precede `--`; the application follows it. Here the
capture profiles one launch of the kernel named `kda_fwd_kernel`.

### Package and execute it

To execute this command remotely, the wrapper packages the input directory
into an archive and supplies a Python runner that unpacks the inputs and
invokes NCU. The following sketch shows how
{harness}`the wrapper's helper <evolution/remote/kcoral_tool.py>` builds the
program. `runner_source` is the helper's remote Python code, and
`input_archive` is the packaged input directory.

```python
from kcoral import Program

program = Program()
module = program.upload(kind="module", source=runner_source)
runner = program.get_function(module=module, name="ncu")
inputs = program.upload(kind="bytes", value=input_archive)
outcome = program.run(
    fn=runner, args=[inputs, ncu_args, application, environment_overrides],
)
program.return_(key="outcome", value=outcome)
```

The command supplies `ncu_args`, `application`, and `environment_overrides`.
After `Client.execute()` submits the program, the server schedules GPU access
and runs the uploaded code. NCU executes in the unpacked input directory,
and the runner returns its exit status and report bytes.

### Retrieve the result

Back on the development machine, the wrapper prints the captured output,
saves the report to `artifacts/overlap-diag-inverse-gram-tail/kda-full.ncu-rep`,
and reports the exit status. In the recorded run, the capture succeeds, and
the agent reads the returned file, and the one it captured for the previous
best, with local `ncu --import` to continue its analysis without another GPU
run:

```text
$ ncu --import artifacts/product-gamma/kda-full.ncu-rep --page details | rg -i 'Duration|...'
    Duration                         us       140.06
$ ncu --import artifacts/overlap-diag-inverse-gram-tail/kda-full.ncu-rep --page details | rg -i 'Duration|...'
    Duration                         us       135.33
```

> **Agent:** NCU confirms the new best reduces the profiled kernel from 140.06 to 135.33
> µs despite adding about 0.23 million executed warp instructions and ~11.7k
> shared-bank conflicts; the win is schedule latency, not instruction
> elimination.

GPU correctness tests and benchmark timings go through the same remote
execution path.

## Scripts the Agent Builds on the Commands

A single command answers one question, but each candidate needs several
answers in a fixed order: whether it is correct, where its time goes, and how
fast it runs. The agent also combines these commands into scripts of its own. Late in the
run, one agent sent all of its GPU work through `run_queue.sh`, a script it
wrote to run one GPU job at a time. For each candidate, `measure` runs a small
correctness test through `kcoral_python.py`, captures an IKET trace, and
submits the official benchmark through `kcoral_remote.py` only if the earlier
steps pass:

```bash
# Sequential GPU job queue for the kcoral server (never overlap GPU jobs).
REMOTE=http://10.0.2.2:8901
...
gputest() {  # $1 = scratch name; returns 0 if 12 clean regimes
  ...
  timeout 900 python evolution/remote/kcoral_python.py --remote $REMOTE --send artifacts/$n/gputest -e CASES=$CASES -e H=8 --timeout 600 -- test_small.py > artifacts/$n/gputest_run1.log 2>&1
  local c; c=$(grep -c 'bad=0.00e+00' artifacts/$n/gputest_run1.log)
  ...
  [ "$c" = "12" ]
}

bench() {  # $1 = scratch name, $2 = log name
  ...
  timeout 1700 python evolution/remote/kcoral_remote.py candidates/kda/forward_b1_t8192_h96 scratch/$n --remote $REMOTE --timeout 1500 > artifacts/$n/$log 2>&1
  ...
}

measure() {  # full pipeline for a scratch candidate
  local n=$1
  if ! checks_clean $n; then stamp "== $n: CPU checks not clean, skipping"; return; fi
  stamp "== $n"
  if ! gputest $n; then stamp "== $n: GPU test not clean, skipping timing"; return; fi
  iket $n
  bench $n bench_run1.log
}
```

`stamp` prints the server's load before and after each step:

```text
[load=0] bench gram-swp-c -> bench_run2.log
[load=0] bench gram-swp-c exit 0
holdout correctness: PASS
  kernel:   0.453386 ms
  speedup:  2.3146x
```

## Case Study: Finding New Bottlenecks in the Kimi Delta Attention Kernel

The commands and scripts above showed how the agent sends an experiment to
KCoral and retrieves its results. We now follow how those results guide the
next optimization. In two episodes from the Kimi Delta Attention run, the
agent uses a profile to identify work it can overlap or data it can move more
efficiently, changes the kernel, and benchmarks the result. A second profile
helps it check whether the change had the intended effect.

The kernel is a single fused kernel with one CTA per head, in which eight
preprocessing warps prepare each chunk for the MMA and state warps. `$WORKLOAD` stands for the
task's workload directory. Output excerpts keep only the key lines; `...`
marks omitted text.

At one point, the best kernel runs at 0.497283 ms, 2.1162× over the baseline.
The agent captures an IKET timeline of it through KCoral and summarizes the time each
preprocessing warp spends in each stage:

```bash
python kcoral_iket.py --remote http://10.0.2.2:8901 --send scratch/product-gamma \
    --output-dir iket_out -- profile --postprocess json -- python capture_iket.py
python iket_analyze.py iket_out/iket_pid_0x8e.trace.json 13
```

```text
kernel span 139.8 us (ts units assumed ns)
-- warp 0 (prep)
   p-gram     mean    0.909 med    0.896 [3072]
   p-tsolve   mean    0.653 med    0.640 [3072]
...
-- warp 2 (prep)
   p-gram     mean    0.579 med    0.576 [3072]
   p-tsolve   mean    0.629 med    0.640 [3072]
```

Warps 2 and 3 finish their share of the Gram stage in 0.58 µs; warps 0 and 1
need 0.91 µs, because they also compute the last two off-diagonal pairs. Until
then, warps 2 and 3 wait.

> **Agent:** A cleaner overlap is available in the existing hierarchy: after
> the four diagonal Gram tiles are stored, warps 2–3 are otherwise idle while
> warps 0–1 compute the last two off-diagonal pairs. I'm testing an aggregate
> diagonal-ready signal so warps 2–3 can perform the eight 8×8 diagonal
> inversions during that tail, removing the same work from the subsequent
> T-solve without duplicating Gram operands.

The agent adds one barrier that the four diagonal producers arrive on, and
moves the diagonal inversions into the Gram tail:

```diff
         m_scaled.init(128)
+        m_diag = txl.MBarrier(smem, 1)
+        m_diag.init(4)
         m_state = txl.TCGen05Bar(smem, 1)
 ...
                 # ===== grams (mma.sync) -> normalized raw G / Bt
+                bstage = st_tok.stage
                 gram_pair(wp, pstage)
+                with txl.If(wp < 4), txl.Then():
+                    txl.cuda.warp_sync()
+                    m_diag.arrive(0, pred=elect())
                 with txl.If(wp < 2), txl.Then():
                     gram_pair(wp + 8, pstage)
+                with txl.If((wp >= 2) & (wp < 4)), txl.Then():
+                    # All four diagonal Gram tiles are ready.  Invert their
+                    # eight 8x8 blocks while warps 0-1 finish pairs 8-9.
+                    m_diag.wait(0, st_diag.phase)
+                    p_beta.full.wait(st_tok.stage, st_tok.phase)
+                    av = s_t[pstage]
+                    invert_diag_8x8(av, ((((wp - 2) & 1) * 32 + lane) >> 3) * 8, bstage)
                 bar_prep()
+                st_diag.advance()
 ...
                         av = s_t[pstage]
-                        with txl.If(wp < 2), txl.Then():
-                            invert_diag_8x8(av, (((wp & 1) * 32 + lane) >> 3) * 8, bstage)
-                        bar_inv()
                         inverse_8_to_16(av, wp * 16)
```

The CPU checks pass, and the official benchmark measures a new best:

```bash
python kcoral_remote.py $WORKLOAD scratch/overlap-diag-inverse-gram-tail --remote http://10.0.2.2:8901 --timeout 600
```

```text
holdout correctness: PASS
...
  baseline: 1.049809 ms
  kernel:   0.487652 ms
  speedup:  2.1528x
```

> **Agent:** IKET confirms the mechanism rather than a timing fluke: the
> 2,048-token capture shrank the worst CTA span from 139.7 to 134.1 µs and the
> steady-state chunk period from 4.139 to 3.972 µs. The critical T-solve range
> fell from about 0.653 to 0.464 µs on the heavy prep warps; I'm now doing the
> required second full official run.

IKET shows where time goes inside the kernel, but not what the hardware is
waiting for. Later in the run, when the best kernel runs at about 0.482 ms,
the agent profiles one launch of it with NCU through KCoral and reads the
report's source counters locally:

```bash
python kcoral_ncu.py --remote http://10.0.2.2:8901 --send scratch/profile-qacc-ring \
    -o kda-full.ncu-rep --set full --launch-count 1 --kernel-name kda_fwd_kernel -- python capture_ncu.py
ncu --import kda-full.ncu-rep --page details --section SourceCounters
```

```text
    OPT   Est. Speedup: 9.055%
          This workload has uncoalesced global accesses resulting in a total of 190464 excessive sectors (10% of the
          total 1837824 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source
          locations.
```

> **Agent:** A more concrete target emerged from NCU: 190,464 excessive global
> sectors, almost exactly the strided `beta[t,h]` traffic. Each head CTA loads
> one bf16 every 192 bytes, while neighboring head CTAs need the same cache
> lines.

> **Agent:** I'm returning to the fastest family with a different mechanism:
> replace the strided per-head beta loads (the exact excessive-sector pattern
> identified by NCU) with a 128-byte TMA gather into shared memory, overlapped
> with the existing Q/K/V TMA issues. The sigmoid remains in-kernel and timed;
> this only changes how raw beta reaches the warp.

Because a TMA transfer must move at least 16 bytes in its innermost
dimension, each CTA gathers an aligned block of eight heads by 64 tokens and
picks out its own head in shared memory:

```diff
+            def load_beta_raw(cc):
+                p_beta_raw.empty.wait(st_brp.stage, st_brp.phase)
+                with txl.If(elected()), txl.Then():
+                    p_beta_raw.full.arrive(st_brp.stage, tx_count=C * 8 * 2)
+                    txl.ptx[TMA_G2S](
+                        s_beta_raw.ptr_to([0, 0, 0]),
+                        txl.address_of(beta_map),
+                        (head >> 3) << 3,
+                        txl.Cast("int32", cc * C),
+                        txl.int32(0),
+                        p_beta_raw.full.ptr_to([st_brp.stage]),
+                        txl.uint64(TMA_EVICT_FIRST),
+                    )
+                st_brp.advance()
 ...
+                load_beta_raw(c)
                 p_beta.empty.wait(st_bp.stage, st_bp.phase)
+                load_tile(p_k, st_kp, k_map, s_k, c)
+                load_tile(p_q, st_qp, q_map, s_q, c)
+                load_tile(p_v, st_vp, v_map, s_v, c)
+                p_beta_raw.full.wait(st_brc.stage, st_brc.phase)
                 for i in range(2):
-                    tok = txl.Cast("int64", c * C + lane + 32 * i)
-                    txl.ptx.ld.global_.nc.b16(hb[i], beta.ptr_to([tok * txl.int64(H) + head64]))
+                    txl.ptx.ld.shared.b16(
+                        hb[i], s_beta_raw.ptr_to([0, lane + 32 * i, head & 7])
+                    )
 ...
                 st_bp.advance()
+                txl.ptx[FENCE_ASYNC]()
+                p_beta_raw.empty.arrive(st_brc.stage)
+                st_brc.advance()
```

The CPU checks pass, and the official benchmark measures a new best:

```bash
python kcoral_remote.py $WORKLOAD scratch/beta-tma --remote http://10.0.2.2:8901 --timeout 600
```

```text
holdout correctness: PASS
...
  baseline: 1.050434 ms
  kernel:   0.477359 ms
  speedup:  2.2005x
```

A second NCU capture of the new kernel confirms that the fix removed the
bottleneck NCU had reported:

> **Agent:** Exact NCU confirms the mechanism: the target kernel fell from
> 190,464 to 6,144 excessive global sectors, while duration improved from
> 135.296 to 133.632 µs. Registers and occupancy are unchanged; the tradeoff
> is more shared-bank conflicts (656,485 → 756,377), but the net kernel time
> is lower.
