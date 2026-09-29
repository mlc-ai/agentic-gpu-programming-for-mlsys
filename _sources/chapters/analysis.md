# Domain-specific Compiler Analysis

Suppose an agent encounters a flaky hang while optimizing a GPU kernel.
A timeout gives little indication of which operation is blocked or why, so
repeated runs can consume the agent's debugging budget without helping it
locate the cause. The agent needs feedback that points to the relevant parts
of the program and helps it understand the failure.

Domain-specific compiler analysis uses the compiler's representation of
operations, layouts, and synchronization to examine how a kernel computes
values, accesses memory, and coordinates execution. This provides feedback
that helps the agent investigate failures and assess correctness.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/analysis-overview.html" title="Domain-specific compiler analysis for an agent" class="diagram-frame" height="420"></iframe>
  <figcaption>Select an analysis to see its inputs and the diagnostic information it returns.</figcaption>
</figure>
```

In this chapter, we will describe how domain-specific compiler analysis
provides feedback that helps agents investigate failures and check correctness.

## Feedback Requirements

One optimization task repeats the find–implement–evaluate loop hundreds or
thousands of times, so correctness feedback for an agent must meet three
requirements:

- **Cheap**: it can run on every iteration, ideally without a GPU.
- **Localizing**: it identifies a mismatched output element or the source
  operations involved in a synchronization error or race.
- **Explicit about scope**: a report must state the assumptions and program
  behaviors it covers. Any check based on concrete execution can draw
  conclusions only for the inputs it tested. When a check cannot finish, it
  must report that it is incomplete rather than treat the result as a pass.

Static analyses reason from the program and its execution model; CPU
simulation complements them by following concrete inputs. Both can use the
compiler's representation of operations, layouts, and source locations to
return useful diagnostics.
The algorithms below illustrate possible implementations of these capabilities;
a compiler harness can provide the same kinds of feedback through other
analysis algorithms.

## Numerical Correctness Validation

Numerical validation normally means running the kernel on a GPU and comparing
its outputs with an independently written reference, but a GPU is not always
available. It may be busy with other work and require queuing. Locating a
single error often means inserting prints and rerunning many times, and if
every run waits for a GPU, progress slows dramatically.

The idea is to let the CPU execute the kernel in place of the GPU. The key is
**faithfulness**: the simulation must execute what the code actually does,
not what it is meant to do. Translating the kernel into equivalent array
operations would already assume the correct indices and layouts, and that is
exactly where errors hide. A faithful simulation needs to:

- **Follow the GPU execution model**, for example keeping a separate value
  for every lane and recording which lanes are active on each branch;
- **Respect instruction semantics**, including rounding, type
  conversion, and data exchange within a warp;
- **Compute addresses from the real layout**, converting logical indices to
  physical bytes, so that different variables pointing to the same storage
  also share it in the simulation.

The simulation can help an agent investigate some memory errors and numerical
mismatches. During execution, it can report out-of-bounds accesses and reads
of uninitialized memory. The agent can also compare the simulated outputs
with an independent reference without running the kernel on a GPU. This
comparison identifies mismatched elements by output name and logical index,
with their actual and expected values.

However, a simulation that executes warps in a fixed order may miss errors
that arise under other orders. Checking these requires examining both the
coordination between warps and the ordering of their memory accesses. The
following sections describe synchronization analysis and race detection for
these two purposes.

## Synchronization Protocol Verification

Warps that share data need rules for when each participant may read or
overwrite it. A **synchronization protocol** is the set of rules that
coordinates their work through signals and waits.

For example, a producer writes a tile and signals that it is ready. The
consumer waits for this signal before reading the tile, then signals that it
has finished. Before overwriting the buffer with the next tile, the producer
must wait for the consumer's completion signal. Together, these rules define
a protocol for safely passing data between the producer and consumer.

A kernel can implement these signals and waits with **`mbarrier`**, a GPU
synchronization primitive. In the protocol above, each `mbarrier` records
signals called arrivals and allows a wait to complete once the required
arrivals have occurred. One `mbarrier` can track tile readiness and another
buffer availability. Errors in these signals and waits can break the
protocol in several ways:

- **Arrival count mismatch**: the number of actual arrivals differs from the
  count set at initialization, so a wait never returns.
- **Circular waiting**: two groups of warps each wait for the other to finish
  first, and neither can proceed.
- **Producer overtaking**: the producer announces each new tile without
  waiting for the consumer to finish using the previous one. The buffer can
  be overwritten while the consumer is still using it; a protocol that
  reuses barrier phases can also lose track of which tile is ready.

These errors may occur only under some execution orders. Because the GPU does
not guarantee the relative speed of warps, rerunning the same kernel can
change that order and make the failure disappear. Numerical tests therefore
cannot reliably reproduce the bug, and manual review easily misses such
interactions in multistage pipelines. Synchronization analysis gives the
agent a way to find protocol errors by examining possible execution orders.

### One Possible Algorithm: State Search

Finding such errors requires examining execution orders beyond those
observed in a GPU test. One possible algorithm extracts a synchronization
model and searches its possible execution orders:

1. **Keep only the synchronization.** Record only where each participant is,
   which phase each barrier is in and how many arrivals it still needs, and
   which asynchronous operations have not completed. Together these form a
   state.
2. **Turn operations into state transitions.** Define how each operation
   changes the modeled state. An `mbarrier` arrival updates its arrival count
   and may complete a phase. A wait advances the waiting participant only
   after the expected phase has completed.
3. **Search all reachable states.** Starting from the initial state, try every
   event that can execute: a state where the program has not finished but
   nothing can advance is a deadlock; a transition the rules forbid is a
   protocol error.
4. **Return a counterexample.** Give the agent the execution order that leads
   to the error, with the source locations of the operations involved.

The number of states in a real kernel grows quickly, so the search must also
merge duplicate states and prune execution orders that differ only in the
order of independent events.

The following producer–consumer program illustrates how state search finds
an execution order that violates the synchronization protocol.

### Example: Cross-CTA Producer Overtaking

Consider a producer–consumer program with three rounds of signaling and
waiting. It runs on two cooperative thread arrays (CTAs, or CUDA thread blocks)
in one cluster. One thread in CTA 1 signals that each round is ready; one
thread in CTA 0 waits for that signal and records the round's result.

CTA 0 owns an `mbarrier` named `full`. This reusable barrier has successive
phases, each initialized to require one arrival. Each producer arrival
therefore completes a phase. The consumer waits using alternating parity
values, 0 and 1, to track the phase it expects. The producer has no wait that
requires the consumer to observe one phase before it completes the next.

The pseudocode and state graph below follow this three-round program.
Barrier initialization and cluster synchronization are already complete;
timing delays are omitted. The graph branches wherever either participant
can advance.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/state-search.html" title="State search for the three-round cross-CTA example" class="diagram-frame" height="720"></iframe>
  <figcaption>Select a completing path or a counterexample through the three-round program. Transition labels refer to the pseudocode lines.</figcaption>
</figure>
```

Adding a delay before each consumer wait lets the producer run ahead,
exposing the missing coordination. The producer can complete
phases 0 and 1 before the consumer observes phase 0, returning the parity to 0. A
[PTX parity wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
can observe only the current or immediately preceding phase, so phase 0 has
already been missed. The consumer can then get stuck in a later wait after
the producer has finished all its arrivals.

In the GPU runs without the consumer delay, the consumer happens to finish
each wait before the producer's next arrival. The bug is therefore not
triggered, and Compute Sanitizer 2026.1's synccheck, racecheck, and memcheck
report no errors.

A synchronization analysis can use state search to explore the failing order
and report the arrival that overtakes the consumer:

```text
Illustrative diagnostic

Barrier phase reused before consumption
  barrier: full, owned by CTA 0, one arrival per phase
  producer: CTA 1, line 3, arrival for phase 1
  consumer: CTA 0, line 6, phase 0 not yet observed
```

## Data Race Detection

A kernel can complete its synchronization protocol while two warps access
shared data without the required ordering.
Two accesses to the same memory, at least one of them a write, with no
guaranteed order between them, form a data race. The result then depends on
which executes first, and is sometimes right and sometimes wrong. Common forms
include:

- **Reusing a buffer too early**: a consumer issues an asynchronous matrix
  operation that reads shared memory and immediately tells the producer that
  it may write, but the operation may not have finished reading, and the
  producer's next round of data overwrites it.
- **Aliasing**: two differently named variables point to the same storage,
  and a write through one is unordered with a read or write through the other.
- **Protecting the wrong accesses**: a barrier does its job, but it does not
  protect the two accesses that conflict.

Passing numerical tests does not rule out a data race. In the buffer-reuse
example above, the results are correct when the asynchronous read finishes
before the producer overwrites the buffer. Changes in load or timing can alter
this order and expose the error, making the failure intermittent and hard to
reproduce. Race detection must therefore check whether the program orders
conflicting accesses, even when the tested execution produces correct values.

### One Possible Algorithm: Vector Clocks

Detecting a race requires knowing which memory accesses the program orders.
A data race occurs when two accesses to the same memory location, at least
one of them a write, have no **happens-before relation** in either direction.
Here, A *happens before* B means that A precedes B according to the program's
execution and [memory-ordering rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).

One possible algorithm for tracking this relation uses
[vector clocks](https://www.cs.williams.edu/~freund/papers/09-pldi.pdf#page=3).
The analysis treats each independently progressing participant, such as a
warp, as an executor. It numbers each executor's modeled events, such as
memory accesses and synchronization operations, in order: 1, 2, 3, and so on.
For each executor, the analysis keeps a vector clock: its local view of how
far each executor has progressed, with one counter per executor. These views
can differ because an executor learns about others' progress through
synchronization, not simply because they have run ahead.

For example, consider warp 3 and warp 7 after their common setup.
Each has a separate clock with entries `[W3,W7]`. Warp 7's clock `[2,3]`
records its own third event and that warp 3's first two events happen before
it. Warp 3 may already have reached event 5; until synchronization communicates
that progress, warp 7's view of warp 3 remains at 2.

To detect races, the analysis saves the clock at each memory access. Access A
happens before B when every entry in A's clock is less than or equal to the
corresponding entry in B's, with at least one strictly smaller. If neither
clock is ordered before the other this way, the accesses are unordered.
Such unordered accesses form a data race when they touch the same location
and at least one writes.

The analysis maintains these clocks with two rules:

1. **Advance locally.** At each modeled event, update that executor's clock
   by incrementing its own entry. Entries for other executors in this same
   clock remain unchanged: another warp running ahead does not by itself
   establish a happens-before relation.
2. **Merge at synchronization.** When synchronization establishes an ordering
   from one executor to another, merge the sender's clock into the receiver's
   by taking the maximum of each pair of entries, then increment the
   receiver's own entry. The maximum preserves the latest known event from
   each executor, including events the sender learned about through earlier
   synchronization.

Which operations establish order, and for which executors, is determined by
the target hardware's memory model. On new hardware, the algorithm can stay the
same, but these rules must be rebuilt and checked against real devices with
small tests.

### Example: A Race in Tensor Memory

Waiting for a warp's own operations does not establish that another warp's
data is ready.
In the {download}`TMEM example kernel <../traces/chapter-3/tmem_race.py>`,
one CTA contains two warpgroups of four warps each. Warpgroup 0 stores a
128-by-32 tile of uint32 values in tensor memory (TMEM); warpgroup 1 loads the
same tile and writes it to global memory. Both groups compute the same row
indices; `lane` ranges from 0 to 31 within each warp. In the pseudocode below,
`values` holds 32 input elements for each producer thread.

The vector-clock illustration follows one conflicting pair: warp 3 in
warpgroup 0 and warp 7 in warpgroup 1 both access rows 96..127. It shows their
two clock entries as `[W3,W7]`, starting at a common baseline after setup.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/race-happens-before.html" title="Vector clocks for the TMEM example kernel" class="diagram-frame" height="760"></iframe>
  <figcaption>Follow the example kernel's store and load, inspect their synchronization, and compare their vector clocks. The values illustrate one possible analysis.</figcaption>
</figure>
```

The [PTX waits](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-wait)
represented by `wait_own_stores()` and `wait_own_loads()` track the completion
of each issuing thread's own operations. The consumer does not wait for the
producer's stores, so it can read the tile before those stores complete.
Setup synchronization occurs before both accesses, and cleanup synchronization
after both; neither orders the store before the load.

In the recorded runs, the consumer reads incorrect values from TMEM, yet
Compute Sanitizer 2026.1's
[racecheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool)
reports `0 hazards`. Racecheck detects hazards in shared memory; tensor memory
is a separate memory space, so this conflicting TMEM load and store fall
outside its coverage. A race detector that models TMEM accesses can identify
the overlapping load and store and the missing ordering between them:

```text
Illustrative diagnostic

Unordered read/write in TMEM allocation "tile"
  write: line 4, warp 3 (warpgroup 0), rows 96..127, columns 0..31
  read:  line 7, warp 7 (warpgroup 1), rows 96..127, columns 0..31
  ordering: no synchronization edge between the groups' accesses
```
