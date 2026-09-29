# Benchmarking and Profiling

An agent needs reliable performance measurements to judge whether a kernel
change is an improvement. Other work running on the same GPU can distort
these measurements, so benchmark runs need to be isolated from competing
workloads.

The agent also needs to understand why a kernel is slow. Profiling helps
identify where time is spent and what limits performance, giving the agent
a basis for choosing its next optimization.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/benchmarking.html" title="Benchmarking and profiling for an agent" class="diagram-frame" height="420"></iframe>
  <figcaption>Select a component to see how the agent uses it. The highlighted connections show the experiment submission, GPU execution, and returned feedback.</figcaption>
</figure>
```

In this chapter, we will describe how to isolate GPU experiments for
reliable benchmarking and use profiling to understand performance
bottlenecks.

## Benchmark Server

When multiple agents share a GPU, concurrent experiments can interfere with
each other's correctness checks, timings, and profiles. A misleading result
can send the search in the wrong direction, so the execution environment
needs to coordinate GPU access across agents.

Experiments can also affect later runs through the state they leave behind.
Temporary files and process memory may persist, and a failed kernel can leave
its CUDA context unusable. Agents could coordinate device access, clean up
workspaces, and restart failed processes themselves, but doing so adds
execution-management work to every search.

The target GPU may also be on a remote server or an edge device.
Even when the target device can host the agent, developing on a separate
workstation leaves more of the device's compute and memory for kernel
execution and helps keep measurements consistent. A remote execution service
runs the agent's experiments on the target device and returns the results for
local inspection.

These needs call for a benchmark server that combines remote execution with
isolation between experiments. It should give each experiment exclusive
access to its assigned GPU, a private workspace, and a fresh process and CUDA
context. The service handles cleanup and returns results to the agent's
development environment.

Benchmark results tell the agent whether a candidate is faster, but choosing
the next change requires understanding where time goes inside the kernel and
what limits its progress. The same service should therefore support profiling
on the target GPU and return timelines and hardware measurements for the agent
to inspect.

## In-Kernel Event Tracing

High-performance kernels often divide work into stages, such as loading a
tile, computing on it, and writing back the result. Different warps can handle
these stages and overlap their execution. A single kernel latency does not
reveal which stage delays the others, so locating bottlenecks in their
cooperation requires:

- **Recording the start and end of each stage inside the kernel**, separately
  for each CTA and warp;
- **Naming stages by meaning**, such as "wait for TMA data," "MMA compute,"
  and "write back," so that the timeline maps back to the source;
- **Disturbing the measured execution as little as possible**: recording must
  be cheap and must not add synchronization of its own.

[IKET (In-Kernel Event Tracing)](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html)
provides this view by recording timestamped markers and named ranges inside
the kernel. The agent places range boundaries around the stages it wants to
inspect, then compares their durations and overlap across warps and CTAs to
locate long waits or imbalanced work.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/iket-timeline.html" title="An in-kernel timeline with TMA, MMA, and epilogue warps" class="diagram-frame" height="320"></iframe>
  <figcaption>An illustrative timeline for three warps in one CTA. The MMA warp spends most of its time waiting for tile data; the epilogue stores the result after the last MMA.</figcaption>
</figure>
```

In this timeline, the MMA warp performs each matrix operation quickly but
waits a long time for the next tile. The load stage is the bottleneck, so the
agent should investigate why it takes so long.

## Hardware Counter Profiling

A timeline can show that a stage is waiting, but further investigation may be
needed to identify the cause: memory bandwidth, too few instructions issued,
or register spills. Hardware profiling provides:

- **Hardware counters**, including throughput, occupancy, memory traffic,
  cache hit rates, and warp stall reasons;
- **Metrics associated with code**, showing where sampled stalls and memory
  accesses concentrate;
- **A selected capture scope**, so the agent can inspect the kernel and
  invocation relevant to its hypothesis.

NVIDIA Nsight Compute (NCU) collects these measurements. Its
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
describes the metric groups and collection methods. The agent uses these
measurements to test a hypothesis from a timeline. As another example, heavy
local-memory traffic can lead the agent to inspect register-spill metrics and
local-memory instructions in the affected code.

The opening figure illustrates how these views fit together. Timing ranks
the candidate against a baseline; a timeline locates a wait; hardware
measurements help investigate its cause. Each result gives the agent evidence
for a more specific next experiment.
