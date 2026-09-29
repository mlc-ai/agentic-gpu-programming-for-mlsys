# KCoral Benchmark Server

The previous chapter showed how to express and inspect a kernel. Determining
whether a candidate is correct and fast also requires measurements on the
target GPU. When several agents share a device, those
measurements require coordinated access and separation between experiments.
KCoral, the GPU evaluation service used by TIRx Harness, provides this shared
environment for correctness tests, benchmarks, and profiling captures.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/kcoral.html" title="KCoral components and deployment" class="diagram-frame" height="430"></iframe>
  <figcaption>Select a component to inspect its role.</figcaption>
</figure>
```

KCoral separates the development environment from the machine hosting the
target GPU. Its server manages worker processes and GPU access; evaluation
code defines the input cases, correctness checks, and timing procedure.
The same service supports different kernel benchmarks and diagnostic tools
without embedding their evaluation rules in the server.

In this chapter, we will describe KCoral's execution interface through a
small experiment, then examine how the service coordinates GPU access,
isolates experiments, and supports benchmarking and profiling tools.

## A Small, General Execution Interface

A remote experiment needs to specify what to execute and which results to
bring back. KCoral represents it as a program: an ordered sequence of
instructions describing its code, inputs, operations, and selected outputs.
A program can contain literal arguments and references to values produced
by earlier instructions. Four core operations describe this execution model:

| Core operation | Purpose |
|---|---|
| `upload` | Supply code, tensors, bytes, compiled libraries, or files. |
| `get_function` | Select a named function or object from a module or library. |
| `run` | Call a function using supplied inputs or earlier results. |
| `return` | Select values or filesystem artifacts to return. |

The Python client exposes `Program` for describing an experiment and `Client`
for access to the service, with helpers for operations such as uploading
folders and returning files. The same execution model accommodates benchmark
results, profiling reports, and diagnostic artifacts.

The example below composes these operations into a program that checks and
times an elementwise GPU addition. `KCORAL_URL` is the address of a running
KCoral service with PyTorch and `cupti-python` installed for GPU timing.

```python
import os
from kcoral import Client, Program

source = """
import torch
from kcoral.builtins import benchmark

def evaluate(n):
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    def candidate():
        torch.add(x, 1.0, out=y)

    candidate()
    torch.testing.assert_close(y, x + 1.0)
    return {"correct": True, "timing": benchmark(candidate)}
"""

program = Program()
module = program.upload(kind="module", source=source)
evaluate = program.get_function(module=module, name="evaluate")
report = program.run(fn=evaluate, args=[4096])
program.return_(key="report", value=report)

with Client(os.environ["KCORAL_URL"]) as client:
    result = client.execute(program)

if not result.completed:
    raise RuntimeError(result.error)
print(result.results["report"])
```

`module`, `evaluate`, and `report` are references to values that will be
created on the server. `Client.execute` submits the program; the uploaded
function defines the check and measurement, and `return_` selects the report
available in `result.results`.

## Self-Contained Experiments

Repeating an experiment should not require reconstructing hidden state left
by earlier runs. Each program therefore identifies the code and inputs it
needs, relative to the server's installed compiler, libraries, and GPU tools. It does not depend on tensors,
functions, or temporary files left by an earlier execution. This stateless
interface makes dependencies explicit and allows workers to be replaced
between experiments.

An output from one experiment can become an explicit input to another.
KCoral caches uploaded content to reduce repeated transfers, but this cache
does not preserve live Python objects or GPU tensors between programs.
Each program still identifies its required inputs.

## Coordinated GPU Access

Comparing two candidates requires measurements that are not distorted by
another worker using the same GPU at the same time. KCoral coordinates
experiments on GPUs dedicated to the service. A GPU lease grants a worker
exclusive access to a managed device for checks, timings, or profiling. Workers sharing that device take turns, while separate
devices can support experiments in parallel. Explicitly CPU-only functions
can run without holding the lease.

Lease coordination assumes that experiments finish their GPU work before
releasing the device. It prevents overlapping GPU work by cooperating
workers; evaluation code still determines the measurement procedure,
including warmup, repetition, and input selection.

## Isolated Files and Process State

Files left by an experiment can affect another experiment's code or results.
With filesystem isolation enabled, KCoral uses bubblewrap to give workers a
private writable workspace and read-only runtime dependencies. Other workers'
workspaces are hidden. Each execution has a temporary working directory that
is removed after completion, including handled failures; retained outputs
are explicit return artifacts.

Process memory needs a separate boundary. Imported modules, allocations, and
a CUDA context can persist in a long-lived worker, and a failed kernel can
leave its context unusable. By default, KCoral replaces the worker after each
program, giving the next experiment a fresh Python process and CUDA context.
Workspace isolation and process replacement address these two sources of
state independently.

## One Endpoint for Multiple GPU Hosts

As evaluation expands to several GPU hosts, clients need a way to submit
experiments without choosing a host themselves. KCoral's optional router
provides one service address for multiple GPU hosts, each running a KCoral
server. The router manages admission and selects
available nodes; each server manages its own workers and GPU leases.
Development tools use the same execution interface without tracking
individual host addresses and availability.

## Integrating Benchmarking, Profiling, and Diagnostics

To understand a kernel's behavior and identify further optimization
opportunities, an agent needs the outputs of GPU tools as well as a timing
number. Those tools have their own launch options
and output formats. TIRx Harness connects them to KCoral with
command-line adapters. Each adapter packages the required code and inputs,
configures the tool in the server's installed environment, and exposes its
output and exit status locally. For tools that produce files, the adapter
also retrieves the selected artifacts:

| Adapter | Responsibility |
|---|---|
| `kcoral_python.py` | Run a user-supplied Python script, command, or module for custom tests and experiments; return standard output, standard error, and exit status. |
| `kcoral_iket.py` | Capture In-Kernel Event Tracing (IKET) timelines and save traces and retained intermediates locally. |
| `kcoral_ncu.py` | Capture a Nsight Compute (NCU) hardware report for local inspection. |
| `kcoral_compute_sanitizer.py` | Run Compute Sanitizer and retrieve selected diagnostic files. |
| `kcoral_remote.py` | Evaluate a baseline or candidate using a packaged workload’s predefined correctness checks and timing procedure; return benchmark results. |

The adapters handle the integration, while KCoral supplies GPU coordination
and experiment isolation. The experiment still provides the kernel, input
setup, and any required instrumentation, such as IKET's named stage markers.
Timelines, hardware reports, and sanitizer findings are diagnostic evidence;
ordinary benchmarks establish performance comparisons because profiling
and instrumentation can affect execution costs.

Together, the compiler foundation and KCoral connect source changes to GPU
results: the agent can express a candidate, submit an experiment, and inspect
the returned measurements and artifacts. Part III begins with a
[running optimization workflow](launching-the-agent.md) that uses these capabilities,
then examines the analysis and benchmark feedback in more detail.
