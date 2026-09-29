# Launching the Agent

Part I introduced the compiler harness and the [agent workflows](agent-workflows.md)
that organize an optimization search; Part II described how TIRx Harness provides the
programming and evaluation tools. We now bring them together to optimize the
M-grouped contiguous FP8 GEMM on a B200 GPU. The agent implements and
optimizes the kernel in TIRx, using DeepGEMM as the correctness reference
and performance baseline.

We use Grouped GEMM here to help you get started quickly: a focused workload
with a straightforward computation lets you run the workflow and see results
before working through a more involved kernel. Starting with
the next chapter's Synccheck and Racecheck examples, we switch to a recorded
Kimi Delta Attention (KDA) run to examine the agent's use of analysis and
performance feedback in detail. The workflow introduced here carries over
to that task.

In this chapter, we will prepare the task workspace and GPU evaluation
service, check the baseline, and launch the optimization run using either
goal mode or Flame Chase. Once the run starts, the chosen workflow carries
the task across agent turns while the harness provides the tools and feedback,
as shown below.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/workflow.html" title="The workflow coordinates agents that use the compiler harness" class="diagram-frame" height="300"></iframe>
  <figcaption>The workflow coordinates agents across turns; the compiler harness supplies the knowledge, tools, and feedback they use. Click a component to inspect its role.</figcaption>
</figure>
```

During the run, the agent retrieves relevant implementations, writes candidate
kernels, and submits them for analysis and GPU evaluation. The returned
feedback guides its next experiment. We first define the task, then prepare
the environment in which this work will take place.

## The Task: Grouped GEMM

Grouped GEMM computes $D_g = A_g B_g^\mathsf{T}$ for groups with different
row counts, as in mixture-of-experts models. We use the M-grouped contiguous
layout with FP8 operands and BF16 output. The task covers four configurations
with 4 or 8 groups and different matrix dimensions.

The agent implements and optimizes this operation in TIRx. The harness fixes
the inputs and correctness checks and measures performance against DeepGEMM.

## Prepare the Environment

An agentic run needs a place for the agent to work and a GPU for its
experiments, both ready before the agent starts. We use two machines: one
runs the coding agent, and the other is a GPU server running KCoral to
execute kernels and collect measurements. The agent and KCoral can also run on the same machine.

First, we will clone the harness and start KCoral on the GPU server. On the
agent machine, `evolution/setup.py` then creates a task worktree and Python
environment with the required packages, skills, and prompt. We will check the
baseline through KCoral before launching the agent in that worktree.

### Prepare the harness checkout

Use Linux x86_64 (glibc 2.38 or newer), Python 3.12 or 3.13, and pip 25.1+
on both machines. The agent machine needs `uv`, Rust 1.89+ with Cargo,
C/C++ build tools, and Python development headers; the GPU server needs
CUDA Toolkit 13.2 and a compatible driver.
For profiling, install Nsight Compute on both machines.

Clone the harness where the agent runs and on the GPU server. If they share
one machine, a single checkout is enough:

```bash
git clone --recursive https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
```

From the checkout root on the agent machine, install the dependencies for
`evolution/setup.py`:

```bash
python -m pip install -r evolution/preparation/requirements.txt
```

### Start KCoral on the GPU server

In the GPU server's terminal, from the harness checkout, install the benchmark
and server dependencies, including DeepGEMM, and start KCoral on GPU 0:

```{warning}
KCoral executes arbitrary code. Allow only trusted clients on an isolated
network; never expose the server to the public internet.
```

```bash
python -m pip install --group server
python -m kcoral server --gpus 0 --host 0.0.0.0 --port 8000
```

Leave this process running. If both roles use the same machine, open another
terminal for the agent.

### Create the run on the agent machine

In the agent machine's terminal, from the harness checkout, set the server
address and check that it responds. Replace `gpu-server`
with the server's hostname or IP address; use `127.0.0.1` for the same machine.

```bash
cd /path/to/TIRx-harness
export KCORAL_URL="http://gpu-server:8000"
curl --fail "$KCORAL_URL/health"
```

Then prepare the task:

```bash
python evolution/setup.py --task grouped_gemm_fp8 \
  --remote "$KCORAL_URL"
```

The run's `.venv` uses the Python interpreter that launched setup and includes
the installed `tirx-harness` wheel. The run directory contains `PROMPT.md`,
`manifest.json`, `flowverse.yaml`, and
`worktree/`; candidate kernels will live under
`candidates/grouped_gemm/fp8/` in that worktree.

Set `run_dir` to the absolute path printed by setup, enter the worktree, and
activate its environment:

```bash
run_dir=/absolute/path/printed/as/run_dir
cd "$run_dir/worktree"
source .venv/bin/activate
cat "$run_dir/PROMPT.md"
```

`PROMPT.md` combines the task definition with this run's worktree path,
service address, benchmark command, authoring rules, and candidate-recording
instructions. The prompt asks the agent to make the kernel as fast as possible
while preserving correctness. Tile sizes, group scheduling, layouts, warp roles,
and pipelining remain the agent's choices.

### Check the baseline

Before starting the agent, run the benchmark command from `PROMPT.md` with
`CANDIDATE` replaced by `baseline`. For this task and remote execution mode,
it has this form:

```bash
python evolution/remote/kcoral_remote.py \
  candidates/grouped_gemm/fp8 baseline \
  --remote "$KCORAL_URL" --timeout 600
```

Confirm that the summary reports `passed: 4/4`. Every group's valid output
rows must pass the correctness check for every configuration; alignment
padding is excluded. For this first call, the candidate is DeepGEMM itself,
the performance baseline. Later calls compare each passing TIRx candidate
against DeepGEMM on the same quantized inputs, with preparation and compilation
outside timing. Each workload reports `DeepGEMM time / candidate time`;
the summary includes the geometric mean across all four workloads.

To evaluate a candidate, replace `baseline` with its directory relative to
the workload, such as `scratch/first`. That directory contains `solution.py`,
which exports `setup(data, G, M, N, K) -> callable` as specified in `PROMPT.md`.

## Kick Off Agentic Runs

An optimization run lasts many turns, so the agent needs a launcher that keeps
it working toward the goal instead of stopping after one answer. Start the run
in `$run_dir/worktree` with the harness environment active. All launch
options use the same task prompt, correctness checks, and benchmark.

::::{container} launch-tabs
```{raw} html
<div class="launch-tablist" aria-label="Choose how to launch the agentic run" hidden>
  <button type="button" id="tab-goal-mode" aria-controls="goal-mode">Goal mode</button>
  <button type="button" id="tab-flame-chase" aria-controls="flame-chase">Flame Chase</button>
</div>
```

:::{container} launch-panel
:name: goal-mode

Goal mode keeps the optimization objective active across turns. In the agent
session, enter the following prompt:

```text
/goal Read ../PROMPT.md in full and carry out the optimization task.
Optimize against the prescribed DeepGEMM performance baseline and retain
reproducibly passing candidates with their measurements in the frontier.
```

Set a time budget or performance target appropriate to this workload.

:::

:::{container} launch-panel
:name: flame-chase

Install Humanize separately using its
[installation instructions](https://github.com/humanfia/humanize#install), then run
[Flame Chase](https://docs.humanfia.ai/humanize/flows/flame-chase)
in the prepared terminal:

```bash
hmz exec -f 'git+https://github.com/humanfia/flowverse#flame_chase' \
  -a 'first_chaser=<first-agent>' -a 'second_chaser=<second-agent>' \
  -b duration=24h \
  "$(cat "$run_dir/PROMPT.md")"
```

Set `first_chaser` and `second_chaser` to the agents you want to use, each
written as `CLI/MODEL:EFFORT`. Agents take turns in fresh sessions using the
same task and worktree. The 24-hour budget controls the run's duration.
:::
::::

Expect the agent to read the task and authoring guidance, inspect relevant
examples, and create an initial implementation. It can diagnose and revise
unsuccessful attempts without waiting for you to propose a fix.

You can use another agent to monitor progress while the optimization agent
keeps working. Give it access to the session trace, retained candidates, and
benchmark reports. Ask it to periodically summarize the best passing result,
recent experiments, current optimization direction, and any issues needing
attention, with evidence from those records.

In the generated worktree, open `frontier/index.json` under
`candidates/grouped_gemm/fp8/` for the retained
candidates and their benchmark results.

## Continuing Through Part III

With the Grouped GEMM workflow in place, we now turn to the recorded KDA
case study. Beginning with Synccheck and Racecheck in the next chapter, we
follow concrete diagnostics, the agent's responses, and the measurements
that guide further optimization. These examples come from the KDA run, not
the Grouped GEMM run you just prepared. The remaining chapters turn to
reviewing results, improving the harness, and keeping future searches
productive:

- [Compiler Analysis Deep Dive](compiler-analysis-deepdive.md) follows
  how the agent calls the analyses, interprets their findings, and repairs
  synchronization and memory-ordering errors.
- [Benchmark Server Deep Dive](benchmark-server-deepdive.md) follows GPU
  experiments through KCoral and shows how timing and profiling feedback
  guides the agent's next optimization.
- [Review and Harness Improvement](self-improvement.md) examines the
  assumptions behind a speedup and shows how review findings become stronger
  checks, improved tools, and reusable kernels.
- [Advanced Tips](advanced-tips.md) discusses how to keep a long search
  productive by starting with one input shape, retaining promising
  alternatives, managing temporary artifacts, and intervening when needed.
