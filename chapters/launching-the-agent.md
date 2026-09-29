# Launching the Agent

Part I introduced the compiler harness and the agent workflows
that organize an optimization search; Part II described how TIRx Harness provides the
programming and evaluation tools. We now bring them together to optimize the
forward pass of Kimi Delta Attention on a B200 GPU, for one sequence of
8,192 tokens with 96 heads.

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
feedback guides its next experiment. We begin by preparing the environment
in which this work will take place.

## Prepare the Environment

An agentic run needs a place for the agent to work and a GPU for its
experiments, both ready before the agent starts. This example optimizes
Kimi Delta Attention forward on a B200 with `B=1`,
`T=8192`, `H=96`, and `K=V=128`. We use two machines: one runs the coding
agent, and the other is a GPU server running KCoral to execute kernels and
collect measurements. The agent and KCoral can also run on the same machine.

First, we will clone the harness and start KCoral on the GPU server. On the
agent machine, `evolution/setup.py` then creates a task worktree and Python
environment with the required packages, skills, and prompt. We will check the
baseline through KCoral before launching the agent in that worktree.

### Prepare the harness checkout

Use Linux x86_64, Python 3.12 or 3.13, and pip 25.1+ on both machines.
The agent machine needs `uv`, Rust 1.89+ with Cargo, C/C++ build tools, and
Python development headers; the GPU server needs CUDA and a compatible driver.
For profiling, install Nsight Compute on both machines.

Clone the harness where the agent runs and on the GPU server. If they share
one machine, a single checkout is enough:

```bash
git clone https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
git submodule update --init thirdparty/tvm-rust-ext
```

From the checkout root on the agent machine, install the dependencies for
`evolution/setup.py`:

```bash
python -m pip install -r evolution/preparation/requirements.txt
```

### Start KCoral on the GPU server

In the GPU server's terminal, from the harness checkout, install the benchmark
and server dependencies and start KCoral on GPU 0:

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

Then prepare the fixed-shape task:

```bash
python evolution/setup.py --task kda_forward_b1_t8192_h96 \
  --remote "$KCORAL_URL"
```

The run's `.venv` uses the Python interpreter that launched setup, with harness
and benchmark dependencies installed from `uv.lock`. The run directory
contains `PROMPT.md`, `manifest.json`,
`flowverse.yaml`, and `worktree/`; candidate kernels will live under
`candidates/kda/forward_b1_t8192_h96/` in that worktree.

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
while preserving correctness. Chunking, fusion, layouts, warp roles, and
pipelining remain the agent's choices.

### Check the baseline

Before starting the agent, run the benchmark command from `PROMPT.md` with
`CANDIDATE` replaced by `baseline`. For this task and remote execution mode,
it has this form:

```bash
python evolution/remote/kcoral_remote.py \
  candidates/kda/forward_b1_t8192_h96 baseline \
  --remote "$KCORAL_URL" --timeout 600
```

Confirm that the output includes the timed row `kda-fwd-h96-fixed`, seven
stress probes, and two fp64 probes, all at the required H=96, T=8192 shape.
The final summary must report `passed: 10/10`; individual rows also print
`passed: 1/1`. The private holdout result is reported separately. For this
first call, the candidate is FlashKDA itself, the performance baseline; later
calls time each passing candidate against it in the same call.

## Kick Off Agentic Runs

An optimization run lasts many turns, so the agent needs a launcher that keeps
it working toward the goal instead of stopping after one answer. Start the run
in the prepared worktree with its environment active. All launch
options use the same generated task, correctness checks, benchmark, and
candidate directory.

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
Reach at least 3.0x speedup over the prescribed performance baseline
with a reproducibly passing kernel and retain it in the frontier.
```

The 3× speedup target is an example; choose a target that fits your task.

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
`candidates/kda/forward_b1_t8192_h96/` for the retained candidates and their
benchmark results.

## Continuing Through Part III

The next two chapters follow analysis and performance feedback from a
recorded Kimi Delta Attention run to show how the agent diagnoses errors and
improves its kernels. The remaining chapters turn to reviewing results,
improving the harness, and keeping future searches productive:

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
