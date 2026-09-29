# Review and Harness Improvement

Before adopting a kernel produced by an agent, we need to understand how it
achieves its speedup and where it can be used. Human review assesses the
kernel's numerical changes, input assumptions, and performance claims against
the original task, examining the implementation alongside its checks and
measurements.

Human review also identifies findings to incorporate into the harness for
future agent runs. In this chapter, we will describe how to review the
assumptions behind a speedup and how to build what the review finds into the
harness.

```{raw} html
<figure class="interactive-figure">
  <iframe src="../_static/diagrams/harness-improvement.html" title="Review kernel results and improve the harness for later runs" class="diagram-frame" height="340"></iframe>
  <figcaption>Select a component to see how review findings become stronger numerical checks, analysis tool fixes, and reusable kernels.</figcaption>
</figure>
```

## Review the Assumptions Behind the Speedup

Start by asking the agent to identify the changes responsible for the speedup
and explain the conditions under which they preserve the task's required
behavior. For numerical rewrites, these conditions may concern intermediate
precision and value ranges; for specializations, they may restrict the
supported inputs.

Then compare those conditions with the existing validation. Where the evidence
is incomplete, have the agent investigate the specific assumption with analysis
or tests and revise the implementation if needed; the reviewer decides whether
the resulting evidence supports adoption.

For example, an earlier agent-evolved Kimi Delta Attention kernel passed its tests despite a
rewrite of its decay calculation that could make intermediate values underflow
or overflow.

### The rewrite: compute the decay factors separately

To reduce exponential calculations, the kernel splits the decay between an
earlier token $j$ and a later token $i$ into two separately computed factors.
For one key channel, let $c_t$ be the cumulative base-2 log decay from the
chunk's start through token $t$. The rewrite is

$$
2^{c_i-c_j}
\quad\longrightarrow\quad
2^{c_i}\left(\frac{1}{2^{c_j}}\right).
$$

For example, the decay from tokens 1 and 2 to token 3 would require
$2^{c_3-c_1}$ and $2^{c_3-c_2}$. After the rewrite, both use the same value
$2^{c_3}$. The kernel computes it once and combines it with the saved
reciprocals $1/2^{c_1}$ and $1/2^{c_2}$. Across a chunk, each token's two
factors are reused for many pairs.

The two expressions are equal in real arithmetic, but not when the kernel
computes the factors separately with its fp32 instructions, which flush tiny
exponentials to zero.

### How a decay of 0.5 becomes NaN

This failure can occur under strong decay, which makes the cumulative log
values very negative. Consider a 64-token chunk whose first 32 decay
coefficients are $1/16$.
Each adds $-4$ to the cumulative log decay, giving $c_j=-128$ at token 32.
If the next coefficient is $1/2$, then $c_i=-129$ at token 33. The decay
between these adjacent tokens should be
$2^{-129-(-128)}=0.5$.

Once the cumulative log decay falls below roughly $-126$, the separated
calculation cannot recover that $0.5$:

| Calculation | Result in this example |
|---|---|
| Earlier token: $2^{c_j}$ | Flushed to $0$ |
| Reciprocal: $1 / 2^{c_j}$ | $+\infty$ |
| Later token: $2^{c_i}$ | Flushed to $0$ |
| Combined decay | $0 \times \infty = \mathrm{NaN}$ |

The NaNs enter the matrix products and can propagate through the recurrent
state; a PyTorch reproduction of the kernel's arithmetic produced NaNs in both
output and state under strong decay.

The candidate passed because the earlier tests reached a cumulative log decay
of only about $-54$, never crossing the failure boundary near $-126$.
Tightening their error tolerance would not expose a failure on inputs they did
not test.

The benchmark can therefore reward an invalid optimization. This is a
reward-hacking risk even without deliberate cheating.

### Repair and validate the candidate

Once targeted tests confirm the failure, the agent needs to revise the decay
calculation to avoid the intermediate reciprocal that can become infinite
under strong decay. Computing the log-decay difference before the exponential
is one way to do this. To retain the matrix-multiplication structure, the
agent can follow the reference implementation, which measures decay from a
boundary between smaller blocks and keeps both factors at most one.

The agent should then test the repaired GPU kernel around the failure boundary
and on real application inputs. Both per-token output and final state must
match an independent reference within the task's error limits, with NaNs and
infinities rejected. Because the repair may change performance, re-benchmark
the repaired kernel against the performance baseline.

## Improve the Harness

Findings that stay in one run's files and notes are easy for later agents to
miss, so they end up repeating the same work. Build them into the harness
instead, where later agents meet them through the tools they already use.

### Expand input coverage

Add inputs that expose invalid optimizations within the supported domain to
the task's regular correctness checks. In this example, strong-decay cases
make the unsafe factorization fail during the search. The task definition
already includes this coverage.

### Repair analysis tools

When debugging exposes an analysis tool defect, use the trace to isolate a reproducer
and validate a repair. For example, replaying Gated DeltaNet evolution kernels exposed
a simulator wake-up bug that made Synccheck falsely report deadlocks. Retain
the reproducer as a regression test and integrate the reviewed fix into the
harness.

### Reuse reviewed kernels

Add reviewed implementations to the {ref}`kernel zoo <knowledge-base>`
with supported inputs, validation results, timings, and explanations of key
design choices. A reviewed kernel can serve as a starting point for later runs
on the same operator, and its design notes help agents adapt its techniques to
related kernels. Recheck correctness and performance when kernel or compiler
changes could affect the result.

These changes close the loop: the next search uses better checks, repaired
tools, and reviewed implementations.
