# Advanced Tips

A long optimization run can stall even when every tool works. A change may
speed up an 8,192-token input but slow down a shorter one, leaving an average
score that hides the tradeoff. An initially faster fused kernel may lead the
agent to abandon a split implementation before exploring its potential.
Notes from many attempts on the fused kernel can then keep the agent focused
on small scheduling edits instead of revisiting that choice.

In this chapter, we will describe practices that keep a long search
productive: optimizing one input shape (the input tensor dimensions, such as
sequence length and head count) before generalizing, keeping promising
alternatives, and cleaning up temporary artifacts.

## Optimize One Shape, Then Generalize

When the application needs several shapes, start with one representative
shape. This can lead to faster performance gains for two reasons:

- Correctness checks and benchmarks run faster on one shape, so the agent
  spends less time waiting for tools and more time reasoning about and
  implementing optimizations.
- A single shape gives the agent a clearer reward signal. With multiple
  shapes, the same change may speed up one configuration and slow down
  another, making it harder to decide which direction to pursue.

Once the agent has a useful candidate, extend it to the other required shapes.
Revisit assumptions tied to the initial shape and check both the original and
added cases against the expanded task's computation and acceptance criteria.
The Kimi Delta Attention task in this part fixes H=96 and T=8192. To generalize it, prepare a task that
explicitly includes the additional shapes and use the retained implementation
as a starting point for that search.

## Keep Promising Alternatives

Even for one shape, early timings may favor an approach that later runs out
of room to improve. Retain the best passing candidate alongside a few that use
different optimization ideas. For example, a fused kernel can reduce launch
overhead and intermediate storage, while splitting stages can expose more
parallel work and improve SM utilization. A currently slower approach may
provide a useful starting point for another shape or optimization.

Keep each candidate's source, checks, paired timings, and a short explanation
of its distinguishing idea together. The example task calls this pool the
frontier. Remove redundant versions while keeping alternatives that still
offer a direction worth exploring.

## Clean Up Temporary Artifacts

Keeping useful alternatives does not require keeping every attempt. A long
attempt history can encourage the agent to keep refining its current route
without reconsidering its assumptions.

Clean up the experiment artifacts periodically, for example every 12 hours.
Keep the current frontier; archive earlier scratch candidates, attempt
histories, and modification logs outside the agent's working directory so they
remain available for review.

During the search, intervene when proceeding requires a change to the task or
additional resources.
