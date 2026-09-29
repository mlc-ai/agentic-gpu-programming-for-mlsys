"""Minimal synchronization bug that depends on timing: cross-CTA producer overtaking.

Two CTAs in one cluster. CTA 1 (producer) arrives on CTA 0's `full` mbarrier
remotely once per round, but never waits for CTA 0 to consume the previous
phase. A round takes a while to produce, so in normal runs CTA 0 keeps up and
the kernel finishes. Nothing forces that order: if CTA 0 falls behind
(--slow-consumer), two phases complete back to back and CTA 0 waits forever.
--fixed adds an `empty` mbarrier on CTA 1 that CTA 0 arrives on remotely.

Written in tirx-lite; needs `tirx_kernels` and `tirx_harness` (TIRx-harness) importable.

usage:
  python cluster_overtake.py synccheck [--fixed]              # CPU: tirx_harness Synccheck
  python cluster_overtake.py gpu [--fixed] [--slow-consumer]  # GPU (also run under compute-sanitizer)
"""
import sys

import numpy as np
import tvm

import tirx_kernels.tirx_lite as txl

FIXED = "--fixed" in sys.argv
SLOW = "--slow-consumer" in sys.argv
ROUNDS = 3


def make_kernel(fixed, slow):
    @txl.kernel(warps=1, grid=False)
    def cluster_overtake(out: txl.gptr[txl.i32, (ROUNDS,)]):
        txl.cta_id([2])
        rank = txl.cta_id_in_cluster([2])
        lane = txl.lane_id()
        bars = txl.smem_pool().alloc((2,), "uint64", align=8)
        full, empty = bars.ptr_to([0]), bars.ptr_to([1])
        remote = txl.local_scalar(txl.u32)
        with txl.If(lane == 0), txl.Then():
            txl.ptx.mbarrier.init.shared.b64(full, txl.uint32(1))
            txl.ptx.mbarrier.init.shared.b64(empty, txl.uint32(1))
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cluster_sync()
        for r in range(ROUNDS):
            with txl.If(txl.And(rank == 1, lane == 0)), txl.Then():  # producer on CTA 1
                if r > 0:
                    txl.cuda.nano_sleep(txl.uint64(200000))          # producing a round takes a while
                    if fixed:
                        txl.cuda.mbarrier_wait(empty, (r - 1) % 2)
                txl.ptx["mapa.shared::cluster.u32"](                  # full barrier on CTA 0
                    remote, txl.cuda.cvta_generic_to_shared(full), txl.uint32(0))
                txl.ptx["mbarrier.arrive.release.cluster.shared::cluster.b64"](remote)
            with txl.If(txl.And(rank == 0, lane == 0)), txl.Then():  # consumer on CTA 0
                if slow:
                    txl.cuda.nano_sleep(txl.uint64(400000))
                txl.cuda.mbarrier_wait(full, r % 2)
                txl.ptx.st.global_.s32(out.ptr_to([r]), txl.int32(r + 1))
                if fixed:
                    txl.ptx["mapa.shared::cluster.u32"](              # empty barrier on CTA 1
                        remote, txl.cuda.cvta_generic_to_shared(empty), txl.uint32(1))
                    txl.ptx["mbarrier.arrive.release.cluster.shared::cluster.b64"](remote)
        txl.cuda.cluster_sync()
    return cluster_overtake


kernel = make_kernel(FIXED, SLOW)
if sys.argv[1] == "synccheck":
    from tirx_harness import synccheck
    synccheck(kernel.func, inputs={"out": np.zeros(ROUNDS, np.int32)}).print()
else:
    ex = kernel.compile()
    dev = tvm.cuda(0)
    out = tvm.runtime.tensor(np.zeros(ROUNDS, np.int32), dev)
    ex["main"](out)
    dev.sync()
    print("fixed" if FIXED else "buggy", "slow-consumer" if SLOW else "", "finished, out =", out.numpy().tolist())
