"""Minimal TMEM race: one warpgroup writes TMEM while another reads it, with no ordering.

Written in tirx-lite; needs `tirx_kernels` and `tirx_harness` (TIRx-harness) importable.

usage:
  python tmem_race.py racecheck [--fixed]   # CPU: tirx_harness Racecheck
  python tmem_race.py gpu [--fixed]         # GPU: run once and compare (use under compute-sanitizer)
"""
import sys

import numpy as np
import tvm

import tirx_kernels.tirx_lite as txl

FIXED = "--fixed" in sys.argv
COLS = 32
TMEM_ST = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TMEM_LD = "tcgen05.ld.sync.aligned.32x32b.x32.b32"


def tmem_sync():
    """Order tcgen05 operations across the threads of the CTA."""
    txl.ptx.tcgen05.fence__before_thread_sync()
    txl.cuda.cta_sync()
    txl.ptx.tcgen05.fence__after_thread_sync()


def make_kernel(fixed):
    @txl.kernel(warps=8)
    def tmem_handoff(out: txl.gptr[txl.u32, (128, COLS)]):
        warp = txl.warp_id()
        warpgroup, row = warp // 4, (warp % 4) * 32 + txl.lane_id()
        mailbox = txl.smem_pool().alloc((1,), "uint32", align=4)
        with txl.If(warp == 0), txl.Then():
            txl.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                mailbox.ptr_to([0]), txl.uint32(COLS))
        tmem_sync()
        base = txl.local_scalar(txl.u32)
        txl.ptx.ld.shared.u32(base, mailbox.ptr_to([0]))
        addr = base + (txl.cast(warp % 4, "uint32") << 21)      # this warp's 32 TMEM lanes
        regs = txl.alloc_local([COLS], "uint32")

        with txl.If(warpgroup == 0), txl.Then():                # producer: registers -> TMEM
            for col in range(COLS):
                txl.assign(regs[col], txl.cast(row * COLS + col + 1, "uint32"))
            txl.ptx[TMEM_ST](addr, *(regs[i] for i in range(COLS)))
            txl.ptx.tcgen05.wait__st.sync.aligned()
        if fixed:                                               # order the write before the read
            tmem_sync()
        with txl.If(warpgroup == 1), txl.Then():                # consumer: TMEM -> registers -> out
            txl.ptx[TMEM_LD](*(regs[i] for i in range(COLS)), addr)
            txl.ptx.tcgen05.wait__ld.sync.aligned()
            for col in range(COLS):
                txl.ptx.st.global_.u32(out.ptr_to([row, col]), regs[col])

        tmem_sync()
        with txl.If(warp == 0), txl.Then():
            txl.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            txl.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(base, txl.uint32(COLS))
    return tmem_handoff


kernel = make_kernel(FIXED)
expected = (np.arange(128 * COLS, dtype=np.uint32) + 1).reshape(128, COLS)

if sys.argv[1] == "racecheck":
    from tirx_harness import racecheck
    report = racecheck(kernel.func, inputs={"out": np.zeros((128, COLS), np.uint32)})
    report.print()
else:
    ex = kernel.compile()
    dev = tvm.cuda(0)
    out = tvm.runtime.tensor(np.zeros((128, COLS), np.uint32), dev)
    ex["main"](out)
    dev.sync()
    got = out.numpy()
    print("fixed" if FIXED else "racy", "output matches expected:", bool(np.array_equal(got, expected)),
          "| wrong elements:", int((got != expected).sum()))
