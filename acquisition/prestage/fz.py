#!/usr/bin/env python3
"""Run a script with subnormal numbers flushed to zero on the CPU (2026-11-15).

  $PY prestage/fz.py w16u_ens.py <arguments of w16u_ens.py>

Reason: in a long fit of ``w16u_ens.py`` on the CPU, operations on subnormal numbers make one round of the setting s04
(width 512, 5493 steps) take about 1160 s instead of 150 s to 224 s (REPORTS/修正記録.md, 2026-11-15). The author decided
on 2026-11-15 that the production path of the pre-stage sets ``torch.set_flush_denormal(True)`` before the script
starts. This changes the computed values (subnormal operands and results are treated as zero), so every computation
of the production path that is compared with another one must go through this wrapper. ``w16u_ens.py`` is unchanged.
"""
import runpy
import sys

import torch

assert torch.set_flush_denormal(True), "this CPU does not support flushing subnormal numbers"
script = sys.argv[1]
sys.argv = sys.argv[1:]
print(f"[fz] torch.set_flush_denormal(True); running {script}", flush=True)
runpy.run_path(script, run_name="__main__")
