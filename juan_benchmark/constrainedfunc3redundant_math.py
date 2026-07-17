"""ConstrainedFunc3Redundant test function (2 vars, 5 inequality constraints).

Ported from xietaorepo/Mathsys_RG_2024/bo/synthetic_test_functions/synthetic_test_functions.py
lines 486-540 (class ConstrainedFunc3Redundant), which extends ConstrainedFunc3
(lines 419-483). Feasibility convention: c_k(x) <= 0.

The source domain is already normalized to [0,1]^2 (_bounds = [(0,1),(0,1)]),
so `unnormalize` is the identity map (kept for interface parity with
pressurevessel_math.py / speedreducer_math.py).

The objective and all constraints are returned raw (unlike the engineering
benchmarks, this synthetic function has no bilog output transform in the
original source -- physical units and normalized units coincide here, so
there is nothing to compress). The feasibility boundary c_k(x) <= 0 is
therefore literal, matching the JSON config's threshold=0.0 for every
constraint.

The objective is a maximization objective as written in the source and is
returned as-is (this framework maximizes).

c4 and c5 are constant (-100), i.e. always trivially satisfied -- included
only to test robustness to redundant constraints.
"""
import torch

N_CONSTRAINTS = 5


def unnormalize(xz):
    return xz.reshape(-1, 2)


def objective_raw(xz):
    x = unnormalize(xz)
    x0, x1 = x.unbind(-1)
    return -((x0 - 1.0) ** 2) - (x1 - 0.5) ** 2


def constraint_raw(xz, idx):
    x = unnormalize(xz)
    x0, x1 = x.unbind(-1)
    if idx == 1:
        return ((x0 - 3.0) ** 2 + (x1 + 2.0) ** 2) * torch.exp(-(x1 ** 7)) - 12.0
    if idx == 2:
        return 10.0 * x0 + x1 - 7.0
    if idx == 3:
        return (x0 - 0.5) ** 2 + (x1 - 0.5) ** 2 - 0.2
    if idx == 4:
        return torch.full_like(x0, -100.0)
    if idx == 5:
        return torch.full_like(x0, -100.0)
    raise ValueError(f"constraint idx must be in 1..{N_CONSTRAINTS}, got {idx}")
