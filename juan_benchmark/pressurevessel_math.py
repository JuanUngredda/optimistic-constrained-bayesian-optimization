"""PressureVessel test function (4 vars, 4 inequality constraints).

Ported from xietaorepo/Mathsys_RG_2024/bo/synthetic_test_functions/synthetic_test_functions.py
lines 694-789 (class PressureVessel). Feasibility convention: g_k(x) <= 0.

x1, x2 are discrete (snapped to multiples of 0.0625 in physical units) per the
canonical formulation.
"""
import math

import torch

LOW = torch.tensor([0.0, 0.0, 10.0, 150.0])
HIGH = torch.tensor([10.0, 10.0, 50.0, 200.0])
DIFF = HIGH - LOW

N_CONSTRAINTS = 4

THICKNESS_INCREMENT = 0.0625


def _round_to_increment(values, increment):
    return torch.round(values / increment) * increment


def unnormalize(xz):
    x = xz.reshape(-1, 4) * DIFF + LOW
    x_snapped = x.clone()
    x_snapped[..., 0] = _round_to_increment(x[..., 0], THICKNESS_INCREMENT)
    x_snapped[..., 1] = _round_to_increment(x[..., 1], THICKNESS_INCREMENT)
    return x_snapped


def objective_raw(xz):
    x = unnormalize(xz)
    x1, x2, x3, x4 = x.unbind(-1)
    return (
        0.6224 * x1 * x3 * x4
        + 1.7781 * x2 * x3 ** 2
        + 3.1661 * x1 ** 2 * x4
        + 19.84 * x1 ** 2 * x3
    )


def bilog(x):
    """Matches botorch.models.transforms.Bilog used in xietaorepo's transform_:
    sign(x) * log1p(|x|). Preserves zero, so feasibility g <= 0 maps to bilog(g) <= 0.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def constraint_raw(xz, idx):
    x = unnormalize(xz)
    x1, x2, x3, x4 = x.unbind(-1)
    if idx == 1:
        return -x1 + 0.0193 * x3
    if idx == 2:
        return -x2 + 0.00954 * x3
    if idx == 3:
        return -math.pi * x3 ** 2 * x4 - (4.0 / 3.0) * math.pi * x3 ** 3 + 1296000.0
    if idx == 4:
        return x4 - 240.0
    raise ValueError(f"constraint idx must be 1..4, got {idx}")


# Outputs are returned in raw (physical) units, matching xietaorepo's PressureVessel
# behaviour. Only inputs are normalized (xz in [0,1]^4 -> physical via `unnormalize`
# above). The feasibility boundary g_k(x) <= 0 is preserved literally, so the JSON
# config uses threshold=0.0 for every constraint.
