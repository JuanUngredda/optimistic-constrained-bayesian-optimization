"""SpeedReducer test function (7 vars, 11 inequality constraints).

Ported from xietaorepo/Mathsys_RG_2024/bo/synthetic_test_functions/synthetic_test_functions.py
lines 884-1039 (class SpeedReducer). Feasibility convention: g_k(x) <= 0.
"""
import torch

LOW = torch.tensor([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
HIGH = torch.tensor([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
DIFF = HIGH - LOW

N_CONSTRAINTS = 11


def unnormalize(xz):
    return xz.reshape(-1, 7) * DIFF + LOW


def objective_raw(xz):
    x = unnormalize(xz)
    x1, x2, x3, x4, x5, x6, x7 = x.unbind(-1)
    return (
        0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
        + 7.4777 * (x6 ** 3 + x7 ** 3)
        + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)
    )


def bilog(x):
    """Matches botorch.models.transforms.Bilog used in xietaorepo's transform_:
    sign(x) * log1p(|x|). Preserves zero, so feasibility g <= 0 maps to bilog(g) <= 0.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def constraint_raw(xz, idx):
    x = unnormalize(xz)
    x1, x2, x3, x4, x5, x6, x7 = x.unbind(-1)
    if idx == 1:
        return 27.0 / (x1 * x2 ** 2 * x3) - 1
    if idx == 2:
        return 397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1
    if idx == 3:
        return 1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1
    if idx == 4:
        return 1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1
    if idx == 5:
        return torch.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (0.1 * x6 ** 3) - 1100
    if idx == 6:
        return torch.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (0.1 * x7 ** 3) - 850
    if idx == 7:
        return x2 * x3 - 40
    if idx == 8:
        return 5 - x1 / x2
    if idx == 9:
        return x1 / x2 - 12
    if idx == 10:
        return (1.5 * x6 + 1.9) / x4 - 1
    if idx == 11:
        return (1.1 * x7 + 1.9) / x5 - 1
    raise ValueError(f"constraint idx must be 1..11, got {idx}")


# Outputs are returned in raw (physical) units, matching xietaorepo's PressureVessel /
# SpeedReducer behaviour. Only inputs are normalized (xz in [0,1]^7 -> physical via
# `unnormalize` above). The feasibility boundary g_k(x) <= 0 is preserved literally,
# so the JSON config uses threshold=0.0 for every constraint.
