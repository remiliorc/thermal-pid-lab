"""Closed-loop performance indicators."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class StepMetrics:
    iae: float
    ise: float
    overshoot_pct: float
    rise_time: float | None
    settling_time: float | None
    total_variation: float

    def as_dict(self) -> dict:
        return asdict(self)


def step_metrics(t, y, u, sp_initial: float, sp_final: float,
                 band: float = 0.02, min_hold: float = 0.1) -> StepMetrics:
    """Step-response indicators.

    Settling time is only reported if the response then stays inside the band
    for at least ``min_hold`` of the window (default 10 %); otherwise the loop
    is considered not settled and ``settling_time`` is None."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    u = np.asarray(u, float)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    e = sp_final - y
    span = sp_final - sp_initial
    if span == 0:
        raise ValueError("setpoint change must be non-zero")
    frac = (y - sp_initial) / span
    overshoot = max(0.0, float(np.max(frac) - 1.0) * 100)

    rise = None
    i10 = np.argmax(frac >= 0.1) if np.any(frac >= 0.1) else None
    i90 = np.argmax(frac >= 0.9) if np.any(frac >= 0.9) else None
    if i10 is not None and i90 is not None:
        rise = float(t[i90] - t[i10])

    outside = np.where(np.abs(frac - 1.0) > band)[0]
    if len(outside) == 0:
        settle = 0.0
    else:
        last = outside[-1]
        held = len(t) - 1 - last
        settle = (float(t[last + 1] - t[0])
                  if held >= max(1, int(min_hold * len(t))) else None)

    return StepMetrics(
        iae=float(np.sum(np.abs(e)) * dt),
        ise=float(np.sum(e ** 2) * dt),
        overshoot_pct=overshoot,
        rise_time=rise,
        settling_time=settle,
        total_variation=float(np.sum(np.abs(np.diff(u)))),
    )
