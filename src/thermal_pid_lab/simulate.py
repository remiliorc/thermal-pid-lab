"""Closed-loop simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .pid import PID
from .plant import ThermalPlant
from .published import DiscretePlant

Plant = ThermalPlant | DiscretePlant


@dataclass
class Trace:
    t: np.ndarray
    sp: np.ndarray
    pv: np.ndarray
    u: np.ndarray


def closed_loop(plant: Plant, pid: PID, duration: float,
                sp_fn: Callable[[float], float],
                dist_fn: Callable[[float], float] = lambda t: 0.0) -> Trace:
    n = int(duration / plant.dt)
    t = np.arange(n) * plant.dt
    sp = np.zeros(n)
    pv = np.zeros(n)
    u = np.zeros(n)
    y = plant.measure()
    for k in range(n):
        # at t[k]: measure, compute, apply; the plant then advances to t[k+1]
        pv[k] = y
        sp[k] = sp_fn(t[k])
        u[k] = pid.update(sp[k], y)
        y = plant.step(u[k], dist_fn(t[k]))
    return Trace(t, sp, pv, u)


def open_loop_step(plant: Plant, u0: float, u1: float, t_step: float,
                   duration: float) -> Trace:
    """Step test starting from the steady state at ``u0``."""
    plant.reset(heater_pct=u0)
    n = int(duration / plant.dt)
    t = np.arange(n) * plant.dt
    u = np.where(t >= t_step, u1, u0).astype(float)
    pv = np.empty(n)
    y = plant.measure()
    for k in range(n):
        pv[k] = y                 # measured at t[k], before u[k] acts
        y = plant.step(u[k])
    return Trace(t, np.full(n, np.nan), pv, u)
