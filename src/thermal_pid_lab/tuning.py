"""Model identification and PID tuning rules (textbook methods).

- FOPDT identification from an open-loop step test (Smith two-point method)
- Relay feedback autotuning (Astrom-Hagglund) to estimate Ku and Pu
- Tuning rules: SIMC (Skogestad 2003), IMC-PI, Ziegler-Nichols, Tyreus-Luyben
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FOPDT:
    gain: float
    tau: float
    dead_time: float


@dataclass(frozen=True)
class PIDParams:
    kc: float
    ti: float
    td: float = 0.0
    rule: str = ""


def identify_fopdt_step(t: np.ndarray, y: np.ndarray, step_size: float,
                        t_step: float = 0.0) -> FOPDT:
    """Smith's method: times to reach 28.3 % and 63.2 % of the final change."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if step_size == 0:
        raise ValueError("step_size must be non-zero")
    pre = y[t < t_step]
    mask = t >= t_step
    t, y = t[mask] - t_step, y[mask]
    # average the pre-step baseline so one noisy sample does not bias y0
    y0 = float(np.mean(pre)) if len(pre) else float(y[0])
    tail = max(1, len(y) // 20)
    dy = float(np.mean(y[-tail:]) - y0)
    if abs(dy) < 1e-9:
        raise ValueError("no response detected")
    frac = (y - y0) / dy
    t28 = float(np.interp(0.283, frac, t)) if np.all(np.diff(frac) >= -1e-9) else _first_cross(t, frac, 0.283)
    t63 = float(np.interp(0.632, frac, t)) if np.all(np.diff(frac) >= -1e-9) else _first_cross(t, frac, 0.632)
    tau = 1.5 * (t63 - t28)
    theta = max(t63 - tau, 0.0)
    return FOPDT(gain=dy / step_size, tau=tau, dead_time=theta)


def _first_cross(t: np.ndarray, frac: np.ndarray, level: float) -> float:
    idx = int(np.argmax(frac >= level))
    if idx == 0:
        return float(t[0])
    f0, f1 = frac[idx - 1], frac[idx]
    return float(t[idx - 1] + (level - f0) / (f1 - f0) * (t[idx] - t[idx - 1]))


def with_sampling_delay(m: FOPDT, ts: float) -> FOPDT:
    """Add the apparent dead time of a zero-order hold (Ts/2) to a model,
    the usual correction before applying continuous tuning rules to a loop
    sampled at ``ts``."""
    return FOPDT(m.gain, m.tau, m.dead_time + ts / 2)


def simc_pi(m: FOPDT, tau_c: float | None = None) -> PIDParams:
    """SIMC PI. Default tau_c = theta (tight but robust)."""
    tc = m.dead_time if tau_c is None else tau_c
    if tc + m.dead_time <= 0:
        raise ValueError("SIMC needs dead_time > 0 or tau_c > 0")
    kc = m.tau / (m.gain * (tc + m.dead_time))
    ti = min(m.tau, 4 * (tc + m.dead_time))
    return PIDParams(kc, ti, 0.0, f"SIMC-PI (tau_c={tc:.0f}s)")


def imc_pi(m: FOPDT, lam: float) -> PIDParams:
    kc = m.tau / (m.gain * (lam + m.dead_time))
    return PIDParams(kc, m.tau, 0.0, f"IMC-PI (lambda={lam:.0f}s)")


def zn_open_loop_pid(m: FOPDT) -> PIDParams:
    if m.dead_time <= 0:
        raise ValueError("Ziegler-Nichols open-loop rules need dead_time > 0")
    theta = m.dead_time
    kc = 1.2 * m.tau / (m.gain * theta)
    return PIDParams(kc, 2 * theta, 0.5 * theta, "Ziegler-Nichols (open loop)")


def zn_ultimate_pid(ku: float, pu: float) -> PIDParams:
    return PIDParams(0.6 * ku, pu / 2, pu / 8, "Ziegler-Nichols (ultimate)")


def tyreus_luyben_pi(ku: float, pu: float) -> PIDParams:
    return PIDParams(ku / 3.2, 2.2 * pu, 0.0, "Tyreus-Luyben PI")


@dataclass(frozen=True)
class RelayResult:
    ku: float
    pu: float
    amplitude: float
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray


def relay_autotune(step_fn: Callable[[float], float], setpoint: float,
                   bias: float, relay_amp: float, dt: float,
                   duration: float, hysteresis: float = 0.0,
                   initial_pv: float | None = None) -> RelayResult:
    """Astrom-Hagglund relay test on a plant exposed through ``step_fn(u)->pv``.

    Ku = 4 d / (pi * sqrt(a^2 - eps^2)) with hysteresis eps, Pu = mean period
    of the last oscillations. This is a describing-function approximation:
    on the continuous FOPDT model in this repo it underestimates Ku
    (conservative), while on a sampled loop that oscillates at the Nyquist
    frequency it overestimates Ku by a factor 4/pi. Both cases are unit-tested.
    """
    n = int(duration / dt)
    t = np.arange(n) * dt
    y = np.zeros(n)
    u = np.zeros(n)
    pv = initial_pv if initial_pv is not None else setpoint
    out = bias + relay_amp
    for k in range(n):
        if pv > setpoint + hysteresis:
            out = bias - relay_amp
        elif pv < setpoint - hysteresis:
            out = bias + relay_amp
        u[k] = out
        pv = step_fn(out)
        y[k] = pv

    up = np.where((u[1:] > u[:-1]))[0] + 1
    if len(up) < 4:
        raise RuntimeError("relay test did not produce enough oscillations")
    periods = np.diff(t[up])[-3:]
    pu = float(np.mean(periods))
    start = up[-4]
    a = float((np.max(y[start:]) - np.min(y[start:])) / 2)
    if a <= 0:
        raise RuntimeError("zero oscillation amplitude")
    if a <= hysteresis:
        raise RuntimeError("oscillation amplitude not above hysteresis")
    ku = 4 * relay_amp / (math.pi * math.sqrt(a ** 2 - hysteresis ** 2))
    return RelayResult(ku, pu, a, t, y, u)
