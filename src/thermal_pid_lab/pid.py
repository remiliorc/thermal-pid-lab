"""Industrial-style discrete PID controller.

Features commonly found in DCS/PLC function blocks:
- ISA (standard) form: Kc * (e + 1/Ti * int(e) + Td * de/dt)
- derivative on measurement with first-order filter Tf = max(Td/N, 2*dt)
  (no derivative kick; filter never faster than the sample time)
- setpoint weighting on the proportional term (beta)
- output limits with back-calculation anti-windup
- bumpless manual -> auto transfer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PID:
    kc: float
    ti: float                  # integral time [s]; float('inf') disables I
    td: float = 0.0            # derivative time [s]
    dt: float = 1.0
    beta: float = 1.0          # setpoint weight on P
    n_filter: float = 10.0     # derivative filter: Tf = max(Td/N, 2*dt)
    out_min: float = 0.0
    out_max: float = 100.0
    tt: float | None = None    # anti-windup tracking time; default sqrt(Ti*Td) or Ti
    auto: bool = True
    _integral: float = field(default=0.0, init=False)
    _d_state: float = field(default=0.0, init=False)
    _last_pv: float | None = field(default=None, init=False)
    output: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.ti <= 0:
            raise ValueError("dt and ti must be positive")
        if self.out_min >= self.out_max:
            raise ValueError("out_min must be below out_max")
        if self.tt is None:
            self.tt = (self.ti * self.td) ** 0.5 if self.td > 0 else self.ti

    def set_manual(self, output: float) -> None:
        self.auto = False
        self.output = min(max(output, self.out_min), self.out_max)

    def set_auto(self, sp: float, pv: float) -> None:
        """Bumpless transfer: preload the integral so the output does not jump."""
        p = self.kc * (self.beta * sp - pv)
        self._integral = self.output - p
        self._d_state = 0.0
        self._last_pv = pv
        self.auto = True

    def update(self, sp: float, pv: float) -> float:
        if not self.auto:
            self._last_pv = pv
            return self.output

        p = self.kc * (self.beta * sp - pv)

        if self.td > 0 and self._last_pv is not None:
            tf = max(self.td / self.n_filter, 2 * self.dt)
            alpha = self.dt / (tf + self.dt)
            raw = -self.kc * self.td * (pv - self._last_pv) / self.dt
            self._d_state += alpha * (raw - self._d_state)
        d = self._d_state if self.td > 0 else 0.0
        self._last_pv = pv

        v = p + self._integral + d
        u = min(max(v, self.out_min), self.out_max)

        if self.ti != float("inf"):
            e = sp - pv
            # back-calculation in exact discrete form: the correction never
            # exceeds the saturation error, even when dt > tt (coarse sampling)
            track = 1.0 - math.exp(-self.dt / self.tt)
            self._integral += self.dt * self.kc / self.ti * e + track * (u - v)

        self.output = u
        return u
