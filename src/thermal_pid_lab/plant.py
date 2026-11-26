"""Generic thermal process: first-order plus dead time (FOPDT) heater model.

A textbook lumped-capacity model of an electrically heated industrial oven
exchanging heat with its surroundings:

    tau * dT/dt = -(T - T_amb) + K * u(t - theta) - d(t)

where ``u`` is heater power in percent (0-100), ``K`` the static gain in
degC per percent, ``tau`` the time constant and ``theta`` the transport
delay; ``d`` is an optional load disturbance. All values are illustrative
defaults, not taken from any real asset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThermalPlant:
    gain: float = 2.5          # degC per % of heater power
    tau: float = 900.0         # time constant [s]
    dead_time: float = 60.0    # transport delay [s]
    t_ambient: float = 22.0    # ambient temperature [degC]
    dt: float = 1.0            # integration step [s]
    noise_std: float = 0.0     # measurement noise [degC]
    seed: int | None = 0
    temperature: float = field(init=False)
    _buffer: deque = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tau <= 0 or self.dt <= 0:
            raise ValueError("tau and dt must be positive")
        if self.dead_time < 0:
            raise ValueError("dead_time must be >= 0")
        self.reset()

    def reset(self, temperature: float | None = None,
              heater_pct: float = 0.0) -> None:
        """Reset state. Without ``temperature`` the plant starts at the
        steady state that corresponds to a constant ``heater_pct``."""
        u0 = float(np.clip(heater_pct, 0.0, 100.0))
        self.temperature = (self.t_ambient + self.gain * u0
                            if temperature is None else temperature)
        delay_steps = int(round(self.dead_time / self.dt))
        self._buffer = deque([u0] * delay_steps, maxlen=delay_steps or None)
        self._rng = np.random.default_rng(self.seed)

    def steady_state_input(self, temperature: float) -> float:
        """Heater power that holds ``temperature`` at steady state."""
        return (temperature - self.t_ambient) / self.gain

    def step(self, heater_pct: float, disturbance: float = 0.0) -> float:
        """Advance one sample. Returns the measured temperature.

        ``disturbance`` is a heat-loss term expressed as the equivalent drop
        in steady-state temperature [degC] (e.g. a door opening or a cold
        load), applied without delay.
        """
        u = float(np.clip(heater_pct, 0.0, 100.0))
        if self._buffer.maxlen:
            delayed_u = self._buffer[0]
            self._buffer.append(u)
        else:
            delayed_u = u
        dTdt = (-(self.temperature - self.t_ambient) + self.gain * delayed_u
                - disturbance) / self.tau
        self.temperature += self.dt * dTdt
        return self.measure()

    def measure(self) -> float:
        if self.noise_std > 0:
            return self.temperature + float(self._rng.normal(0.0, self.noise_std))
        return self.temperature
