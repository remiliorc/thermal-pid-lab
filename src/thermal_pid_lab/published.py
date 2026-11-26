"""Published, experimentally identified models of two laboratory thermal processes.

Source (open access, CC BY-NC-SA 4.0):
    Rivero-Contreras, R., Zamarreño, J. M., Tadeo, F. (2022).
    "Modelado experimental de dos sistemas de laboratorio multivariables".
    XLIII Jornadas de Automática: libro de actas, pp. 599-606.
    https://doi.org/10.17979/spudc.9788497498418.0599

Only values printed in the paper are used: first-order discrete transfer
functions identified with the output-error (OE) method from sequential step
inputs (test T.1), which the paper recommends for both systems.
Poles: Tables 3 and 9. Gains: Tables 5 and 11.

Each channel is a first-order model with the unit-delay operator:

    G(z) = K (1 - p) z^-1 / (1 - p z^-1)

i.e. y(k+1) = p y(k) + K (1 - p) u(k), in deviation variables around the
identification operating point (the paper high-pass filters the data to move
the operating point to zero, and does not report the operating point itself).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CITATION = ("Rivero-Contreras, Zamarreño, Tadeo (2022). Modelado experimental de "
            "dos sistemas de laboratorio multivariables. XLIII Jornadas de "
            "Automática, pp. 599-606. doi:10.17979/spudc.9788497498418.0599")


@dataclass(frozen=True)
class Channel:
    gain: float        # steady-state gain [degC per %]
    pole: float        # discrete pole at the system's sample time

    def time_constant(self, ts: float) -> float:
        """Equivalent continuous time constant: tau = -Ts / ln(p)."""
        if not 0 < self.pole < 1:
            raise ValueError("time constant only defined for 0 < pole < 1")
        return -ts / math.log(self.pole)


@dataclass(frozen=True)
class PublishedSystem:
    name: str
    description: str
    ts: float                                   # sample time used for identification [s]
    inputs: dict[str, str]
    outputs: dict[str, str]
    channels: dict[tuple[str, str], Channel]    # (output, input) -> channel
    notes: str = ""

    def gain_matrix(self) -> np.ndarray:
        outs, ins = list(self.outputs), list(self.inputs)
        m = np.full((len(outs), len(ins)), np.nan)
        for (o, i), ch in self.channels.items():
            m[outs.index(o), ins.index(i)] = ch.gain
        return m


TCLAB = PublishedSystem(
    name="TCLab",
    description=("Temperature Control Lab: two heaters (0-100 %) and two "
                 "temperature sensors on one board, with thermal coupling."),
    ts=20.0,
    inputs={"u1": "heater 1 power [%]", "u2": "heater 2 power [%]"},
    outputs={"y1": "temperature 1 [degC]", "y2": "temperature 2 [degC]"},
    channels={
        ("y1", "u1"): Channel(0.878, 0.886),
        ("y1", "u2"): Channel(0.177, 0.935),
        ("y2", "u1"): Channel(0.378, 0.942),
        ("y2", "u2"): Channel(0.492, 0.879),
    },
    notes="Data logged every 1 s, resampled to 20 s for identification.",
)

CONDITIONING_SYSTEM = PublishedSystem(
    name="Temperature conditioning system",
    description=("Blower-heater supplying hot air (power 0-100 %) and a fan "
                 "supplying ambient air (speed 0-100 %), mixed downstream; "
                 "three PT-100 sensors."),
    ts=60.0,
    inputs={"u1": "blower-heater power [%]", "u2": "fan speed [%]"},
    outputs={"y1": "hot-air outlet [degC]", "y2": "ambient-air outlet [degC]",
             "y3": "mixing point [degC]"},
    channels={
        ("y1", "u1"): Channel(0.318, 0.517),
        ("y2", "u1"): Channel(0.019, 0.960),
        ("y3", "u1"): Channel(0.155, 0.855),
        # outlier: the same paper reports -0.044 (LS) and -0.069 (OE, test T.2)
        # for this channel; kept as published for the recommended OE/T.1 model
        ("y3", "u2"): Channel(-0.464, 0.735),
    },
    notes=("Data logged every 5 s, resampled to 60 s and filtered. The paper "
           "reports no first-order OE model for y1<-u2 (N/A) and a pole of "
           "1.000 for y2<-u2; both are omitted here."),
)

SYSTEMS = {"tclab": TCLAB, "conditioner": CONDITIONING_SYSTEM}


@dataclass
class DiscretePlant:
    """Single channel of a published model, run at its own sample time.

    Works with the same simulation helpers as ``ThermalPlant``. The output is
    the temperature rise above the starting point (``t_ambient`` = 0), and the
    model is assumed valid from 0 % power upwards: an assumption made for this
    demo, because the paper identifies the model around an operating point it
    does not report.
    """

    channel: Channel
    dt: float
    noise_std: float = 0.0
    seed: int | None = 0
    t_ambient: float = 0.0
    temperature: float = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.channel.pole < 1 or self.dt <= 0:
            raise ValueError("need a stable pole (0 < p < 1) and dt > 0")
        self.reset()

    @property
    def gain(self) -> float:
        return self.channel.gain

    def reset(self, temperature: float | None = None, heater_pct: float = 0.0) -> None:
        u0 = float(np.clip(heater_pct, 0.0, 100.0))
        self.temperature = self.gain * u0 if temperature is None else temperature
        self._rng = np.random.default_rng(self.seed)

    def steady_state_input(self, temperature: float) -> float:
        return (temperature - self.t_ambient) / self.gain

    def step(self, heater_pct: float, disturbance: float = 0.0) -> float:
        """One sample: y(k+1) = p y(k) + K (1-p) u(k) - (1-p) d(k).

        ``disturbance`` is a load expressed as the equivalent drop in
        steady-state temperature [degC]."""
        u = float(np.clip(heater_pct, 0.0, 100.0))
        p = self.channel.pole
        self.temperature = p * self.temperature + (1 - p) * (self.gain * u - disturbance)
        return self.measure()

    def measure(self) -> float:
        if self.noise_std > 0:
            return self.temperature + float(self._rng.normal(0.0, self.noise_std))
        return self.temperature


def published_loop(system: str = "tclab", output: str = "y1", input: str = "u1",
                   noise_std: float = 0.0, seed: int | None = 0) -> DiscretePlant:
    """A ready-to-simulate SISO loop from one published channel."""
    sysm = SYSTEMS[system]
    return DiscretePlant(sysm.channels[(output, input)], sysm.ts, noise_std, seed)
