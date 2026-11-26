"""thermal-pid-lab: a generic thermal process simulator with PID tuning tools."""

from .plant import ThermalPlant
from .pid import PID
from .tuning import (FOPDT, PIDParams, identify_fopdt_step, simc_pi, imc_pi,
                     zn_open_loop_pid, zn_ultimate_pid, tyreus_luyben_pi,
                     relay_autotune, with_sampling_delay)
from .metrics import step_metrics, StepMetrics
from .simulate import closed_loop, open_loop_step, Trace

__all__ = [
    "ThermalPlant", "PID", "FOPDT", "PIDParams", "identify_fopdt_step",
    "simc_pi", "imc_pi", "zn_open_loop_pid", "zn_ultimate_pid",
    "tyreus_luyben_pi", "relay_autotune", "with_sampling_delay", "step_metrics", "StepMetrics",
    "closed_loop", "open_loop_step", "Trace",
]
__version__ = "0.1.0"
