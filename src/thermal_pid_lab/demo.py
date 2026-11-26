"""End-to-end demo on a published, experimentally identified model.

Plant: TCLab heater 1 -> temperature 1 (first-order OE model, Ts = 20 s), from
Rivero-Contreras, Zamarreño, Tadeo (2022), XLIII Jornadas de Automática.
Steps: identify -> tune -> compare -> report.

Run (after `pip install -e .`):  python -m thermal_pid_lab.demo --out docs
or from a checkout without installing:  PYTHONPATH=src python -m thermal_pid_lab.demo --out docs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import (PID, closed_loop, identify_fopdt_step,
               open_loop_step, relay_autotune, simc_pi, step_metrics,
               tyreus_luyben_pi, with_sampling_delay, zn_open_loop_pid)

from .published import CITATION, published_loop

# temperatures are rises above the starting point [degC]
SP0, SP1, T_SP, T_DIST, DURATION = 30.0, 40.0, 200.0, 2000.0, 4000.0
DIST = 5.0        # load disturbance [degC equivalent]
NOISE = 0.1       # measurement noise added for the demo (not from the paper)
BAND = 0.05       # settling band: 5 % of the step, i.e. +/-0.5 degC, above the noise


def make_plant():
    return published_loop("tclab", "y1", "u1", noise_std=NOISE, seed=1)


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    # 1) Open-loop step test for identification
    p = make_plant()
    step = open_loop_step(p, u0=20.0, u1=50.0, t_step=200.0, duration=4000.0)
    model = identify_fopdt_step(step.t, step.pv, step_size=30.0, t_step=200.0)
    # the loop runs at Ts = 20 s: account for the zero-order hold (Ts/2)
    tuning_model = with_sampling_delay(model, p.dt)

    # 2) Relay autotune (closed loop, no model needed)
    p = make_plant()
    p.reset(heater_pct=p.steady_state_input(SP0))
    relay = relay_autotune(lambda u: p.step(u), setpoint=SP0, bias=34.0,
                           relay_amp=15.0, dt=p.dt, duration=4000.0,
                           hysteresis=0.1, initial_pv=SP0)

    rules = [simc_pi(tuning_model), zn_open_loop_pid(tuning_model),
             tyreus_luyben_pi(relay.ku, relay.pu)]

    results, traces = [], []
    for r in rules:
        plant = make_plant()
        u_ss = plant.steady_state_input(SP0)
        plant.reset(heater_pct=u_ss)
        pid = PID(kc=r.kc, ti=r.ti, td=r.td, dt=plant.dt)
        pid.set_manual(u_ss)
        pid.set_auto(SP0, SP0)
        tr = closed_loop(plant, pid, DURATION,
                         sp_fn=lambda t: SP1 if t >= T_SP else SP0,
                         dist_fn=lambda t: DIST if t >= T_DIST else 0.0)
        sel = (tr.t >= T_SP) & (tr.t < T_DIST)
        m = step_metrics(tr.t[sel], tr.pv[sel], tr.u[sel], SP0, SP1, band=BAND)
        results.append({"rule": r.rule, "kc": round(r.kc, 3),
                        "ti": round(r.ti, 1), "td": round(r.td, 1),
                        **{k: (round(v, 2) if isinstance(v, float) else v)
                           for k, v in m.as_dict().items()}})
        traces.append((r.rule, tr))

    # Figures
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(step.t, step.pv)
    ax[0].set_ylabel("Temperature rise [degC]")
    ax[0].set_title(f"Open-loop step | identified K={model.gain:.2f}, "
                    f"tau={model.tau:.0f}s, theta={model.dead_time:.0f}s")
    ax[1].step(step.t, step.u, where="post")
    ax[1].set_ylabel("Heater [%]")
    ax[1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(out / "01_step_identification.png", dpi=110, metadata={"Software": None})
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(relay.t, relay.y)
    ax[0].axhline(SP0, ls="--", lw=0.8)
    ax[0].set_ylabel("Temperature rise [degC]")
    ax[0].set_title(f"Relay autotune | Ku={relay.ku:.2f}, Pu={relay.pu:.0f}s")
    ax[1].step(relay.t, relay.u, where="post")
    ax[1].set_ylabel("Heater [%]")
    ax[1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(out / "02_relay_autotune.png", dpi=110, metadata={"Software": None})
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(traces[0][1].t, traces[0][1].sp, "k--", lw=0.8, label="SP")
    for name, tr in traces:
        ax[0].plot(tr.t, tr.pv, label=name)
        ax[1].plot(tr.t, tr.u, label=name)
    ax[0].axvline(T_DIST, color="grey", lw=0.6)
    ax[0].set_ylabel("Temperature rise [degC]")
    ax[0].set_title(f"Setpoint step (t={T_SP:.0f} s) and load disturbance (t={T_DIST:.0f} s)")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("Heater [%]")
    ax[1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(out / "03_tuning_comparison.png", dpi=110, metadata={"Software": None})
    plt.close(fig)

    report = {
        "plant": "TCLab y1<-u1, published first-order OE model (K=0.878, p=0.886, Ts=20 s)",
        "source": CITATION,
        "identified_model": {"gain": round(model.gain, 3),
                             "tau": round(model.tau, 1),
                             "dead_time": round(model.dead_time, 1),
                             "dead_time_for_tuning": round(tuning_model.dead_time, 1)},
        "relay": {"ku": round(relay.ku, 3), "pu": round(relay.pu, 1)},
        "closed_loop": results,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs", type=Path)
    args = ap.parse_args()
    print(json.dumps(run(args.out), indent=2))


if __name__ == "__main__":
    main()
