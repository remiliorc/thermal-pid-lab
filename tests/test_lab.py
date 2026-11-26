import numpy as np
import pytest

from thermal_pid_lab import (PID, ThermalPlant, closed_loop,
                             identify_fopdt_step, open_loop_step,
                             relay_autotune, simc_pi, step_metrics)


def test_plant_steady_state_matches_gain():
    p = ThermalPlant(gain=0.5, tau=50, dead_time=5, t_ambient=20)
    for _ in range(2000):
        y = p.step(40)
    assert y == pytest.approx(20 + 0.5 * 40, abs=0.01)


def test_dead_time_delays_response():
    p = ThermalPlant(dead_time=10, dt=1)
    p.reset(heater_pct=0)
    ys = [p.step(100) for _ in range(10)]
    assert all(y == pytest.approx(p.t_ambient) for y in ys)
    assert p.step(100) > p.t_ambient


def test_identification_recovers_model():
    p = ThermalPlant(gain=2.5, tau=900, dead_time=60)
    tr = open_loop_step(p, 40, 60, 250, 10000)
    m = identify_fopdt_step(tr.t, tr.pv, 20, t_step=250)
    assert m.gain == pytest.approx(2.5, rel=0.03)
    assert m.tau == pytest.approx(900, rel=0.1)
    assert m.dead_time == pytest.approx(60, abs=25)


def test_relay_autotune_finds_oscillation():
    p = ThermalPlant()
    p.reset(heater_pct=p.steady_state_input(150))
    r = relay_autotune(lambda u: p.step(u), 150, 50, 15, 1.0, 7500, 0.1, 150)
    assert r.ku > 0 and 100 < r.pu < 2000


def test_pid_tracks_setpoint_and_rejects_disturbance():
    p = ThermalPlant()
    m = identify_fopdt_step(*_step(), step_size=20, t_step=250)
    par = simc_pi(m)
    p.reset(heater_pct=p.steady_state_input(150))
    pid = PID(par.kc, par.ti, par.td)
    pid.set_manual(p.steady_state_input(150))
    pid.set_auto(150, 150)
    tr = closed_loop(p, pid, 12000, lambda t: 180 if t >= 300 else 150,
                     lambda t: 10 if t >= 6000 else 0)
    assert tr.pv[5900] == pytest.approx(180, abs=0.3)
    assert tr.pv[-1] == pytest.approx(180, abs=0.3)


def _step():
    p = ThermalPlant()
    tr = open_loop_step(p, 40, 60, 250, 10000)
    return tr.t, tr.pv


def test_anti_windup_limits_overshoot_when_saturated():
    p = ThermalPlant()
    p.reset(heater_pct=0)
    pid = PID(kc=5, ti=300, out_max=100)
    tr = closed_loop(p, pid, 15000, lambda t: 200)
    m = step_metrics(tr.t, tr.pv, tr.u, 22, 200)
    assert m.overshoot_pct < 15


def test_bumpless_transfer():
    pid = PID(kc=2, ti=100)
    pid.set_manual(37.0)
    pid.set_auto(sp=50, pv=45)
    assert pid.update(50, 45) == pytest.approx(37.0, abs=1e-9)


def test_metrics_basic():
    t = np.arange(100.0)
    y = np.minimum(t / 50, 1.0) * 10
    m = step_metrics(t, y, np.zeros(100), 0, 10)
    assert m.overshoot_pct == 0
    assert m.rise_time == pytest.approx(40, abs=1)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        ThermalPlant(tau=0)
    with pytest.raises(ValueError):
        PID(kc=1, ti=0)


def test_no_derivative_kick_on_setpoint_step():
    pid = PID(kc=2, ti=100, td=20, beta=1.0)
    pid.set_manual(30)
    pid.set_auto(sp=40, pv=40)
    u0 = pid.update(40, 40)
    u1 = pid.update(50, 40)          # setpoint jumps, PV unchanged
    assert u1 - u0 == pytest.approx(2 * 10, abs=1e-6)   # P only, no D kick


def test_setpoint_weighting_reduces_proportional_kick():
    full = PID(kc=2, ti=1e9, beta=1.0)
    half = PID(kc=2, ti=1e9, beta=0.5)
    for c in (full, half):
        c.set_manual(30)
        c.set_auto(40, 40)
    assert full.update(50, 40) - 30 == pytest.approx(20, abs=1e-3)
    assert half.update(50, 40) - 30 == pytest.approx(10, abs=1e-3)


def test_tuning_rules_formulas():
    from thermal_pid_lab import FOPDT, imc_pi, zn_ultimate_pid, tyreus_luyben_pi
    m = FOPDT(gain=2.0, tau=100.0, dead_time=10.0)
    p = imc_pi(m, lam=20)
    assert p.kc == pytest.approx(100 / (2 * 30)) and p.ti == 100
    z = zn_ultimate_pid(ku=4, pu=40)
    assert (z.kc, z.ti, z.td) == (pytest.approx(2.4), 20, 5)
    t = tyreus_luyben_pi(ku=3.2, pu=10)
    assert t.kc == pytest.approx(1.0) and t.ti == pytest.approx(22)


def test_relay_estimate_close_to_analytic_ultimate_point():
    # analytic ultimate point of K e^{-theta s}/(tau s + 1): phase = -180 deg
    import math
    K, tau, theta = 2.5, 900.0, 60.0
    lo, hi = 1e-4, 1.0
    for _ in range(100):
        w = (lo + hi) / 2
        phase = -w * theta - math.atan(w * tau)
        lo, hi = (w, hi) if phase > -math.pi else (lo, w)
    ku = math.sqrt(1 + (w * tau) ** 2) / K
    pu = 2 * math.pi / w
    p = ThermalPlant()
    p.reset(heater_pct=p.steady_state_input(150))
    r = relay_autotune(lambda u: p.step(u), 150, 50, 15, 1.0, 7500, 0.1, 150)
    assert r.pu == pytest.approx(pu, rel=0.2)
    assert 0.55 * ku < r.ku <= ku      # describing function is conservative


def test_anti_windup_is_stable_with_coarse_sampling():
    # dt larger than the tracking time used to make the integrator diverge
    # and the output chatter between the limits
    pid = PID(kc=1, ti=1, dt=3)
    outs, integ = [], []
    for _ in range(200):
        outs.append(pid.update(sp=1000, pv=0))   # permanently saturated high
        integ.append(pid._integral)
    assert set(outs[-50:]) == {100.0}            # no chatter
    assert abs(integ[-1] - integ[-2]) < 1e-6     # converged, bounded
    for _ in range(5):
        u = pid.update(sp=0, pv=50)              # error reverses
    assert u < 100                               # leaves saturation at once


def test_settling_requires_staying_in_band():
    t = np.arange(100.0)
    y = np.ones(100) * 10
    y[95] = 5                     # leaves the band near the end of the window
    assert step_metrics(t, y, np.zeros(100), 0, 10).settling_time is None
    y[95] = 10
    y[50] = 5
    assert step_metrics(t, y, np.zeros(100), 0, 10).settling_time == 51


def test_trace_timestamps_measure_before_acting():
    p = ThermalPlant(dead_time=0, dt=1)
    tr = open_loop_step(p, 0, 50, 10, 30)
    assert tr.pv[10] == pytest.approx(p.t_ambient)   # step applied at t=10
    assert tr.pv[11] > p.t_ambient                    # visible one sample later
