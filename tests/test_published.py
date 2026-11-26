import math

import pytest

from thermal_pid_lab import (FOPDT, open_loop_step, relay_autotune, simc_pi,
                             with_sampling_delay, zn_open_loop_pid)
from thermal_pid_lab.published import (CONDITIONING_SYSTEM, TCLAB, Channel,
                                       DiscretePlant, published_loop)


def test_values_match_the_paper_tables():
    # Tables 3 and 5 (TCLab) and 9 and 11 (conditioning system), OE, test T.1
    assert TCLAB.ts == 20 and CONDITIONING_SYSTEM.ts == 60
    assert TCLAB.channels[("y1", "u1")] == Channel(0.878, 0.886)
    assert TCLAB.channels[("y1", "u2")] == Channel(0.177, 0.935)
    assert TCLAB.channels[("y2", "u1")] == Channel(0.378, 0.942)
    assert TCLAB.channels[("y2", "u2")] == Channel(0.492, 0.879)
    assert CONDITIONING_SYSTEM.channels[("y1", "u1")] == Channel(0.318, 0.517)
    assert CONDITIONING_SYSTEM.channels[("y3", "u2")] == Channel(-0.464, 0.735)


def test_time_constant_from_pole():
    tau = TCLAB.channels[("y1", "u1")].time_constant(TCLAB.ts)
    assert tau == pytest.approx(-20 / math.log(0.886))
    with pytest.raises(ValueError):
        Channel(1.0, 1.0).time_constant(20)


def test_discrete_plant_reaches_published_gain():
    p = published_loop("tclab", "y1", "u1")
    p.reset(heater_pct=0)
    for _ in range(300):
        y = p.step(40)
    assert y == pytest.approx(0.878 * 40, rel=1e-6)


def test_identification_recovers_published_model():
    p = published_loop("tclab", "y1", "u1")
    tr = open_loop_step(p, 20, 50, 200, 4000)
    from thermal_pid_lab import identify_fopdt_step
    m = identify_fopdt_step(tr.t, tr.pv, 30, t_step=200)
    assert m.gain == pytest.approx(0.878, rel=0.01)
    assert m.tau == pytest.approx(-20 / math.log(0.886), rel=0.05)


def test_relay_matches_discrete_ultimate_point():
    # proportional loop on y(k+1) = p y + b u(k): unstable when Kc > (1+p)/b,
    # oscillating at the Nyquist frequency (period 2 Ts)
    K, pole, ts = 0.878, 0.886, 20.0
    ku = (1 + pole) / (K * (1 - pole))
    p = published_loop("tclab", "y1", "u1")
    p.reset(heater_pct=p.steady_state_input(30))
    r = relay_autotune(lambda u: p.step(u), 30, 34, 15, ts, 4000, 0.1, 30)
    assert r.pu == pytest.approx(2 * ts)
    # At the Nyquist frequency the plant does not filter the relay's square
    # wave, so the describing-function estimate is high by a factor 4/pi.
    assert r.ku == pytest.approx(4 / math.pi * ku, rel=0.05)


def test_sampling_delay_and_guards():
    m = FOPDT(1.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        zn_open_loop_pid(m)
    with pytest.raises(ValueError):
        simc_pi(m)
    m2 = with_sampling_delay(m, 20)
    assert m2.dead_time == 10
    assert simc_pi(m2).kc > 0 and zn_open_loop_pid(m2).ti == 20


def test_discrete_plant_validation():
    with pytest.raises(ValueError):
        DiscretePlant(Channel(1.0, 1.0), 20)
