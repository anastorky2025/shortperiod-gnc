import unittest
import numpy as np
from shortperiod_gnc import (example_aircraft, lqr, closed_loop_lqr, pid_pitch_loop,
                             step_metrics, simulate)


class Tests(unittest.TestCase):
    def test_open_loop_stable_short_period(self):
        self.assertTrue(np.all(example_aircraft().short_period_poles().real < 0))

    def test_wn_zeta_match_poles(self):
        m = example_aircraft()
        wn, z = m.natural_frequency_damping()
        p = m.short_period_poles()
        self.assertAlmostEqual(abs(p[0]), wn, places=6)
        self.assertAlmostEqual(-p[0].real / wn, z, places=6)

    def test_lqr_scalar(self):
        K, P = lqr([[0.0]], [[1.0]], [[1.0]], [[1.0]])
        self.assertAlmostEqual(float(K[0, 0]), 1.0, places=6)

    def test_lqr_closed_loop_stable_and_tracks(self):
        A, B, C, D, K = closed_loop_lqr(example_aircraft())
        self.assertTrue(np.all(np.linalg.eigvals(A).real < 0))
        t = np.arange(0, 20, 0.01)
        y, _ = simulate(A, B, C, D, t, np.full_like(t, 0.1))
        self.assertAlmostEqual(float(y[-1, 0]), 0.1, places=3)

    def test_pid_tracks(self):
        t = np.arange(0, 20, 0.002)
        th, _ = pid_pitch_loop(example_aircraft(), -2.0, -0.5, -0.4, t, np.full_like(t, 0.1))
        self.assertAlmostEqual(float(th[-1]), 0.1, delta=0.005)

    def test_step_metrics_first_order(self):
        t = np.linspace(0, 10, 10001)
        s = step_metrics(t, 1 - np.exp(-t), final=1.0)
        self.assertAlmostEqual(s["rise_time"], np.log(9), delta=0.01)
        self.assertEqual(s["overshoot_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
