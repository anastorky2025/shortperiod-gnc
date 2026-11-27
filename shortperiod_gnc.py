"""shortperiod-gnc: aircraft short-period dynamics and pitch autopilot design (LQR and PID).

Linear short-period longitudinal model, states x = [alpha, q, theta]
(angle of attack, pitch rate, pitch angle; rad, rad/s, rad), input u = elevator [rad]:

    alpha_dot = Z_alpha * alpha + q + Z_de * u
    q_dot     = M_alpha * alpha + M_q * q + M_de * u
    theta_dot = q
"""
import argparse
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_are

__version__ = "0.1.0"


@dataclass(frozen=True)
class ShortPeriod:
    Z_alpha: float
    Z_de: float
    M_alpha: float
    M_q: float
    M_de: float

    def matrices(self):
        """Return (A, B, C, D) with states [alpha, q, theta], output theta."""
        A = np.array([[self.Z_alpha, 1.0, 0.0],
                      [self.M_alpha, self.M_q, 0.0],
                      [0.0, 1.0, 0.0]])
        B = np.array([[self.Z_de], [self.M_de], [0.0]])
        C = np.array([[0.0, 0.0, 1.0]])
        D = np.zeros((1, 1))
        return A, B, C, D

    def short_period_poles(self):
        """Eigenvalues of the 2x2 alpha/q block."""
        A2 = np.array([[self.Z_alpha, 1.0], [self.M_alpha, self.M_q]])
        return np.linalg.eigvals(A2)

    def natural_frequency_damping(self):
        """Return (omega_n [rad/s], zeta) of the short-period mode."""
        det = self.Z_alpha * self.M_q - self.M_alpha
        tr = self.Z_alpha + self.M_q
        wn = float(np.sqrt(det))
        return wn, float(-tr / (2 * wn))


def example_aircraft():
    """Illustrative coefficients, not a specific real aircraft."""
    return ShortPeriod(Z_alpha=-1.0, Z_de=-0.1, M_alpha=-8.0, M_q=-2.0, M_de=-12.0)

def lqr(A, B, Q, R):
    """Continuous-time LQR gain K (u = -K x) and Riccati solution P."""
    A, B, Q, R = map(np.atleast_2d, (A, B, Q, R))
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def closed_loop_lqr(model, Q=None, R=None):
    """LQR state feedback with a reference prefilter for pitch tracking.

    Returns (A_cl, B_cl, C, D, K). The closed-loop input is theta_ref [rad].
    """
    A, B, C, D = model.matrices()
    Q = np.diag([1.0, 1.0, 50.0]) if Q is None else Q
    R = np.array([[1.0]]) if R is None else R
    K, _ = lqr(A, B, Q, R)
    A_cl = A - B @ K
    # state feedback u = -K x + N*r. Plant has an integrator on theta, so use
    # u = -K (x - x_ref) with x_ref = [0, 0, r]: equivalent to B_cl = B * K[:,2].
    B_cl = B @ K[:, 2:3]
    return A_cl, B_cl, C, D, K


def simulate(A, B, C, D, t, u, x0=None):
    """Simulate a linear system with piecewise-constant input (exact discretisation)."""
    t = np.asarray(t, float)
    n, m = B.shape
    x = np.zeros(n) if x0 is None else np.asarray(x0, float)
    ys, xs = [], []
    cache = {}
    for k in range(len(t)):
        uk = np.atleast_1d(u[k])
        ys.append((C @ x + D @ uk).ravel())
        xs.append(x.copy())
        if k + 1 < len(t):
            dt = round(float(t[k + 1] - t[k]), 12)
            if dt not in cache:
                M = np.zeros((n + m, n + m))
                M[:n, :n] = A
                M[:n, n:] = B
                cache[dt] = expm(M * dt)
            E = cache[dt]
            x = E[:n, :n] @ x + E[:n, n:] @ uk
    return np.array(ys), np.array(xs)


def pid_pitch_loop(model, kp, ki, kd, t, theta_ref, N=50.0):
    """Closed-loop pitch response with a PID on theta error driving the elevator.

    Fixed-step Euler plant and a filtered derivative (filter coefficient N).
    Gains are negative because positive elevator gives negative pitch moment here.
    Returns (theta, elevator).
    """
    A, B, C, D = model.matrices()
    t = np.asarray(t, float)
    x = np.zeros(3)
    integ = 0.0
    d_state = 0.0
    prev_e = 0.0
    th, de = [], []
    for k in range(len(t)):
        e = theta_ref[k] - x[2]
        if k:
            dt = t[k] - t[k - 1]
            integ += e * dt
            d_state += dt * N * ((e - prev_e) / dt - d_state)
        u = kp * e + ki * integ + kd * d_state
        prev_e = e
        th.append(x[2]); de.append(u)
        if k + 1 < len(t):
            h = t[k + 1] - t[k]
            x = x + h * (A @ x + (B * u).ravel())
    return np.array(th), np.array(de)


def step_metrics(t, y, final=None, band=0.02):
    """Rise time (10-90%), overshoot [%], settling time (+/- band) and final value."""
    t = np.asarray(t, float); y = np.asarray(y, float).ravel()
    yf = y[-1] if final is None else float(final)
    if abs(yf) < 1e-12:
        raise ValueError("final value is zero")
    r = y / yf
    def first(level):
        idx = np.nonzero(r >= level)[0]
        return t[idx[0]] if idx.size else float("nan")
    rise = first(0.9) - first(0.1)
    overshoot = max(0.0, (r.max() - 1.0) * 100.0)
    out = np.nonzero(np.abs(r - 1.0) > band)[0]
    settle = 0.0 if out.size == 0 else (t[out[-1] + 1] if out[-1] + 1 < len(t) else float("nan"))
    return {"rise_time": float(rise), "overshoot_pct": float(overshoot),
            "settling_time": float(settle), "final_value": float(yf)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Short-period pitch autopilot demo")
    p.add_argument("--theta-deg", type=float, default=5.0, help="pitch step command [deg]")
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--plot", metavar="PNG", help="save comparison plot to this file")
    a = p.parse_args(argv)

    m = example_aircraft()
    wn, zeta = m.natural_frequency_damping()
    print(f"open-loop short period: wn={wn:.2f} rad/s zeta={zeta:.2f}")
    t = np.arange(0, a.duration, 0.005)
    ref = np.full_like(t, np.deg2rad(a.theta_deg))

    A, B, C, D, K = closed_loop_lqr(m)
    y_lqr = simulate(A, B, C, D, t, ref)[0].ravel()
    y_pid, _ = pid_pitch_loop(m, kp=-2.0, ki=-0.5, kd=-0.4, t=t, theta_ref=ref)
    for name, y in (("LQR", y_lqr), ("PID", y_pid)):
        s = step_metrics(t, y, final=ref[0])
        settle = "n/a" if np.isnan(s["settling_time"]) else f"{s['settling_time']:.2f}s"
        print(f"{name}: rise {s['rise_time']:.2f}s overshoot {s['overshoot_pct']:.1f}% "
              f"settle {settle}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(t, np.rad2deg(ref), "k--", label="command")
        plt.plot(t, np.rad2deg(y_lqr), label="LQR")
        plt.plot(t, np.rad2deg(y_pid), label="PID")
        plt.xlabel("time [s]"); plt.ylabel("pitch angle [deg]")
        plt.title("Pitch step response, example short-period model")
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(a.plot, dpi=130)
        print("saved", a.plot)


if __name__ == "__main__":
    main()
