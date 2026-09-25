import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

dt, g, L, c = 0.05, 9.81, 1.0, 0.1   # time step, gravity, length, damping

def pendulum(t, s):
    theta, omega = s
    return [omega, -(g/L)*np.sin(theta) - c*omega]

def simulate(theta0, steps=200):
    t = np.arange(steps) * dt
    sol = solve_ivp(pendulum, (0, t[-1]), [theta0, 0.0], t_eval=t, rtol=1e-9, atol=1e-9)
    return sol.y                      # rows: theta, omega

def fit_dmd(start_angles):
    trajs = [simulate(a) for a in start_angles]
    X  = np.hstack([T[:, :-1] for T in trajs])   # "now"
    Xn = np.hstack([T[:, 1:]  for T in trajs])   # "next"
    return Xn @ np.linalg.pinv(X)                # A = X' X+

def rollout(A, x0, steps=200):
    xs = [x0]
    for _ in range(steps - 1):
        xs.append(A @ xs[-1])
    return np.array(xs).T

A_small = fit_dmd(np.linspace(-0.3, 0.3, 10))    # trained on small swings only
A_big   = fit_dmd(np.linspace(-2.5, 2.5, 10))    # trained on small + big swings

for test in [0.2, 2.0]:
    true = simulate(test)
    for name, A in [("small-swing model", A_small), ("mixed model", A_big)]:
        err = np.abs(rollout(A, true[:, 0])[0] - true[0]).max()
        print(f"test swing {test} rad | {name}: max error {err:.3f} rad")

for name, A in [("small", A_small), ("mixed", A_big)]:
    lam = np.linalg.eigvals(A)
    print(name, "eigenvalues:", lam.round(4), "| size:", np.abs(lam).round(4))

true = simulate(2.0); t = np.arange(200)*dt
plt.plot(t, true[0], label="true")
plt.plot(t, rollout(A_small, true[:, 0])[0], "--", label="DMD (small-swing model)")
plt.xlabel("time [s]"); plt.ylabel("angle [rad]"); plt.legend(); plt.savefig("pendulum_dmd.png")
