# koopman_mpc.py
# Koopman (DMDc) MPC on the pendulum:
#   1. collect data with random pushes
#   2. learn a LINEAR model  x_next = A x + B u   (DMDc)
#   3. use that model inside MPC to choose pushes that stop the swinging pendulum
#   4. apply each push to the REAL nonlinear pendulum, re-plan every step

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from pendulum_lib import dt, update, simulate, simulate_with_control, fit_dmdc

np.random.seed(0)   # same random pushes every run -> repeatable results


# ---------------------------------------------------------------
# MPC: plan H pushes with the learned model, return only the first
# ---------------------------------------------------------------
def mpc_step_pendulum(x, A, B, target, H=10, Q=np.diag([10.0, 1.0]), R=0.1, u_max=2.0):
    u = cp.Variable(H)                     # the H pushes we want the solver to find
    x_pred = x                             # prediction starts from the REAL current state
    cost = 0
    for k in range(H):
        x_pred = A @ x_pred + B[:, 0] * u[k]                               # learned model, one step
        cost += cp.quad_form(x_pred - target, Q) + R * cp.square(u[k])     # distance + effort
    problem = cp.Problem(cp.Minimize(cost), [cp.abs(u) <= u_max])          # push limit
    problem.solve()
    return u.value[0]                      # receding horizon: use only the first push


# ---------------------------------------------------------------
# 1. Learn A and B from data (random pushes)
# ---------------------------------------------------------------
train_states, train_inputs = simulate_with_control(0.5, 0.0, steps=100)
A, B = fit_dmdc(train_states, train_inputs)
print("Learned A:\n", A)
print("Learned B:\n", B)

# ---------------------------------------------------------------
# 2. MPC loop: stop a pendulum swinging from 1.0 rad
# ---------------------------------------------------------------
start = np.array([1.0, 0.0])
target = np.array([0.0, 0.0])      # hanging still
steps = 60                         # 6 seconds

x = start
history_x = [x]
history_u = []
for k in range(steps):
    u = mpc_step_pendulum(x, A, B, target)   # plan with the LEARNED model
    x = update(x, u)                         # apply to the REAL pendulum
    history_x.append(x)
    history_u.append(u)

history_x = np.array(history_x)

# for comparison: the same pendulum with no control at all
no_control = simulate(start[0], start[1], steps=steps + 1)

print(f"final angle with MPC:   {history_x[-1, 0]:+.4f} rad")
print(f"final angle, no control: {no_control[-1, 0]:+.4f} rad")

# ---------------------------------------------------------------
# 3. Plot: angle (top) and pushes (bottom)
# ---------------------------------------------------------------
t = np.arange(steps + 1) * dt
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

ax1.plot(t, no_control[:, 0], color="gray", alpha=0.6, label="no control")
ax1.plot(t, history_x[:, 0], label="Koopman MPC")
ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax1.set_ylabel("angle θ [rad]")
ax1.set_title("Stopping a swinging pendulum with Koopman (DMDc) MPC")
ax1.legend()

ax2.step(t[:-1], history_u, where="post", label="push u")
ax2.axhline(2.0, color="red", linestyle=":", label="push limit")
ax2.axhline(-2.0, color="red", linestyle=":")
ax2.set_ylabel("push u")
ax2.set_xlabel("time [s]")
ax2.legend()

plt.tight_layout()
plt.show()