# Model Predictive Control Basics
# it predicts best sequence of moves that get to target position or reduce the cost, and then only executes first step of it and then re-predicts again using real updated state, and so on

#we will optimize eqn, cost = Q.(dist from target)^2 + R.(push effort)^2

import cvxpy as cp
import numpy as np

u = cp.Variable()  # control input variable
Q = 1.0
R = 3.0
x = 4.0
target = 1.0
cost = Q * cp.square(x + u - target) + R * cp.square(u)

prob = cp.Problem(cp.Minimize(cost), [cp.abs(u) <= 1])
prob.solve()
print(u.value)
# this above checks only for one step ahead, now we go for H steps ahead, which actually happens in practice

def mpc_step(x, target, H=5, Q=1.0, R=1.0, u_max=1.0):
    # 1. create u as a Variable with H entries
    u = cp.Variable(H)  # control input variable for H steps

    # 2. build the cost with the loop above
    cost = 0
    for h in range(H):
        cost += Q * cp.square(x + cp.sum(u[:h + 1]) - target) + R * cp.square(u[h])
    # 3. solve with the limit  cp.abs(u) <= u_max
    prob = cp.Problem(cp.Minimize(cost), [cp.abs(u) <= u_max])
    prob.solve()
    # 4. return u.value[0]   <- only the FIRST push
    return u.value[0]

print("mpc_step:", mpc_step(4.0,0.0))


import matplotlib.pyplot as plt  

# ---------- the full MPC loop (receding horizon) ----------
x = 4.0            # current (real) state
target = 0.0
steps = 15

history_x = [x]
history_u = []

for k in range(steps):
    u = mpc_step(x, target)          # 1. plan H steps ahead, keep only the FIRST push
    x = x + u                        # 2. apply it to the "real" system
    history_x.append(x)              # 3. record, then repeat from the new real state
    history_u.append(u)
    print(f"step {k:2d}: push u = {u:6.3f}  ->  x = {x:6.3f}")

# ---------- plot: position (top) and pushes (bottom) ----------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(history_x, "o-")
ax1.axhline(target, color="gray", linestyle="--", label="target")
ax1.set_ylabel("x")
ax1.legend()

ax2.step(range(steps), history_u, where="post")
ax2.axhline(-1.0, color="red", linestyle=":", label="push limit")
ax2.axhline(1.0, color="red", linestyle=":")
ax2.set_ylabel("push u")
ax2.set_xlabel("step")
ax2.legend()

plt.tight_layout()
plt.show()