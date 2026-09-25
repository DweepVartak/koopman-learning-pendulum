import numpy as np
import matplotlib.pyplot as plt  

dt = 0.1
theta0 = 0.5
omega0 = 0.0
g_over_L = 9.81

def update(s):
    theta, omega = s
    new_omega = omega - dt*g_over_L*np.sin(theta)
    new_theta = theta + dt*new_omega
    return np.array([new_theta, new_omega])

def simulate(theta0, omega0, steps=100):
    old_s = np.array([theta0, omega0])
    states = [old_s]

    for _ in range(steps - 1):
        new_s = update(old_s)
        states.append(new_s)
        old_s = new_s

    states = np.array(states)
    return states


# print("all states shape:", states.shape)
#print("all states:", states)
# print("max angle:", np.max(np.abs(states[:, 0])))


# states = [theta, omega, np.sin(theta), np.cos(theta)]
# take base two states and create sin and cos theta
def add_features(states):
    theta = states[:, 0]
    omega = states[:, 1]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    new_states = np.column_stack((theta, omega, sin_theta, cos_theta))
    return new_states



def fit_dmd(states):
    X = states[:-1].T
    X_dash = states[1:].T
    psuedo_inverse_X = np.linalg.pinv(X)
    A = X_dash @ psuedo_inverse_X
    return A

# print("DMD eigenvalues:", np.linalg.eigvals(A))
# print("DMD matrix A:", A)
# print("X shape:", X.shape)
# print("X_dash shape:", X_dash.shape)

#now predict the next 100 states using DMD matrix A, and only initial state will be given
def predicted_dmd(A, initial_state, steps=100):            
    predicted_states = [initial_state]

    for _ in range(steps - 1):
        new_state = A @ predicted_states[-1]
        predicted_states.append(new_state)  

    predicted_states = np.array(predicted_states)
    return predicted_states
#print("predicted states ", predicted_states)

#now comparing with plot

# # 1. learn from the 0.5 swing only
# train_states = simulate(0.5, 0.0, steps=100)
# A = fit_dmd(train_states)

# # 2. test on a swing DMD has never seen
# test_angle = 1
# test_states = simulate(test_angle, 0.0, steps=100)          # the true answer
# predicted_states = predicted_dmd(A, [test_angle, 0.0], steps=100)  # DMD's guess, same start


# # 4. plot truth vs prediction for the SAME swing
# t = np.arange(len(test_states)) * dt
# plt.plot(t, test_states[:, 0], label="true")
# plt.plot(t, predicted_states[:, 0], "--", label="DMD prediction")
# plt.xlabel("time [s]"); plt.ylabel("angle [rad]"); plt.legend()
# plt.show()


# edmd_states = add_features(simulate(0.5, 0.0, steps=100))
# print(edmd_states)
# print("EDMD states shape:", edmd_states.shape)

#now try fit and test for edmd and also new A will be trained
# # 1. learn from the 0.5 swing only
# train_states = add_features(simulate(5, 0.0, steps=100))
# A = fit_dmd(train_states)

# # 2. test on a swing DMD has never seen
# test_angle = 5.1
# test_states = add_features(simulate(test_angle, 0.0, steps=100))          # the true answer
# first_lifted = add_features(test_states[:1])[0]
# predicted_states = predicted_dmd(A, first_lifted, steps=100)


# # 4. plot truth vs prediction for the SAME swing
# t = np.arange(len(test_states)) * dt
# plt.plot(t, test_states[:, 0], label="true")
# plt.plot(t, predicted_states[:, 0], "--", label="DMD prediction")
# plt.xlabel("time [s]"); plt.ylabel("angle [rad]"); plt.legend()
# plt.show()

# ---------- compare plain DMD vs EDMD ----------

# # 1. train both models on the SAME 0.5 swing
# train_states = simulate(0.5, 0.0, steps=100)
# A_dmd  = fit_dmd(train_states)                  # uses [theta, omega]           -> 2 x 2
# A_edmd = fit_dmd(add_features(train_states))    # uses [theta, omega, sin, cos] -> 4 x 4
# print("A_dmd shape:", A_dmd.shape, "| A_edmd shape:", A_edmd.shape)

# # 2. test both on swings they have never seen
# print(f"{'test angle':>10} | {'DMD error %':>11} | {'EDMD error %':>12}")

# for test_angle in [0.2, 1.0, 2.0]:
#     test_states = simulate(test_angle, 0.0, steps=100)   # the true answer
#     true_theta = test_states[:, 0]

#     # predictions: each model starts from the same first state (EDMD needs it lifted)
#     pred_dmd  = predicted_dmd(A_dmd,  test_states[0], steps=100)
#     pred_edmd = predicted_dmd(A_edmd, add_features(test_states[:1])[0], steps=100)

#     # max error as % of the swing size (compare column 0 = theta)
#     err_dmd  = np.max(np.abs(pred_dmd[:, 0]  - true_theta)) / test_angle * 100
#     err_edmd = np.max(np.abs(pred_edmd[:, 0] - true_theta)) / test_angle * 100
#     print(f"{test_angle:>10} | {err_dmd:>11.1f} | {err_edmd:>12.1f}")

#     # 3. plot the last test (2.0) so you can see both predictions
#     if test_angle == 2.0:
#         t = np.arange(len(test_states)) * dt
#         plt.plot(t, true_theta, label="true")
#         plt.plot(t, pred_dmd[:, 0], "--", label="plain DMD")
#         plt.plot(t, pred_edmd[:, 0], ":", label="EDMD (sin, cos)")
#         plt.xlabel("time [s]"); plt.ylabel("angle [rad]")
#         plt.title(f"test angle = {test_angle}")
#         plt.legend(); plt.show()
from matplotlib.animation import FuncAnimation

# ---------- animate: true vs DMD vs EDMD ----------
train_angle = 5      # what the models learn from
test_angle  = 5.1      # the swing they have never seen
L = 1.0

# 1. train both models
train_states = simulate(train_angle, 0.0, steps=100)
A_dmd  = fit_dmd(train_states)
A_edmd = fit_dmd(add_features(train_states))

# 2. true test swing + both predictions (same starting state)
test_states = simulate(test_angle, 0.0, steps=100)
pred_dmd    = predicted_dmd(A_dmd,  test_states[0], steps=100)
pred_edmd   = predicted_dmd(A_edmd, add_features(test_states[:1])[0], steps=100)

# 3. the three pendulums to draw: (name, theta over time, colour)
pendulums = [
    ("true",      test_states[:, 0], "black"),
    ("plain DMD", pred_dmd[:, 0],    "tab:red"),
    ("EDMD",      pred_edmd[:, 0],   "tab:blue"),
]

# 4. set up the picture, with one rod + bob per pendulum
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")
ax.set_title(f"trained on {train_angle} rad, tested on {test_angle} rad")
ax.plot(0, 0, "k+")                                   # the pivot

drawn = []
for name, theta, color in pendulums:
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    rod, = ax.plot([], [], "-", lw=2, color=color, alpha=0.7)
    bob, = ax.plot([], [], "o", markersize=12, color=color, label=name)
    drawn.append((x, y, rod, bob))

ax.legend(loc="lower right")
time_text = ax.text(-1.1, 1.05, "")

# 5. what to draw at frame k: update all three pendulums
def draw_frame(k):
    artists = []
    for x, y, rod, bob in drawn:
        rod.set_data([0, x[k]], [0, y[k]])
        bob.set_data([x[k]], [y[k]])
        artists += [rod, bob]
    time_text.set_text(f"t = {k*dt:.1f} s")
    return artists + [time_text]

anim = FuncAnimation(fig, draw_frame, frames=len(test_states), interval=dt*1000, blit=True)
plt.show()