import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
theta = 0.5
omega = 0.0
g_over_L = 9.81

def update(s,u=0.0):
    theta, omega = s
    new_omega = omega - dt*g_over_L*np.sin(theta) + dt*u
    new_theta = theta + dt*new_omega
    return np.array([new_theta, new_omega])

def simulate_with_control(theta0, omega0, steps=100):
    old_s = np.array([theta0, omega0])
    states = [old_s]
    inputs = []

    for _ in range(steps - 1):
        u = np.random.uniform(-1, 1)  # Random control input between -1 and 1
        inputs.append(u)
        new_s = update(old_s, u)
        states.append(new_s)
        old_s = new_s

    states = np.array(states)
    inputs = np.array(inputs)
    return states, inputs

def fit_dmdc(states, inputs):
    X = states[:-1].T
    X_dash = states[1:].T
    U = inputs.reshape(1, -1)  # Reshape inputs to match dimensions
    X_and_U = np.vstack((X, U))  # Stack states and inputs
    psuedo_inverse_X_and_U = np.linalg.pinv(X_and_U)
    G = X_dash @ psuedo_inverse_X_and_U
    A = G[:, :2]  # Extract A matrix
    B = G[:, 2:]  # Extract B matrix
    return A, B

# states, inputs = simulate_with_control(0.5, 0.0, steps=100)
# print("states:", states.shape)
# print("inputs:", inputs.shape)
# A, B = fit_dmdc(states, inputs)
# print("A:\n", A)
# print("B:\n", B)

#Testing
def predicted_dmdc(A, B, initial_state, inputs):
    predicted_states = [initial_state]
    current_state = initial_state

    for u in inputs:
        # A @ state -> shape (2,),  B @ [u] -> shape (2,)  => both flat, adds cleanly
        new_state = A @ current_state + B @ np.array([u])
        predicted_states.append(new_state)
        current_state = new_state

    return np.array(predicted_states)


# ---------- 1. train: learn A and B from one run ----------
states, inputs = simulate_with_control(0.5, 0.0, steps=100)
A, B = fit_dmdc(states, inputs)
print("A:\n", A)
print("B:\n", B)

# ---------- 2. test: a NEW run (different start, new random pushes) ----------
test_angle = 0.3
test_states, test_inputs = simulate_with_control(test_angle, 0.0, steps=100)

# the model gets ONLY the starting state + the pushes, and must predict the rest
pred = predicted_dmdc(A, B, test_states[0], test_inputs)
print("pred shape:", pred.shape)

error = np.max(np.abs(pred[:, 0] - test_states[:, 0]))
print(f"max angle error: {error:.4f} rad")

# ---------- 3. plot true vs predicted ----------
t = np.arange(len(test_states)) * dt
plt.plot(t, test_states[:, 0], label="true")
plt.plot(t, pred[:, 0], "--", label="DMDc prediction")
plt.xlabel("time [s]")
plt.ylabel("angle [rad]")
plt.title(f"DMDc test: start {test_angle} rad, new random pushes")
plt.legend()
plt.show()
