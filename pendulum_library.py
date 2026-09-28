
# Shared building blocks for the pendulum experiments.
# This file contains ONLY functions (nothing runs when you import it).
#
# Usage in another file:
#   from pendulum_lib import update, simulate_with_control, fit_dmdc, predicted_dmdc

import numpy as np

dt = 0.1          # time step [s]
g_over_L = 9.81   # gravity / length (L = 1 m)


def update(s, u=0.0):
    """One step of the REAL (nonlinear) pendulum, semi-implicit Euler.
    s = [theta, omega], u = push (angular acceleration from the motor)."""
    theta, omega = s
    new_omega = omega - dt * g_over_L * np.sin(theta) + dt * u
    new_theta = theta + dt * new_omega
    return np.array([new_theta, new_omega])


def simulate(theta0, omega0, steps=100):
    """Pendulum with no push. Returns states, shape (steps, 2)."""
    s = np.array([theta0, omega0])
    states = [s]
    for _ in range(steps - 1):
        s = update(s)
        states.append(s)
    return np.array(states)


def simulate_with_control(theta0, omega0, steps=100, u_range=1.0):
    """Pendulum with a new RANDOM push every step (training data for DMDc).
    Returns states (steps, 2) and inputs (steps-1,)."""
    s = np.array([theta0, omega0])
    states = [s]
    inputs = []
    for _ in range(steps - 1):
        u = np.random.uniform(-u_range, u_range)
        inputs.append(u)
        s = update(s, u)
        states.append(s)
    return np.array(states), np.array(inputs)


def fit_dmdc(states, inputs):
    """DMDc: find A and B so that  x_next ≈ A x + B u  (least squares)."""
    X = states[:-1].T                    # (2, N)  now
    X_dash = states[1:].T                # (2, N)  next
    U = inputs.reshape(1, -1)            # (1, N)  pushes
    Omega = np.vstack((X, U))            # (3, N)  now + pushes stacked
    G = X_dash @ np.linalg.pinv(Omega)   # (2, 3)  = [A | B]
    A = G[:, :2]
    B = G[:, 2:]
    return A, B


def predicted_dmdc(A, B, initial_state, inputs):
    """Roll the learned model forward using a given list of pushes."""
    x = np.asarray(initial_state, dtype=float)
    predicted = [x]
    for u in inputs:
        x = A @ x + B @ np.array([u])
        predicted.append(x)
    return np.array(predicted)