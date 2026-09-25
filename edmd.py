import numpy as np

def F(x):                                  # the "unknown" system (used only to make data)
    return np.array([0.9*x[0], 0.5*x[1] + x[0]**3])

# 1. collect snapshot pairs from 20 random starting points
rng = np.random.default_rng(0)
X, Y = [], []
for _ in range(20):
    x = rng.uniform(-1, 1, 2)
    for _ in range(10):
        y = F(x); X.append(x); Y.append(y); x = y
X, Y = np.array(X).T, np.array(Y).T

# 2. plain DMD
A = Y @ np.linalg.pinv(X)
print("DMD eigenvalues:", np.linalg.eigvals(A))

# 3. EDMD with dictionary (x1, x2, x1^3)
lift = lambda Z: np.vstack([Z[0], Z[1], Z[0]**3])
K = lift(Y) @ np.linalg.pinv(lift(X))
print("EDMD eigenvalues:", np.linalg.eigvals(K))