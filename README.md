# Learning Koopman: pendulum experiments

## What's here

- Pendulum simulator (semi-implicit Euler)
- DMD and EDMD (features: θ, ω, sin θ, cos θ)

## Findings

- Plain Euler adds energy → switched to semi-implicit Euler
- DMD eigenvalue: size ≈ 1 (no energy loss), 17.7°/step → period ≈ 2.03 s
- Trained on 0.5 rad, tested on other swings:

| Test angle | DMD error % | EDMD error % |
| ---------- | ----------- | ------------ |
| 0.2        | 40.4        | 38.7         |
| 1.0        | 136.9       | 129.0        |
| 2.0        | 201.3       | 185.3        |

- At 5.0 rad, EDMD wins big because it gets an offset (cos θ ≈ 1), not because of sin/cos themselves

## Next

- DMDc (add torque), then Koopman MPC
