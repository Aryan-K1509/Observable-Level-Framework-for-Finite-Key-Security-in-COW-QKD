# Observable-level COW-QKD Simulation

A modular simulation framework for finite-key Coherent-One-Way (COW) quantum key
distribution. The physical layer is simulated with the Strawberry Fields Gaussian
backend; security evaluators act only on the resulting detector record, so
different finite-key treatments can be compared on identical data.

## Layout

```
core/
  observable.py       Layer 1  quantum simulation (Strawberry Fields) -> raw click counts
  detector.py         Layer 2  dark counts, dead time, scaling, confidence intervals
  security_bounds.py  Layer 3  Kato, Azuma-Hoeffding, Cao/Gao asymptotic bounds; attack ceilings
  framework.py        Layer 4  COWParams, run_experiment, sweeps, (mu, tB) optimizer
data/
  <figure>/           each manuscript figure (PDF/PNG) with the CSV(s) it is plotted from
```

## Setup

```bash
pip install -r requirements.txt
```

## Example

```python
import sys; sys.path.insert(0, "core")
from framework import COWParams, run_experiment

params = COWParams(distance_km=50, mu=2e-3, tB=0.3)
result = run_experiment(params, N_target=int(1e10), seed=0)
for name, r in result.bounds.items():
    print(name, r.secure, r.key_rate_per_pulse)
```

## References

- Gao et al., Opt. Express 30, 23783 (2022)
- Cao et al., Sci. Adv. (2026), arXiv:2601.06772
- Li et al., Phys. Rev. Research 6, 013022 (2024)
