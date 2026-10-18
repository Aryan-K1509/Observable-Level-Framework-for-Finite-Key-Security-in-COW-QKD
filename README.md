# Observable-level COW-QKD Simulation

A modular simulation framework for finite-key Coherent-One-Way (COW) quantum key
distribution.

## Layout

```
core/
  observable.py       quantum simulation (Strawberry Fields)
  detector.py         detector physics
  security_bounds.py  security bounds
  framework.py        experiment orchestration and optimization
data/
  <figure>/           figure with the CSV data it is plotted from
```

## Setup

```bash
pip install -r requirements.txt
```
