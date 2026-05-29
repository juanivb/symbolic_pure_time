# A-000 — Replication

Self-contained replication code and pre-computed results for the empirical
prelude of A-000 (the chaos catalogue and the econometric catalogue reported
in the paper).

## Layout

```
replication/
├── scripts/
│   ├── ptls.py                          core Pure-Time Least Squares module
│   ├── action_graded_multivariate.py    bundled estimator helpers (frozen with
│   ├── action_graded_universal.py         the paper, so the numbers reproduce
│   ├── action_graded_quadratic.py         exactly)
│   ├── action_graded_mv_quadratic.py
│   ├── st_lorenz_recovery.py            Lorenz '63 identification (chaos)
│   ├── st_universal_battery.py          canonical chaos zoo battery
│   ├── st_multivariate_battery.py       multivariate VAR/VECM regimes
│   ├── st_panel_forecast.py             one-step panel forecast battery
│   ├── st_real_cointegration.py         real-data cointegration recovery
│   ├── st_real_extended.py              extended real-data run
│   ├── symbolic_generate_zoo_figures.py figures for the chaos zoo
│   ├── symbolic_generate_figures.py     general figure generator
│   └── walkthrough_*.py                 per-system walkthroughs (3-body,
│                                          double pendulum, pendulum, SINDy)
└── results/
    ├── *.csv                            tabular outputs from each script
    ├── *.txt                            console logs (recovered values, metrics)
    └── walkthrough_*.json               trajectory data and recovered coefficients
```

## Running

```bash
cd scripts/
python3 st_lorenz_recovery.py     # chaos catalogue, Lorenz '63
python3 st_universal_battery.py   # canonical chaos zoo battery
python3 st_multivariate_battery.py
python3 st_panel_forecast.py
python3 st_real_cointegration.py  # needs pandas
```

Outputs are written to `../results/`. The figures are produced by
`symbolic_generate_zoo_figures.py` and `symbolic_generate_figures.py`, which
read from `../results/`.

## Dependencies

- `numpy` — required by all scripts.
- `pandas` — required by the real-data scripts (`st_real_*`).
- `matplotlib` — required by the figure generators and `walkthrough_*`.

```bash
pip install -r requirements.txt
```

## Note

This folder is a **frozen reproduction artifact**: the estimator code bundled
here (`ptls.py`, `action_graded_*.py`) is the exact code that produced the
numbers in the paper, kept alongside the results so they reproduce bit-for-bit.
The forward-facing, maintained interface for new work is the `gsd` library at
`../../../lib/gsd`.
