# F-004 — *Probability as a Time Series*

Replication materials for the paper *Probability as a Time Series* by
J.~I.~Vázquez Broquá (May 2026 preprint). The paper PDF lives at the
programme's project website at
<https://juanivb.github.io/symbolic_pure_time/>; this folder hosts the
code and numerical artefacts used to reproduce every experiment in the
paper.

## What is in this folder

```
F-004/
├── README.md                 ← you are here
├── experiments/              ← 23 reproducible Python scripts (one per Exp.)
├── notebooks/                ← 3 pedagogical Jupyter notebooks
├── figures/                  ← rendered PNGs cited in the paper
└── results/                  ← per-experiment CSV readouts + markdown notes
```

## Reproducing the paper's claims

Every numerical claim in the paper is reproducible from a single Python
script in `experiments/`. The scripts have no external state and fix
their random seeds explicitly. Dependencies are `numpy`, `scipy` and
`matplotlib` only; the empirical reading does not use any specialised
library, statistical or geometric.

### The 23 experiments

| Paper label | Script                                  | What it shows                                              |
|:------------|:----------------------------------------|:-----------------------------------------------------------|
| Exp. 1      | `E00_fair_coin.py`                      | Fair coin trajectory, count of trajectories ↔ Binomial     |
| Exp. 2      | `E01_biased_coin.py`                    | Biased coin, rate of approach to the Gaussian              |
| Exp. 3      | `E02_six_sided_die.py`                  | Fair die, kurtosis-driven approach                         |
| Exp. 4      | `E03_rare_event_coin.py`                | Rare events $p = 0.01$, path to Poisson                    |
| Exp. 5      | `E04_urn_without_replacement.py`        | Urn without replacement, finite-population dependence      |
| Exp. 6      | `E05_sticky_coin.py`                    | Markov coin $q = 0.7$, operator displacement               |
| Exp. 7      | `E06_gaussian_walk.py`                  | Gaussian random walk                                       |
| Exp. 8      | `E07_sum_of_squared_gaussians.py`       | $\chi^2_n$ from sum of squared standard normals            |
| Exp. 9      | `E08_sum_of_exponentials.py`            | Gamma from sum of exponentials                             |
| Exp. 10     | `E09_log_normal_product.py`             | Log-normal as exponentiated random walk                    |
| Exp. 11     | `E10_cauchy_sum.py`                     | Cauchy random walk (heavy tail, no moments)                |
| Exp. 12     | `E11_ar1_gaussian.py`                   | AR(1) Gaussian trajectory                                  |
| Exp. 13     | `E12_ar2_gaussian.py`                   | AR(2) Gaussian, two-step memory                            |
| Exp. 14     | `E13_t_student_sum.py`                  | Student-$t_3$ sum, partial-moments regime                  |
| Exp. 15     | `E14_stable_alpha_1p5_sum.py`           | Symmetric $\alpha$-stable, $\alpha = 1.5$                  |
| Exp. 16     | `E15_sptls_on_ar1_jet.py`               | SPTLS operator recovers $\phi$ from AR(1) trajectory       |
| Exp. 17     | `E16_sptls_on_ar2_jet.py`               | SPTLS on AR(2) jet                                         |
| Exp. 18     | `E17_sptls_on_random_walk.py`           | SPTLS on the unit-root Gaussian walk                       |
| Exp. 19     | `E18_sptls_on_heavy_tail_walks.py`      | SPTLS on heavy-tail walks (robustness)                     |
| Exp. 20     | `E19_sptls_on_lognormal.py`             | SPTLS on log-normal: additive vs multiplicative reading    |
| Exp. 21     | `E20_forecast_comparison.py`            | Forecast comparison: SPTLS vs Gaussian baseline            |
| Exp. 22–23  | `E21_E22_fx_real_data.py`               | FX returns (JPY/GBP/EUR vs USD): real-data validation      |
| Exp. 24     | `E23_trajectory_entropy.py`             | Trajectory entropy $H_{\mathrm{tray}}$ vs Shannon (§13)    |

To reproduce a single experiment, e.g. Exp. 24:

```bash
cd experiments
python3 E23_trajectory_entropy.py
```

The script prints a readout to standard output and writes a CSV (where
applicable) to `../results/`.

### Figures

Final rendered figures cited in the paper live in `figures/`:

| Paper figure | File                              | What it shows                                          |
|:-------------|:----------------------------------|:-------------------------------------------------------|
| Fig. 1       | `fig01_count_vs_trajectory.png`   | Counting paths vs accumulating a step (Pascal's $n{=}4$) |
| Fig. 2       | `fig02_octant_cube.png`           | The two-coordinate octant cube                         |
| Fig. 3       | `fig03_invariants_scatter.png`    | Invariants scatter across the panel                    |
| Fig. 4       | `fig04_entropy_gap.png`           | $H_{\mathrm{Shannon}}$ vs $H_{\mathrm{tray}}$ on AR(1) |
| Fig. 5       | `fig05_ar1_vs_rw.png`             | AR(1) operator signature vs random-walk baseline       |

### Pedagogical notebooks

The three companion notebooks in `notebooks/` walk through the core
ideas at the level of a single reader running cells one by one:

- `nb01_coin_toss_operator.ipynb` — the fair coin and the one-step
  operator (companion to §§1–6).
- `nb02_other_distributions.ipynb` — extending the reading to biased
  coins, Gaussian walks, Cauchy walks, and AR(1) (companion to §§7–11).
- `nb03_entropy_revision.ipynb` — trajectory entropy vs Shannon
  (companion to §13).

Each notebook can be opened in JupyterLab or VS Code; the cell outputs
are already rendered so the notebook is readable without re-execution.

## Dependencies

```text
numpy
scipy
matplotlib
```

All scripts run on Python 3.10+ with the above three packages installed.
No GPU is required; the whole panel runs in under a minute on a single
CPU core.

## Citation

If you use this material, please cite the paper (preprint version, May
2026) and the Zenodo record of the *Symbolic Pure Time* programme. The
recommended BibTeX entry is published on the project website together
with the paper PDF.
