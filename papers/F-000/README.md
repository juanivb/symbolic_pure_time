# Symbolic Pure Time

Foundational paper of the programme. Introduces the algebra
$\mathcal{A} \otimes \mathrm{Cl}(3,0)$ ($\mathcal{A}$ = Weyl, $\mathrm{Cl}(3,0)$ = real spatial
Clifford) and Symbolic Pure-Time Least Squares (SPTLS): an ordinary
least-squares fit of the one-step transition of the kinematic jet
$\mathbf{q}(t) = (y_t, \Delta y_t, \Delta^2 y_t)$. The estimator
returns an operator $\hat{M}$ rather than a vector of scalar
coefficients; the operator admits a polar decomposition into a rotor
(the genuinely non-abelian, rotational content of the dynamics) and
a stretch (the abelian gains an ordinary lag model sees), and a
Clifford-grade decomposition into scalar, bivector and pseudoscalar
pieces.

## Paper

- **PDF:** [`symbolic_pure_time.pdf`](symbolic_pure_time.pdf)
- **LaTeX source:** [`source/symbolic_pure_time.tex`](source/symbolic_pure_time.tex)
- **Zenodo DOI:** [10.5281/zenodo.20298431](https://doi.org/10.5281/zenodo.20298431)

## What is in this folder

```
F-000/
├── README.md                       ← this file
├── symbolic_pure_time.pdf          ← compiled paper
├── source/
│   └── symbolic_pure_time.tex      ← LaTeX source
├── replication/
│   ├── build_case_figures.py       ← regenerates §5 figures (logistic + mixed VAR)
│   ├── build_case_table.py         ← regenerates the §5 quality-of-fit table
│   ├── build_horizon_table.py      ← regenerates the multi-horizon forecast table
│   ├── appendix_ergodicity_test.py ← regenerates the ergodicity-test appendix
│   └── appendix_numbers.py         ← regenerates the small numbers cited in the appendix
├── figures/                        ← PNG figures used by §5
│   ├── case_chaotic.png            ← Case 1: logistic map (phase portrait + residuals)
│   └── case_mixed_var.png          ← Case 2: mixed multivariate VAR (trajectories + scatter)
└── results/                        ← CSV / TeX result tables
    ├── case_fit_table.{csv,tex}    ← §5 quality-of-fit table
    ├── case_horizon_table.{csv,tex}← multi-horizon forecast table
    └── appendix_ergodicity_test.csv← ergodicity-test summary
```

## Replication

The replication scripts depend only on the reference NumPy implementation
shipped in the repository root (`gsd/`). To rebuild every figure, table
and number cited in the paper:

```
# §5 case figures and tables
python replication/build_case_figures.py
python replication/build_case_table.py
python replication/build_horizon_table.py

# Appendix ergodicity diagnostic
python replication/appendix_ergodicity_test.py
python replication/appendix_numbers.py
```

All scripts are deterministic (fixed seeds) and run end-to-end on
numpy / scipy / pandas / matplotlib only.

## Worked examples in §5

The paper develops two worked examples, each compared against the
corresponding best-practice OLS-with-feature-engineering workflow:

1. **Case 1 — deterministic chaos (logistic map).** AR(2), OLS with
   quadratic features, and $\mathrm{SPT}_2$ are all compared on $R^2$,
   residual autocorrelation, and the structural reading of $\hat M$.
   The fit qualities are matched; the differential is the polar +
   grade reading of the fitted operator.
2. **Case 2 — mixed multivariate VAR.** Three series with one linear
   and one quadratic cointegration plus lags. The classical
   econometric workflow needs five separate procedures (unit-root,
   lag-order, Johansen rank, VECM, non-linearity); the joint SPTLS
   solve reads all of them off the block structure of $\hat M$ and
   its cross-bivector channel in a single fit.

## License

Code: MIT. Paper text: CC-BY 4.0.
