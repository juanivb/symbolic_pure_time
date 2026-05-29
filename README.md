# Symbolic Pure Time

A research programme in **Geometric Signal Dynamics (GSD)**: a single
algebraic substrate, $A \otimes \mathrm{Cl}(3,0)$, on which time-series
dynamics are read as motion of a kinematic jet $q(t) = (z, \Delta z,
\Delta^2 z)$, and a single estimator — **Symbolic Pure-Time Least Squares
(SPTLS)** — that fits the one-step map of that jet as a non-abelian operator,
read by polar decomposition into a rotor (orientation) and a stretch (gain).

This repository is the public home of the programme: the reference
implementation library, pedagogical notebooks, the foundational papers, and
selected applied papers, each with its replication code.

## Contents

```
lib/gsd/            reference implementation (numpy): OLS and SPTLS,
                    a white-box diagnostic suite, walk-forward validation
tutorial/           worked notebooks (start with tutorial/ols_vs_sptls/)
papers/             foundational papers (PDF + replication code)
Applications/       applied papers (PDF + replication code)
```

## The library in one minute

```python
import gsd
y = gsd.datasets.rotation()
print(gsd.OLS(2).fit(y).summary())     # AR(2) regression table
print(gsd.SPTLS().fit(y).report().summary())   # phase-space reading
```

`SPTLS().fit(y).report()` is the distinctive piece — the white-box reading of
the fitted operator (rotor angle, stretch gains, operator spectrum, grade
energy, and, for several series, a directed cross-effect reading and a shared-
direction reading of the joint embedding). See `lib/README.md`.

## Citing this work

Please cite the programme through its archived release (a single Zenodo DOI
versioned per release); see `CITATION.cff`. When linking to the code or
replication material, reference this repository:
`https://github.com/juanivb/symbolic_pure_time`.

## Acknowledgments

I am deeply indebted to the partners and institutions that made this research
possible. Formal acknowledgments are deferred to the final version of this
work to ensure all contributors are properly recognized. The views expressed
herein are the author's alone and do not imply endorsement by any supporting
entity.

## Get in touch

[![Get in touch](https://img.shields.io/badge/Get%20in%20touch-LinkedIn-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/jivb/)

---

Juan Ignacio Vázquez Broquá · ORCID [0009-0008-7156-3093](https://orcid.org/0009-0008-7156-3093)
