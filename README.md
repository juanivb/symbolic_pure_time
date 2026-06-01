# Symbolic Pure Time

> Reference repository for the **Geometric Signal Dynamics (GSD)** research programme.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-lightgrey.svg)](./LICENSE-text.md)

**📖 Project website: [juanivb.github.io/symbolic_pure_time](https://juanivb.github.io/symbolic_pure_time)** — overview, paper index, tutorials, and live documentation.

This repository consolidates the foundational papers of the programme, the companion library, the introductory tutorial, and the applied papers that build on the foundations.

---

## Repository structure

```
symbolic_pure_time/
├── papers/         Foundational series (F-XXX): substrate, estimator theory, geometric companion
├── Applications/   Applied series (A-XXX): substrate readings of dynamical-system and physics questions
├── lib/            Reference library (SemVer-versioned)
├── tutorial/       Introductory walkthrough and notebooks
├── docs/           Library documentation
└── bibliography/   Master .bib shared by all papers
```

The two paper series serve different purposes. **`papers/F-XXX/`** establish the algebraic substrate, the SPTLS estimator, and the geometric foundation. **`Applications/A-XXX/`** apply the substrate to dynamical-system and physics questions where the data admits confirmatory validation. Browse each folder for an up-to-date list of available papers and their content.

Every paper subfolder (`F-XXX/`, `A-XXX/`) follows the same internal layout:

```
README.md         Title, abstract, plain-language summary, replication notes, DOI
paper.pdf         Compiled version
source/           LaTeX source files and figures
replication/      Code to reproduce results
data/             Datasets (when applicable)
```

---

## Citing this work

A single Zenodo record versions the entire repository. Cite the DOI of the release corresponding to the paper version you are referencing.

---

## AI assistance

All manuscripts of the GSD programme were developed with extensive assistance from Claude (Anthropic). See [`AI_USE_STATEMENT.pdf`](./AI_USE_STATEMENT.pdf) for the formal declaration, which is also attached to every submission to a journal.

---

## Acknowledgments

I am deeply indebted to the partners and institutions that made this research possible. Formal acknowledgments are deferred to the final version of this work to ensure all contributors are properly recognized. The views expressed herein are the author's alone and do not imply endorsement by any supporting entity.

---

## Get in touch

Questions, ideas, or collaboration? Feel free to reach out.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Get%20in%20touch-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jivb/)

— Juan Ignacio Vázquez Broquá ([ORCID 0009-0008-7156-3093](https://orcid.org/0009-0008-7156-3093))
