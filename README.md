# Behind the Score
### Neighborhood Policy Context and Regents Exam Performance in NYC Public High Schools

**Vi Kovacevic** · M.S. Data Analysis & Visualization · CUNY Graduate Center · 2026
**Adviser:** Dr. Howard Everson

---

## Overview

This repository contains all data, analysis code, documentation, and the interactive scrollytelling site for the M.S. thesis *Behind the Score*. The study uses two-level Hierarchical Linear Modeling (HLM) to test whether neighborhood zoning policy independently predicts NYC Regents Exam performance after controlling for school-level compositional factors across 409 public high schools nested within 43 community districts.

**Central finding:** The zoning character of a community district is significantly associated with ELA (γ₀₁ = +1.18, p = .003) and Living Environment (γ₀₁ = +0.93, p = .004) school performance after all school-level controls — but not with Algebra (p = .368). This cross-subject specificity is the thesis contribution.

**Open the site:** Download `index.html` and open it in any modern browser. No server required.

---

## Repository Structure

```
behind-the-score/
│
├── index.html                          ← Interactive scrollytelling site (open in browser)
│
├── data/
│   ├── primary/
│   │   ├── regents_school_level_2023.csv   Primary analytic dataset — 1,154 school-exam records
│   │   ├── regents_school_level_2022.csv   Validation year dataset — 1,046 records
│   │   └── hlm_results_summary.csv         Model results summary
│   │
│   └── neighborhood/
│       ├── neq_index_primary.csv           NEQ index — 43 community districts (primary 2-component)
│       ├── neq_v2_permits.csv              NEQ v2 — adds DOB building permit activity (robustness)
│       ├── neq_v4_highway_airquality.csv   NEQ v4 — adds highway proximity + PM2.5 + NO2
│       ├── neighborhood_context.csv        Neighborhood-level covariates (income, NYCCHA, parks)
│       ├── neighborhood_residuals.csv      Level 1 residuals β₀j per neighborhood per exam
│       └── neighborhood_summary.csv        Neighborhood summary for visualization
│
├── analysis/
│   ├── hlm_analysis.py                 Master HLM estimation script (REML, all 3 exams × 2 years)
│   ├── level1_data_prep.py             Merges NYSED sources into analytic dataset
│   ├── level1_audit.py                 23-check audit of model assumptions
│   └── nysed_extract.py               Extracts data from NYSED .mdb databases*
│
├── docs/
│   ├── Kovacevic_Model_Report.docx    Full quantitative methods report (9 sections)
│   ├── Kovacevic_Variable_Codebook.docx  Complete variable codebook
│   └── Kovacevic_Data_Tables.xlsx     8-sheet data workbook with all model results
│
└── figures/
    ├── fig1_icc_by_exam.png            ICC — % of variance between neighborhoods
    ├── fig2_variance_decomposition.png  Variance waterfall across 3 model steps
    ├── fig3_fixed_effects_forest_plot.png  All fixed effects with 95% CI
    ├── fig4_neq_ela_scatter.png        NEQ vs. ELA residual — central finding
    ├── fig5_neighborhood_profiles.png   All 43 neighborhood profiles
    └── fig6_three_year_replication.png  γ₀₁ across 2022–2024
```

*`nysed_extract.py` contains a hardcoded local path — update `MDB_PATH` before running.

---

## Model

| | |
|---|---|
| **Design** | Two-level Hierarchical Linear Model (HLM) |
| **Level 1** | 409 NYC public high schools |
| **Level 2** | 43 community districts |
| **Estimation** | Restricted Maximum Likelihood (REML) |
| **Software** | Python 3.12 · statsmodels.formula.api.mixedlm() · Powell optimizer |
| **Exams** | Algebra I (CC) · ELA (CC) · Living Environment |
| **Primary year** | 2023 · Validation: 2022 and 2024 |
| **Exclusions** | Special Education schools (n=16) |

**Level 1 equation:**
`Score_ij = β₀j + β₁(Black%) + β₂(Hispanic%) + β₃(Asian%) + β₄(ELL%) + β₅(EconDisadv%) + β₆(ClassSize) + β₇(CTE) + β₈(Transfer) + r_ij`

**Level 2 equation:**
`β₀j = γ₀₀ + γ₀₁(NEQ_j) + U₀j`

---

## NEQ Index

The Neighborhood Educational Quality (NEQ) index measures the zoning and land use character of each community district — a direct output of planning policy decisions, constructed entirely from NYC PLUTO 2025 (857,736 lot records). No Regents data enters the construction.

| Component | Variable | Direction |
|---|---|---|
| 1 — Commercial density | Mean commercial Floor Area Ratio (CommFAR) | Higher = more commercially dense |
| 2 — Residential zoning | % lots in residential zoning | Flipped: lower residential = higher NEQ |

**Construction:** Both z-scored → summed → rescaled 1–10 → z-scored for regression (Bryk, Lee & Holland, 1993 method).

**Scale:** 1.00 (Fresh Meadows — suburban residential) to 10.00 (Lower Manhattan — dense commercial).

### Robustness specifications tested

| Specification | ELA γ₀₁ | Sci γ₀₁ | Verdict |
|---|---|---|---|
| **Primary (CommFAR + ResZone%)** | **+1.18 ★★** | **+0.93 ★★** | **Cleanest, strongest** |
| + DOB permit activity (NEQ v2) | +1.22 ★ | +1.00 ★ | Replicates; permit count collinear with CommFAR |
| + Highway proximity + PM2.5 (NEQ v4 composite) | +0.02 ns | +0.09 ns | Signal cancels — environmental vars fight zoning in composite |
| NEQ + highway separately (Model B) | +1.19 ★★ | +1.00 ★★ | NEQ holds; highway adds nothing at CD level |
| NEQ + PM2.5 separately (Model C) | +0.90 ★ | +0.73 ★ | NEQ holds; PM2.5 collinear with commercial density |

---

## Key Results

| Exam | ICC | Level 1 explains | NEQ γ₀₁ | p | 95% CI |
|---|---|---|---|---|---|
| Algebra I | 11.1% | 97.1% | +0.275 | .368 ns | [−0.32, +0.87] |
| **ELA** | **21.4%** | **78.8%** | **+1.180** | **.003 ★★** | **[+0.41, +1.95]** |
| **Living Environment** | **21.2%** | **95.2%** | **+0.931** | **.004 ★★** | **[+0.29, +1.57]** |

---

## Data Sources

| Dataset | Source | Records | Used for |
|---|---|---|---|
| Regents mean scores | NYSED SRC2024_Group5.mdb | ~3,100 | Outcome variable |
| School demographics | NYSED ENROLL2024.mdb | 409 schools | Level 1 composition |
| Class sizes | NYSED STUDED2024.mdb | 409 schools | Level 1 organization |
| School locations | NYC OpenData | 409 schools | Neighborhood assignment |
| Zoning (NEQ) | NYC PLUTO 2025 v25.2 | 857,736 lots | Level 2 NEQ index |
| Building permits | NYC DOB Open Data 2020–2026 | 851,067 permits | NEQ v2 robustness |
| Street centerline | NYC LION Centerline | 54,061 segments | Highway proximity |
| Air quality | NYC DOHMH EHDP | 262 NTAs | Environmental robustness |
| Median income | CCC NYC / ACS 2023 | 59 CDs | Context table only |
| Public housing | NYCCHA Data Book 2025 | 346 developments | Context table only |

---

## Citation

Kovacevic, V. (2026). *Behind the Score: Neighborhood Policy Context and Regents Exam Performance in NYC Public High Schools*. M.S. thesis, CUNY Graduate Center.

---

## GitHub Pages

Once pushed, enable GitHub Pages (Settings → Pages → Deploy from branch → main → / root).
Site will be live at: `https://[your-username].github.io/behind-the-score`
