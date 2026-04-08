# AI-Derived Transition Probabilities for Periodontal Disease Markov Models

## Technical Requirements for AI/ML Contributors

**Study:** AI-Derived Transition Probabilities for Periodontal Disease Markov Models: A Cross-Country Validation Study Using Longitudinal Cohort Data from the UK and US

**Target journals:** Value in Health, Medical Decision Making, npj Digital Medicine

**Principal Investigator (Health Economics):** Edward — health economist at 21D Clinical Limited  
**Contact:** [insert email]  
**Last updated:** April 2026

---

## 1. Study Overview

This study compares AI-derived covariate-stratified transition probabilities against literature-derived population-average parameters for periodontal disease progression, and evaluates whether the methodological choice produces materially different economic projections for non-surgical periodontal therapy (NSPT). A dual-country design (UK and US) provides external validation across distinct healthcare systems.

The primary output is a cost-consequence analysis (CCA) reporting teeth retained, years in edentulism, and incremental cost. ICER-based cost-effectiveness analysis is reported as a secondary outcome for methodological completeness, but is expected to be uninformative due to the insensitivity of generic utility instruments (EQ-5D) to periodontal disease states.

**The core research question:** Do AI-derived, covariate-stratified transition probabilities produce materially different clinical and economic projections compared to literature-derived population-average parameters?

**Material difference thresholds:**
- ≥0.5 teeth difference at 10 years
- ≥5 percentage point difference in edentulism incidence
- \>20% change in ICER (secondary)

---

## 2. What the AI Scientist Needs to Build

### 2.1 The End-to-End Pipeline

The AI contribution has three sequential phases:

```
Phase 1: Synthetic Population Generation
    ↓
Phase 2: Survival ML Model Training
    ↓
Phase 3: Transition Probability Matrix Extraction
    ↓  (output handed to health economist)
Phase 4: Integration into Markov Economic Model ← Edward's responsibility
```

### 2.2 Required Outputs

The final deliverable from the AI scientist is **24 covariate-stratified transition intensity matrices** (one per stratum), each a 5×5 matrix of annual transition intensities (yr⁻¹) between the five periodontal disease states defined below.

Each matrix must include:
- Point estimates for each non-zero transition intensity
- 95% confidence intervals (from 1,000 bootstrap iterations)
- The stratum definition (diabetes status × smoking status × age group × country)

The 24 strata are defined by:
- **Diabetes:** No / Yes (2 levels)
- **Smoking:** Never / Former / Current (3 levels)
- **Age group:** 18–44 / 45–64 / 65+ (3 levels, condensed from continuous)
- **Country:** UK / US (2 levels)

Note: The workbook labels these as "24 Stratified Matrices" (reduced from the original 36 when former smoking was collapsed in some early strata).

---

## 3. Disease State Definitions

The model uses five periodontal disease states based on the 2018 EFP/AAP classification:

| State | Name | Definition | Absorbing? |
|-------|------|-----------|------------|
| S1 | Periodontal Health | No sites CAL ≥2mm; BOP ≤25% of sites | No |
| S2 | Gingivitis / Stage I | CAL 1–2mm; pocket depth ≤4mm | No |
| S3 | Stage II Periodontitis | CAL 3–4mm; max pocket depth 5–6mm | No |
| S4 | Stage III/IV Periodontitis | CAL ≥5mm; PD ≥7mm or tooth loss due to periodontitis | No |
| S5 | Edentulism | Complete tooth loss | Yes |
| DEATH | Death | Competing absorbing state — UK/US life tables by age+sex | Yes |

**Transitions are forward-only** (S1→S2→S3→S4→S5). Treatment is not modelled as backward state transitions but as hazard ratios (HRs) reducing forward transition intensities.

**Base transition intensity matrix (literature-derived, annual, yr⁻¹):**

| From \ To | S1 | S2 | S3 | S4 | S5 |
|-----------|----|----|----|----|-----|
| S1 Health | 1−Σrow | 0.0450 (q₁₂) | — | — | — |
| S2 Gingivitis | — | 1−Σrow | 0.0220 (q₂₃) | — | — |
| S3 Stage II | — | — | 1−Σrow | 0.0180 (q₃₄) | — |
| S4 Stage III/IV | — | — | — | 1−Σrow | 0.0310 (q₄₅) |
| S5 Edentulism | — | — | — | — | 1.0000 |

---

## 4. Phase 1: Synthetic Population Generation

### 4.1 Why Synthetic Data?

Direct pooling of the source longitudinal datasets is not valid due to three categories of incompatibility: measurement protocol heterogeneity (manual CAL vs CPITN vs electronic probing), outcome definition heterogeneity (clinical measurements vs ICD-10 codes), and temporal scale heterogeneity (1-year trials to 40-year cohorts). Additionally, commercial affiliation restrictions prevent access to NHANES, ELSA, UK Biobank, and All of Us.

The solution is a three-layer data integration framework:

- **Layer 1 — Published longitudinal cohorts** (Mdala 2014, Faddy 2000, Schätzle 2009, Axelsson 2004, Kocher/SHIP 2023, Mailoa 2015, Ramseier 2017): provide base transition intensities and covariate effect estimates.
- **Layer 2 — MIMIC-IV** (PhysioNet): provides comorbidity-specific modifiers (diabetes OR, smoking OR validation, CRP/HbA1c distributions). **Note:** MIMIC-IV is used for covariate characterisation only, not for direct model training. K05.x prevalence in MIMIC-IV is low and the periodontal signal is too weak for staged disease classification.
- **Layer 3 — Synthetic Data Vault (SDV)**: reconciles Layers 1 and 2 into a coherent synthetic longitudinal population for ML training.

### 4.2 Synthetic Population Specification

Use the **Synthetic Data Vault (SDV)** Python library to generate a synthetic longitudinal dataset with:

- **N ≥ 10,000 patients** (per country)
- **Covariates:** age (continuous), sex, smoking status (never/former/current), diabetes status (no/controlled/uncontrolled), BMI, baseline periodontal state (S1–S4)
- **Longitudinal structure:** irregular observation times mimicking clinical visit patterns (6-monthly to annual)
- **Outcome:** periodontal state at each visit, tooth count, time to edentulism, time to death

**Covariate distributions** should be calibrated to:
- Published periodontal cohort demographics (Mdala: median age 52, range 26–84)
- MIMIC-IV derived biomarker profiles (if available — see `MIMIC_IV_preparation_guide.md` in the repo)
- National prevalence data for diabetes and smoking by age/sex

**Base transition intensities** should follow the literature-derived values in the workbook (Sheet 1: Literature Parameters), with covariate multipliers applied multiplicatively:
- Diabetes: 2.10× on all forward transitions
- Current smoking: 1.64× on all forward transitions
- Former smoking: 1.20× (assumed)
- Age 45–64: 1.22× on q₃₄
- Age 65+: 1.49× on q₃₄
- US country adjustment: 0.92× (reflecting different treatment-seeking patterns)

### 4.3 Key Constraint

**MIMIC-IV data must not be shared through LLM APIs or third-party cloud services** (PhysioNet Data Use Agreement). All MIMIC-IV work must occur in a local or institutional compute environment.

---

## 5. Phase 2: Survival ML Model Training

### 5.1 Three-Model Comparison

Train three models and compare their performance:

#### Model 1: DynForest (PRIMARY)

- **What:** Random Survival Forest with multivariate longitudinal endogenous covariates
- **Language:** Python (scikit-survival with custom longitudinal feature engineering)
- **Reference:** Devaux et al. (2023) Stat Methods Med Res
- **Why primary:** Handles irregular, multi-visit structure of periodontal cohort data without parametric assumptions. Mixed model node transformations; Aalen-Johansen estimators averaged across trees.

**Hyperparameters:**

| Parameter | Value | Note |
|-----------|-------|------|
| n_trees | 200 | Increase if OOB error plateaus |
| mtry | √p | p = number of predictors |
| min_nodesize | 20 | Minimum events per terminal node |
| bootstrap_n | 1,000 | For CI generation on probability matrices |
| landmark_times | 1, 3, 5, 10 yr | Prediction horizons |

#### Model 2: Dynamic-DeepHit (SECONDARY)

- **What:** Deep learning competing-risks model with LSTM encoder for longitudinal trajectories
- **Language:** Python (PyTorch)
- **Reference:** Lee et al. (2019) IEEE Trans Biomed Eng
- **Why secondary:** Provides a methodologically distinct comparator. Cause-specific CIFs; competing risks: progression vs tooth loss vs death.

**Hyperparameters:**

| Parameter | Value | Note |
|-----------|-------|------|
| hidden_dim | 64 | LSTM hidden dimension |
| n_layers | 2 | LSTM layers |
| learning_rate | 0.001 | Adam optimiser |
| batch_size | 256 | |
| epochs | 100 | With early stopping (patience=10) |
| dropout | 0.3 | |

#### Model 3: msm Benchmark (COMPARATOR)

- **What:** Continuous-time multi-state Markov model
- **Language:** Python (`pymsm` package)
- **Reference:** Metzger & Greenspan, pymsm (Python multi-state survival models)
- **Why comparator:** This is the classical parametric approach. Direct estimation of transition intensities as functions of observed covariates. Preserves the Markov structure required for downstream economic modelling. Provides the baseline against which ML models are compared.

### 5.2 Performance Targets

| Metric | Target | Note |
|--------|--------|------|
| Harrell's C-statistic | ≥0.70 | Discrimination |
| Time-dependent AUC (1yr, 5yr, 10yr) | ≥0.70 at each horizon | |
| Integrated Brier Score | ≤0.25 | Calibration |
| OOB error rate (DynForest only) | < 0.30 | |

### 5.3 Validation Strategy

- **Internal:** 5-fold cross-validation on synthetic population
- **Cross-country:** Train on UK synthetic data, validate on US (and vice versa). This tests whether AI-derived transition probabilities transport across healthcare systems — a key finding for the paper.
- **Benchmark comparison:** Compare DynForest and Dynamic-DeepHit predictions against msm maximum-likelihood estimates for each stratum.

---

## 6. Phase 3: Transition Probability Matrix Extraction

This is the critical translation step. The ML models output individual-level survival predictions; these must be converted into the structured 5×5 transition intensity matrices that the economic model requires.

### 6.1 Procedure

For each of the 24 covariate strata:

1. Select all synthetic patients belonging to that stratum
2. From the trained model, extract predicted state-occupation probabilities at t = 1 year for patients starting in each state (S1, S2, S3, S4)
3. Convert the 1-year transition probability matrix P(1) to a transition intensity matrix Q using the matrix logarithm: Q = logm(P)
4. Repeat across 1,000 bootstrap samples to generate empirical 95% CIs
5. Output: Q matrix with point estimates and CIs for each stratum

### 6.2 Output Format

Each stratum should be delivered as a CSV or JSON with the following structure:

```json
{
  "stratum": 1,
  "diabetes": "No",
  "smoking": "Never",
  "age_group": "18-44",
  "country": "UK",
  "Q_matrix": {
    "q12": {"point": 0.043, "ci_lower": 0.031, "ci_upper": 0.058},
    "q23": {"point": 0.019, "ci_lower": 0.011, "ci_upper": 0.030},
    "q34": {"point": 0.015, "ci_lower": 0.007, "ci_upper": 0.028},
    "q45": {"point": 0.028, "ci_lower": 0.013, "ci_upper": 0.049}
  },
  "model": "DynForest",
  "n_patients_in_stratum": 850,
  "n_bootstrap": 1000
}
```

Deliver one file per model (DynForest, Dynamic-DeepHit, msm) × 24 strata = 72 output files.

### 6.3 Technical Notes on Intensity vs Probability

The economic model works in **transition intensities** (instantaneous hazard rates, yr⁻¹), not discrete-time probabilities. This is because the source studies report data at irregular, varying time intervals, and intensities are time-scale invariant.

The conversion: for a single transition, the annual probability p relates to the intensity q by `p = 1 − exp(−q)`, and inversely `q = −ln(1 − p)`. For the full multi-state model, `P(t) = expm(t × Q)` gives the transition probability matrix for any cycle length t.

At the low rates in this model (0.018–0.045), intensity and probability values are nearly identical, but the proper matrix exponential conversion should be used throughout, particularly for higher-risk strata where multiplied intensities get larger and the approximation breaks down.

---

## 7. Repository Structure

```
periodontal-ai-markov/
├── README.md                          ← This file
├── data/
│   ├── literature_parameters/         ← Extracted from published papers (Edward)
│   │   └── periodontal_ai_study_workbook.xlsx
│   ├── synthetic_population/          ← SDV-generated (AI scientist)
│   └── mimic_iv/                      ← RESTRICTED — do not commit to repo
│       └── .gitignore
├── models/
│   ├── dynforest/                     ← Python scripts
│   ├── dynamic_deephit/               ← Python/PyTorch
│   └── msm_benchmark/                 ← Python scripts
├── outputs/
│   ├── transition_matrices/           ← 72 JSON/CSV files (24 strata × 3 models)
│   └── performance_metrics/           ← C-statistics, Brier scores, AUCs
├── economic_model/                    ← Edward's domain
│   └── CCA_Markov_model.py           ← Python implementation
├── manuscript/
│   ├── Manuscript.docx
│   └── AI_transProb_proposal.docx
└── docs/
    └── MIMIC_IV_preparation_guide.md
```

---

## 8. Division of Labour

| Task | Owner | Status |
|------|-------|--------|
| Literature parameter extraction | Edward | In progress |
| Study workbook maintenance | Edward | In progress |
| CCA Markov model (Python) | Edward | Conceptualised in Excel; to be reimplemented in Python |
| Manuscript drafting | Edward | In progress |
| MIMIC-IV credentialing & scoping queries | Edward | Access granted |
| Synthetic population generation (SDV) | AI scientist | **TODO** |
| DynForest implementation | AI scientist | **TODO** |
| Dynamic-DeepHit implementation | AI scientist | **TODO** |
| msm benchmark | AI scientist | **TODO** |
| Transition matrix extraction & bootstrap CIs | AI scientist | **TODO** |
| Cross-country validation analysis | AI scientist | **TODO** |
| PSA integration (10,000 MC samples) | Edward + AI scientist | **TODO** |
| EVPPI analysis | Edward | **TODO** |

---

## 9. Technical Environment

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥3.10 | All models, synthetic data generation, economic model |
| PyTorch | ≥2.0 | Dynamic-DeepHit |
| Key packages | `scikit-survival`, `pymsm`, `lifelines`, `sdv`, `torch`, `numpy`, `scipy` | Survival modelling, multi-state Markov, synthetic data, matrix operations |


### Compute Requirements

- DynForest (200 trees × 1,000 bootstrap): moderate CPU, ~4–8 hours on 16-core machine
- Dynamic-DeepHit: GPU recommended (NVIDIA with ≥8GB VRAM), ~2–4 hours training
- SDV population generation: CPU, ~30 minutes for 20,000 patients

---

## 10. Key References

These papers define the methodological framework. Read before starting implementation.

| Ref | Paper | Relevance |
|-----|-------|-----------|
| **[1]** | Mdala et al. (2014) J Clin Periodontol 41:837–845 | Only 3-state Markov model of periodontal disease with covariates. Primary source for q₁₂ and q₃₄ intensities, smoking HR, age effects. Uses `msm` in R. |
| **[2]** | Faddy et al. (2000) J Periodontol 71:454–459 | First 2-state Markov model in periodontitis. Found smoking inhibits healing rather than promoting progression. Source for regression rate β. |
| **[3]** | Schätzle et al. (2009) J Clin Periodontol 36:365–371 | 26-year Norwegian cohort with Markov modelling. Predictive factors for progression. |
| **[4]** | Devaux et al. (2023) Stat Methods Med Res | DynForest methodology paper — RSF with longitudinal endogenous covariates. |
| **[5]** | Lee et al. (2019) IEEE Trans Biomed Eng | Dynamic-DeepHit methodology paper — deep learning competing risks. |
| **[6]** | pymsm — Python multi-state survival models | Python equivalent of msm for continuous-time multi-state Markov modelling. |
| **[7]** | Haug et al. (2024) JAMIA 31:1093–1101 | Demonstrated patient-level Markov TPs from real-world data replicate across international databases. Precedent for cross-country validation. |
| **[8]** | Kocher et al. (2023) Periodontol 2000 | SHIP cohort — 21-year follow-up with site-level CAL progression data. Key source for age effects and annual progression rates. |
| **[9]** | Axelsson et al. (2004) J Clin Periodontol 31:749–757 | 30-year maintenance study — source for treatment effect parameters (NSPT effectiveness). |
| **[10]** | Mailoa et al. (2015) J Periodontol 86:1150–1158 | Systematic review of surgical vs non-surgical therapy — source for HR_surgical. |

---

## 11. Getting Started Checklist

For a new AI contributor:

- [ ] Read this README in full
- [ ] Read the study protocol (`AI_transProb_proposal.docx`)
- [ ] Open the study workbook (`periodontal_ai_study_workbook.xlsx`) and review all 7 sheets
- [ ] Read Mdala et al. (2014) — this is the closest methodological precedent
- [ ] Read Devaux et al. (2023) — DynForest methodology
- [ ] Set up Python environment with `scikit-survival`, `pymsm`, `lifelines`
- [ ] Set up PyTorch environment for Dynamic-DeepHit (GPU recommended)
- [ ] Confirm compute access (GPU for Dynamic-DeepHit)
- [ ] Contact Edward to discuss timeline and any questions

---

## 12. FAQ

**Q: Are we training DynForest on MIMIC-IV directly?**  
A: No. MIMIC-IV is used only for covariate characterisation (diabetes OR, CRP distributions). The AI models are trained on the SDV-generated synthetic population, which is parameterised partly from MIMIC-IV-derived modifiers and partly from published literature parameters. This distinction is critical.

**Q: Why intensities and not probabilities?**  
A: The source studies report data observed at irregular, varying time intervals (Mdala's 6-month visits vs Ramseier's 40-year cumulative data). Intensities are time-scale invariant — you can estimate them from data observed at any interval and convert to probabilities for whatever cycle length the economic model needs.

**Q: Why not use real patient data directly?**  
A: Commercial affiliation restrictions prevent access to NHANES, ELSA, UK Biobank, and All of Us. Additionally, no single dataset contains all the covariates, disease states, and follow-up duration needed. The synthetic population approach resolves both problems.

**Q: What if DynForest performance doesn't reach the C-statistic target?**  
A: Report it transparently. If the AI models don't materially outperform the msm benchmark, that is itself a finding — it would suggest that for periodontal disease, the classical Markov approach is adequate and the additional complexity of ML is not justified.

**Q: How do the AI outputs feed into the economic model?**  
A: Edward takes each stratum's Q matrix, plugs the intensities into the Parameters sheet of the CCA Markov model (Excel), and re-runs. The model runs at minimum 4 times: literature-derived UK, literature-derived US, AI-derived UK, AI-derived US. The difference between AI-derived and literature-derived results is the primary finding.
