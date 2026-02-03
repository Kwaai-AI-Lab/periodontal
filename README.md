# Periodontal Health Impact Simulator

A research initiative to quantify the systemic health impacts of periodontal disease through health economic modeling and predictive simulation.

## Project Overview

This repository contains two major health economic modeling projects examining the relationship between periodontal disease and systemic health conditions:

1. **Periodontal Disease and Cardiovascular Disease (CVD)** - Markov cohort model
2. **Periodontal Disease and Alzheimer's Disease/Dementia (AD)** - Individual-based microsimulation

---

## 🔴 CVD Project: SUBMITTED TO JOURNAL

**Status**: ✅ **Paper submitted to Cost Effectiveness and Resource Allocation journal**

### Model Overview

**Model Type**: State-transition Markov cohort model
**Population**: 65-year-old adults with severe periodontal disease
**Time Horizon**: 10 years
**Intervention**: Non-surgical periodontal therapy
**Perspective**: NHS and societal (informal care costs)

### Key Features

- **Eight health states**: Well, post-stroke, post-MI, stroke+MI combinations, death
- **Tunnel states**: Differentiates acute event phase (first year) from chronic phase
- **Cost-effectiveness analysis**: Evaluated against NICE thresholds (£20,000-£30,000/QALY)
- **Probabilistic sensitivity analysis**: 10,000 Monte Carlo simulations
- **One-way sensitivity analysis**: Tornado diagrams for key parameters

### Treatment Effects

| Outcome | Hazard Reduction Range | Median Used |
|---------|----------------------|-------------|
| Stroke | 0.40-0.78 | 0.59 |
| Myocardial Infarction | 0.54-0.90 | 0.72 |

### Key Results

- **Base case ICER**: Cost-effective intervention
- **Probabilistic analysis**: Robust across parameter uncertainty
- **Sensitivity analysis**: Treatment effects and utilities are key drivers

### Publication Files

| File | Description |
|------|-------------|
| `Main_Text_CVD_Paper_finalised.tex` | **FINAL MANUSCRIPT** (submitted to journal) |
| `Supplementary_Material_CVD.tex` | Supplementary appendix with technical details |
| `CVD_Paper.tex` | Earlier draft version |
| `CVD_consolidated.md` | Markdown version for review |
| `images_CVD/` | Publication-quality figures (600 DPI) |
| `plots/figure1_tornado_plot.png` | One-way sensitivity analysis |
| `plots/figure2_ce_plane.png` | Cost-effectiveness plane (10,000 PSA iterations) |
| `plots/figure3_ceac.png` | Cost-effectiveness acceptability curve |

### Data Sources

- UK Biobank (treatment effects, epidemiological parameters)
- NHS National Cost Collection (healthcare costs)
- ONS life tables (background mortality)
- Published literature (utilities, transition probabilities)

### Compliance

- **Reporting Standard**: CHEERS 2022 (Consolidated Health Economic Evaluation Reporting Standards)
- **Guidelines**: NICE health technology assessment guidelines
- **Cost Year**: 2024 GBP (HM Treasury GDP deflators)
- **Discount Rate**: 3.5% annual (costs and QALYs)

---

## 🔵 CVD Web Application: IN DEVELOPMENT

**Next Phase**: Python backend health model web application

A modern web application is being developed to make the CVD Markov model accessible to researchers without requiring Excel expertise. The application will enable interactive parameter adjustment, real-time simulation, and dynamic visualization.

### Planned Features

**Core Functionality:**
- **Parameter Adjustment**: Modify baseline hazards, treatment effects, costs, and utilities through an intuitive web interface
- **Simulation Execution**:
  - Deterministic base case analysis (1-2 seconds)
  - Probabilistic sensitivity analysis (60-90 seconds with 10,000 Monte Carlo iterations)
- **Interactive Visualizations**:
  - Cost-effectiveness plane (scatter plot with 10,000 PSA iterations)
  - Cost-effectiveness acceptability curve (CEAC)
  - Markov trace (state occupancy over time)
  - Tornado plots for one-way sensitivity analysis
  - Summary statistics with 95% confidence intervals
- **Results Export**: Download results as Excel workbooks, CSV files, or PNG charts

### Technology Stack

- **Backend**: Python 3.11+, FastAPI, NumPy, pandas
- **Frontend**: React 18+, Redux Toolkit, Chart.js, Plotly
- **Testing**: pytest (backend), Jest (frontend)
- **Deployment**: Docker, CI/CD pipelines

### Development Status

- Phase 1 (Backend - Markov Engine): In Progress
- Phase 2 (Backend - PSA Implementation): Planned
- Phase 3 (Backend - API Endpoints): Planned
- Phase 4 (Frontend - React UI): Planned
- Phase 5 (Integration & Testing): Planned

### Repository Structure

```
backend/                            # CVD web app backend (FastAPI)
├── models/                         # Markov engine, PSA engine
├── api/                            # REST API endpoints
├── utils/                          # Calculations, distributions
├── config/                         # Configuration management
├── tests/                          # Backend tests
└── requirements.txt                # Python dependencies
frontend/                           # CVD web app frontend (React, planned)
```

### Contributing

- See [TODO.md](TODO.md) for available tasks
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Check implementation plan at `.claude/plans/` for detailed specifications

**Areas for Contribution:**
- **Backend Development**: Python, FastAPI, Markov modeling, PSA implementation
- **Frontend Development**: React, Redux, data visualization (Chart.js, Plotly)
- **Testing & QA**: Unit tests, integration tests, validation against Excel model
- **Documentation**: API documentation, user guides, tutorials
- **DevOps**: Docker containerization, deployment pipelines

---

## 🟡 AD Project: MODEL COMPLETE, PAPER IN PROGRESS

**Status**: ✅ Model runs complete | 🔄 Manuscript in preparation (target: 04 February 2026)

### Model Overview

**Model Type**: Individual-based microsimulation (IBM)
**Population**: **UK adults aged 65 and over** (~10.8 million individuals)
**Time Horizon**: 2023-2040 (18 years, annual cycles)
**Intervention**: Periodontal disease prevention/treatment (counterfactual comparison)
**Perspective**: NHS and societal (informal caregiver costs)

### Key Model Configuration (IBM_PD_AD_v2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Population Size** | 10,787,479 | 65+ UK population |
| **Time Steps** | 17 | 2023-2040 (18 years) |
| **New Entrants/Year** | 700,000 | Distributed across 65+ age bands |
| **Entry Age** | Distributed (65-100+) | Age band multipliers by year |
| **Entrant Growth Rate** | 0.59% annually | Based on ONS projections 2019-2040 |
| **Base Onset Probability** | 0.0025 annually | Baseline risk of dementia onset |
| **Incidence Growth** | 0.0% | Constant hazard over time (no temporal growth) |
| **Discount Rate** | 3.5% | End-of-cycle discounting (costs and QALYs) |

### Key Model Features

- **Four dementia stages**: Cognitively normal → Mild → Moderate → Severe → Death
- **Open population**: 700,000 new entrants annually (distributed across age bands 65-100+)
- **Risk factor modeling**: Periodontal disease, smoking, diabetes, cerebrovascular disease, cardiovascular disease
- **Living settings**: Home care vs institutional care transitions
- **Probabilistic sensitivity analysis**: 500 iterations across 3 prevalence scenarios
- **Age-dependent dynamics**: Age band multipliers adjust entrant age distribution by year (2025, 2030, 2035, 2040)

### Initial Prevalence Calibration

The model is calibrated to match observed dementia prevalence in the 65+ population:

| Age Band | Female Prevalence | Male Prevalence |
|----------|-------------------|-----------------|
| 65-79 | 2.07% | 1.93% |
| 80-100 | 14.63% | 10.59% |

**Target**: Approximately 850,000 prevalent dementia cases at baseline (t=0, year 2023)

### Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Periodontal Disease Prevalence** | 25%, 50%, 75% scenarios | Adult Oral Health Survey 2021 |
| **PD Hazard Ratio (Onset)** | 1.47 (95% CI: 1.32-1.65) | Meta-analysis |
| **PD Hazard Ratio (Mortality)** | 1.36 (95% CI: 1.10-1.69) | Meta-analysis |
| **Stage Transition Durations** | Mild→Moderate: 2.2y, Moderate→Severe: 2y, Severe→Death: 4y | Tariot et al. 2024 |
| **Discount Rate** | 3.5% annual (costs and QALYs) | NICE guidelines |

### Model Results

Final results are stored in **`AD_Microsimulation_results/`**:

| File | Description |
|------|-------------|
| `Baseline_Model_25.xlsx` | Results for 25% periodontal disease prevalence |
| `Baseline_Model_50.xlsx` | Results for 50% periodontal disease prevalence |
| `Baseline_Model_75.xlsx` | Results for 75% periodontal disease prevalence |
| `Full_Results.xlsx` | Consolidated results across all scenarios |

Each Excel file contains:
- **Summary statistics**: Mean and 95% confidence intervals (scaled to full 65+ population)
- **Key outcomes**: Incident dementia cases, total QALYs, total costs (NHS and informal)
- **Age-stratified results**: Outcomes by age band (65-79, 80+)
- **Validation metrics**: Model calibration checks
- **PSA draws**: Individual iteration results for transparency

**PSA Detailed Results**: The directory also contains three subdirectories (`psa_results_25_v2/`, `psa_results_50_v2/`, `psa_results_75_v2/`) with detailed PSA outputs including compressed results files (`.pkl.gz`), Excel workbooks, and methods justification text files.

### PSA Methodology

The analysis uses an efficient two-level nested design (O'Hagan et al., 2007):
1. **Population scaling**: 1% sample per iteration (~107,875 individuals)
2. **Iterations**: 500 per prevalence scenario
3. **Scaling**: Absolute counts multiplied by 100× to full population
4. **Validation**: Rates remain invariant after scaling

**Computational efficiency**: 100× reduction in runtime vs full population

### Expected Outcomes

The model quantifies:
- **Incident dementia cases** prevented by eliminating periodontal disease
- **QALYs lost** due to periodontal disease (patient and caregiver burden)
- **Healthcare costs** attributable to periodontal disease (NHS and informal care)
- **Cost-effectiveness** of hypothetical periodontal interventions

### Manuscript Status

**Target Completion**: 04 February 2026 (end of day)
**Format**: Microsoft Word (.docx) and Markdown (.md)
**Target Journal**: To be determined

Expected manuscript sections:
- Introduction: Periodontal disease epidemiology, dementia burden, mechanistic pathways
- Methods: Microsimulation model structure, data sources, PSA methodology
- Results: Incident cases, QALYs, costs across prevalence scenarios
- Discussion: Public health implications, cost-effectiveness interpretation
- Supplementary Material: Technical appendix, parameter tables, validation

### Data Sources

- **NHS England Primary Care Dementia Data** (initial prevalence)
- **ONS population projections** (65+ population, new entrants, age structure evolution)
- **ONS life tables** (background mortality by single year of age)
- **Adult Oral Health Survey 2021** (periodontal disease prevalence)
- **Published meta-analyses** (hazard ratios for PD effects)
- **NHS National Cost Collection** (healthcare costs)
- **Reed et al. 2017** (caregiver utilities)

---

## Repository Structure

```
periodontal/
├── README.md                                  # This file
├── LICENSE                                    # MIT License
│
├── ────────────────────────────────────────
│   CVD PROJECT FILES (SUBMITTED)
├── ────────────────────────────────────────
├── Main_Text_CVD_Paper_finalised.tex         # ✅ FINAL CVD MANUSCRIPT (submitted)
├── Supplementary_Material_CVD.tex            # CVD supplementary appendix
├── CVD_Paper.tex                             # Earlier CVD draft
├── CVD_consolidated.md                       # CVD markdown version
├── CVD consolidated.docx                     # CVD Word version
├── images_CVD/                               # CVD manuscript figures
│   ├── figure_1.png                          # Model schematic
│   ├── figure_2.png                          # Markov trace
│   ├── figure_3.png                          # Cost-effectiveness plane
│   └── figure_4.png                          # CEAC
├── generate_cvd_figures.py                   # Python script for CVD figures
│
├── ────────────────────────────────────────
│   CVD WEB APPLICATION (IN DEVELOPMENT)
├── ────────────────────────────────────────
├── backend/                                  # Python FastAPI backend
│   ├── models/                               # Markov engine, PSA engine
│   ├── api/                                  # REST API endpoints
│   ├── utils/                                # Calculations, distributions
│   ├── config/                               # Configuration management
│   ├── tests/                                # Backend tests
│   └── requirements.txt                      # Python dependencies
├── frontend/                                 # React frontend (planned)
│
├── ────────────────────────────────────────
│   AD PROJECT FILES (PAPER IN PROGRESS)
├── ────────────────────────────────────────
├── IBM_PD_AD.py                              # Original full-population model (ages 35-100+)
├── IBM_PD_AD_v2.py                           # ✅ CURRENT MODEL: 65+ only
├── IBM_PD_AD_V2_README.md                    # Detailed v2 documentation
├── run_psa_direct.py                         # PSA script for original model
├── run_psa_direct_v2.py                      # ✅ PSA script for v2 model
├── AD_Microsimulation_results/               # ✅ FINAL MODEL RESULTS
│   ├── Baseline_Model_25.xlsx                # 25% PD prevalence results
│   ├── Baseline_Model_50.xlsx                # 50% PD prevalence results
│   ├── Baseline_Model_75.xlsx                # 75% PD prevalence results
│   ├── Full_Results.xlsx                     # Consolidated results
│   ├── psa_results_25_v2/                    # PSA outputs (25% prevalence)
│   ├── psa_results_50_v2/                    # PSA outputs (50% prevalence)
│   └── psa_results_75_v2/                    # PSA outputs (75% prevalence)
├── Methodology_AD.docx                       # AD methodology documentation
├── external_validation.py                    # Model validation against external data
├── generate_validation_data.py               # Validation data generation
├── EXTERNAL_VALIDATION_README.md             # Validation documentation
│
├── ────────────────────────────────────────
│   SENSITIVITY ANALYSIS TOOLS
├── ────────────────────────────────────────
├── pd_sensitivity_analysis.py                # Periodontal disease sensitivity analysis
├── PD_SENSITIVITY_README.md                  # Sensitivity analysis documentation
├── run_pd_tornado.py                         # Tornado plot generation
└── example_psa_visualization.py              # PSA visualization examples
│
├── ────────────────────────────────────────
│   SUPPORTING FILES
├── ────────────────────────────────────────
├── plots/                                    # All generated figures
├── TODO.md                                   # Development task list
├── CONTRIBUTING.md                           # Contributor guidelines
├── .gitignore                                # Git ignore patterns
├── periodontal.code-workspace                # VS Code workspace
└── convert_word_to_md.py                     # Document conversion utility
```

---

## Key Evidence Base

Both studies are grounded in extensive systematic reviews and meta-analyses:

### Periodontal Disease and Dementia

- **Systematic Review**: Periodontal disease associated with 22% higher dementia risk (RR: 1.18, 95% CI: 1.06-1.31)
- **Our Model**: Hazard ratio of 1.47 (95% CI: 1.32-1.65) for dementia onset
- **Mortality Effect**: 36% higher mortality risk with periodontal disease (HR: 1.36)
- **Mechanism**: Systemic inflammation, *P. gingivalis* in brain tissue, chronic bacteremia

### Periodontal Disease and Cardiovascular Disease

- **Systematic Review**: 22% higher CVD risk with periodontal disease (dose-response by severity)
- **Treatment Effects**: Non-surgical periodontal therapy reduces:
  - Inflammatory biomarkers (CRP, IL-6)
  - Endothelial dysfunction (flow-mediated dilation improvement)
  - CVD mortality (observational data)
- **Mechanisms**: Bacteremia, systemic inflammation, atherosclerotic plaque formation

### Economic Context

- **UK Dementia Costs**: Projected at £80.4 billion by 2040
- **Periodontal Disease Prevalence**: Affects approximately 50% of UK adults
- **Severe Periodontal Disease**: Projected to increase 56.7% by 2050
- **NICE Thresholds**: £20,000-£30,000 per QALY gained

---

## Methodology Summary

### Alzheimer's Disease Study

- **Design**: Hazard-based microsimulation with annual time steps
- **Population**: 10,787,479 individuals aged 65+ (calibrated to ~850,000 prevalent dementia cases)
- **Discount Rate**: 3.5% (costs and QALYs, end-of-cycle)
- **Analysis**: 500 PSA iterations per prevalence scenario (25%, 50%, 75%)
- **Efficiency**: Two-level nested design with 1% population sample per iteration
- **Outcomes**: Incident dementia cases, QALYs (patient and caregiver), costs (NHS and informal)
- **Open Population**: 700,000 annual entrants distributed across age bands with evolving age structure

### Cardiovascular Disease Study

- **Design**: State-transition Markov cohort model with 1-year cycles
- **Population**: 65-year-old adults with severe periodontal disease
- **Discount Rate**: 3.5% (costs and QALYs)
- **Analysis**: One-way sensitivity analysis + 10,000 PSA iterations
- **Outcomes**: Incremental cost-effectiveness ratio (ICER), cost-effectiveness plane, CEAC

---

## Project Timeline

### CVD Project: SUBMITTED ✅
- ✅ **Methodology**: Complete
- ✅ **Model Development**: Excel Markov model validated
- ✅ **Results**: Base case and PSA complete
- ✅ **Manuscript**: Submitted to *Cost Effectiveness and Resource Allocation*
- ✅ **Figures**: Publication-quality figures generated (600 DPI)
- ✅ **Supplementary Material**: Technical appendix finalized
- 🔵 **Web Application**: Python backend in development

### AD Project: FINAL STAGE 🔄
- ✅ **Methodology**: Complete (v2 model for 65+ population)
- ✅ **Model Development**: IBM_PD_AD_v2.py fully implemented
- ✅ **Model Runs**: All scenarios complete (25%, 50%, 75% prevalence)
- ✅ **Results**: Final outputs in `AD_Microsimulation_results/`
- 🔄 **Manuscript**: In preparation (target: 04 February 2026)
- 🔄 **Figures**: To be generated from results files
- 🔄 **Supplementary Material**: To be finalized with manuscript

### Future Work
- [ ] CVD web application development (Python backend + React frontend)
- [ ] AD manuscript submission to peer-reviewed journal
- [ ] Periodontal disease and Type 2 diabetes model
- [ ] Periodontal disease and adverse pregnancy outcomes
- [ ] Comprehensive health prediction simulator (multi-disease integration)

---

## Compliance and Standards

Both projects adhere to international standards:

- **Economic Evaluation Guidelines**: NICE health technology assessment
- **Reporting Standards**: CHEERS 2022 (Consolidated Health Economic Evaluation Reporting Standards)
- **Cost Year**: 2024 GBP (HM Treasury GDP deflators)
- **Utility Measures**: EQ-5D derived from UK population norms
- **Discount Rate**: 3.5% annual (NICE reference case)
- **Perspective**: NHS and societal (informal caregiver costs included)

---

## Running the Models

### CVD Model (Excel)
The CVD model is currently implemented in Excel:
- **File**: `PD_CVD_markov - PSA On.xlsm`
- **Requirements**: Microsoft Excel with macros enabled
- **Outputs**: Base case results, PSA results, sensitivity analyses

### CVD Web Application (In Development)
Python backend with FastAPI:
```bash
cd backend
pip install -r requirements.txt
pytest tests/  # Run tests
# API server launch (in development)
```

### AD Model (Current Version: IBM_PD_AD_v2)

```bash
# Run single scenario
python IBM_PD_AD_v2.py

# Run full PSA across all prevalence scenarios
python run_psa_direct_v2.py
```

**Requirements**:
- Python 3.11+
- pandas, numpy, matplotlib
- tqdm (optional, for progress bars)

**Configuration**: Edit `general_config` dictionary in `IBM_PD_AD_v2.py`

**Output**: Results saved to `psa_results_XX_v2/` directories and `AD_Microsimulation_results/`

See `IBM_PD_AD_V2_README.md` for detailed documentation.

---

## Citation

### CVD Study
```
[Citation to be added upon publication in Cost Effectiveness and Resource Allocation]
```

### AD Study
```
[Citation to be added upon publication]
```

---

## Contact

For questions about the models, data sources, or collaboration inquiries:

[Contact information to be added]

---

## Acknowledgments

This research draws on extensive epidemiological evidence from:
- UK Biobank
- NHS England Primary Care Dementia Data
- Office for National Statistics (ONS)
- Systematic reviews and meta-analyses from the global periodontal, cardiovascular, and dementia research communities

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Kwaai, Personal AI Lab

---

**Project Status**: 🔴 CVD Paper Submitted + 🔵 Web App in Development | 🟡 AD Paper in Preparation
**Last Updated**: 03 February 2026
