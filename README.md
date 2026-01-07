# Periodontal Health Impact Simulator

A research initiative to quantify the systemic health impacts of periodontal disease through health economic modeling and predictive simulation.

## Project Overview

This project has two primary objectives:

1. **Scholarly Research**: Develop a series of peer-reviewed articles demonstrating the causal relationships and economic impacts of periodontal disease on systemic health conditions
2. **Predictive Simulator**: Build a comprehensive health prediction tool that accepts oral biomarkers and risk factors as inputs to forecast disease outcomes and healthcare costs

## Motivation

Periodontal disease affects approximately 50% of the UK population, with severe periodontal pocketing projected to increase by 56.7% by 2050. Growing evidence suggests that periodontal disease is not merely a local oral condition but a significant risk factor for:

- Alzheimer's disease and dementia
- Cardiovascular disease (stroke, myocardial infarction)
- Diabetes complications
- Other systemic inflammatory conditions

The mechanisms include systemic inflammation, bacteremia from oral pathogens (particularly *Porphyromonas gingivalis* and *Fusobacterium nucleatum*), and chronic immune activation. Understanding and quantifying these relationships can inform public health policy and clinical practice.

## Current Research

### 1. Periodontal Disease and Alzheimer's Disease/Dementia

**Model Type**: Individual-level microsimulation
**Time Horizon**: 2023-2040
**Population**: England

**Key Features**:
- Four dementia stages: mild → moderate → severe → death
- Baseline periodontal prevalence scenarios: 25%, 50%, 75%
- Co-morbidity modeling: smoking, diabetes, cerebrovascular disease, cardiovascular disease
- Dual cost perspective: NHS and informal caregivers
- Outcomes: QALYs, costs, dementia cases prevented

**Current Status**: Methodology complete, awaiting model execution for results

**Files**:
- `Intro_AD.docx` - Background and literature review
- `Methodology_AD.docx` - Model structure and parameters
- `Results_AD.docx` - Results template (pending)
- `Supplementary_Material_AD.docx` - Technical appendix

### 2. Periodontal Disease and Cardiovascular Disease

**Model Type**: Markov cohort model
**Time Horizon**: 10 years
**Population**: 65-year-old adults with severe periodontal disease

**Key Features**:
- Eight health states including post-stroke, post-MI, and combinations
- Intervention: Non-surgical periodontal therapy
- Cost-effectiveness analysis against NICE thresholds (£20,000-£30,000/QALY)
- Tunnel states for acute vs. chronic event phases
- Probabilistic sensitivity analysis (10,000 Monte Carlo simulations)

**Current Status**:
- ✅ Methodology complete
- ✅ Excel model built and validated
- ✅ Research paper drafted (3,744 words)
- ✅ Figures generated (tornado plot, CE plane, CEAC)
- 🔄 Supplementary material in progress
- 🔄 Web application development initiated

**Paper Progress (Ready for Submission):**
- Complete manuscript: `CVD_consolidated.md` (3,744 words)
- Publication-quality figures (600 DPI):
  - `plots/figure1_tornado_plot.png` - One-way sensitivity analysis
  - `plots/figure2_ce_plane.png` - Cost-effectiveness plane (10,000 PSA iterations)
  - `plots/figure3_ceac.png` - Cost-effectiveness acceptability curve
- Tables: Base case results and PSA summary statistics
- References: 50 citations from peer-reviewed literature

**Remaining Tasks:**
- Supplementary material (detailed parameter tables, CHEERS checklist)
- Final formatting for journal submission

**Files**:
- `CVD_consolidated.md` - Complete research paper
- `Intro_CVD.docx` - Background and literature review
- `Methodology_CVD.docx` - Model structure and parameters
- `Supplementary_Material_CVD.docx` - Technical appendix (in progress)
- `PD_CVD_markov - PSA On.xlsm` - Excel-based Markov model
- `generate_cvd_figures.py` - Python script for figure generation
- `plots/` - Publication-ready figures
- `backend/` - FastAPI backend (initiated)
- `frontend/` - React frontend (planned)

**Web Application**: A modern web interface is being developed to make this model accessible to researchers without Excel expertise. See [TODO.md](TODO.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Repository Structure

```
periodontal/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── TODO.md                             # Task list for contributors
├── CONTRIBUTING.md                     # Contributor guidelines
├── Intro_AD.docx                      # AD study introduction
├── Intro_CVD.docx                     # CVD study introduction
├── Methodology_AD.docx                # AD model methodology
├── Methodology_CVD.docx               # CVD model methodology
├── Results_AD.docx                    # AD results (pending)
├── Supplementary_Material_AD.docx     # AD technical appendix
├── Supplementary_Material_CVD.docx    # CVD technical appendix
├── PD_CVD_markov - PSA On.xlsm       # CVD Markov model (Excel)
├── backend/                            # CVD web app backend (FastAPI)
│   ├── models/                         # Markov engine, PSA engine
│   ├── api/                            # REST API endpoints
│   ├── utils/                          # Calculations, distributions
│   ├── tests/                          # Backend tests
│   └── requirements.txt                # Python dependencies
└── frontend/                           # CVD web app frontend (React, planned)
```

## CVD Web Application (In Development)

A modern web application is being developed to make the CVD Markov model accessible to researchers without requiring Excel expertise. The application will enable:

**Features:**
- **Parameter Adjustment:** Modify baseline hazards, treatment effects, costs, and utilities through an intuitive web interface
- **Simulation Execution:** Run deterministic base case analysis (1-2 seconds) or probabilistic sensitivity analysis (60-90 seconds with 10,000 Monte Carlo iterations)
- **Interactive Visualizations:**
  - Cost-effectiveness plane (scatter plot with 10,000 PSA iterations)
  - Cost-effectiveness acceptability curve (CEAC)
  - Markov trace (state occupancy over time)
  - Summary statistics with 95% confidence intervals
- **Results Export:** Download results as Excel workbooks, CSV files, or PNG charts

**Technology Stack:**
- **Backend:** Python 3.11+, FastAPI, NumPy, pandas
- **Frontend:** React 18+, Redux Toolkit, Chart.js, Plotly
- **Testing:** pytest (backend), Jest (frontend)

**Development Status:**
- Phase 1 (Backend - Markov Engine): In Progress
- Phase 2 (Backend - PSA Implementation): Planned
- Phase 3 (Backend - API Endpoints): Planned
- Phase 4 (Frontend - React UI): Planned
- Phase 5 (Integration & Testing): Planned

**Contributing:**
- See [TODO.md](TODO.md) for available tasks
- See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- Check the implementation plan at `.claude/plans/distributed-discovering-balloon.md`

**Timeline:** 7-8 weeks for full implementation

## Methodology Summary

### Alzheimer's Disease Study

- **Design**: Hazard-based microsimulation
- **Cycle**: Annual time steps
- **Discount Rate**: 3.5% (costs and QALYs)
- **Data Sources**: NHS England Primary Care Dementia Data, ONS population projections
- **Key Parameters**:
  - Periodontal disease HR for dementia onset: 1.47 (95% CI: 1.32-1.65)
  - Periodontal disease HR for mortality: 1.36 (95% CI: 1.10-1.69)
- **Analysis**: Probabilistic sensitivity analysis with 1,000 iterations

### Cardiovascular Disease Study

- **Design**: State-transition Markov model
- **Cycle**: 1 year
- **Discount Rate**: 3.5% (costs and QALYs)
- **Data Sources**: UK Biobank, NHS cost collections, ONS life tables
- **Treatment Effects**:
  - Stroke hazard reduction: 0.40-0.78 (median used)
  - MI hazard reduction: 0.54-0.90 (median used)
- **Analysis**: One-way and probabilistic sensitivity analysis

## Key Evidence Base

Both studies are grounded in extensive systematic reviews and meta-analyses:

- **Dementia**: PD associated with 22% higher dementia risk (RR: 1.18, 95% CI: 1.06-1.31)
- **CVD**: PD associated with 22% higher CVD risk, with dose-response by severity
- **Treatment Effects**: Non-surgical periodontal therapy reduces inflammatory biomarkers, improves endothelial function, and may reduce CVD mortality
- **Economic Impact**: England dementia costs projected at £80.4 billion by 2040

## Roadmap

### Phase 1: Complete Current Studies (In Progress)
- [ ] Execute AD microsimulation model
- [ ] Generate AD study results and statistical analysis
- [ ] Complete CVD model parameter tables
- [ ] Run CVD model and generate cost-effectiveness results
- [ ] Finalize supplementary materials
- [ ] Prepare manuscripts for journal submission

### Phase 2: Expand Research Portfolio
- [ ] Periodontal disease and Type 2 diabetes
- [ ] Periodontal disease and adverse pregnancy outcomes
- [ ] Periodontal disease and respiratory infections
- [ ] Cost-effectiveness of population-level oral health interventions

### Phase 3: Build Predictive Simulator
- [ ] Define comprehensive input parameters (clinical, biomarker, demographic)
- [ ] Select technical platform and architecture
- [ ] Integrate multi-disease models
- [ ] Develop user interface for clinicians/researchers
- [ ] Validate against real-world cohort data
- [ ] Deploy as web-based or clinical decision support tool

## Planned Simulator Features

**Inputs**:
- Oral biomarkers: Bacterial load (P. gingivalis, F. nucleatum, etc.)
- Clinical measures: Pocket depth, bleeding on probing, attachment loss
- Inflammatory markers: IL-6, CRP, TNF-α
- Patient characteristics: Age, sex, smoking status, comorbidities
- Treatment history: Frequency and type of periodontal interventions

**Outputs**:
- Multi-year disease risk predictions (dementia, CVD, diabetes, etc.)
- Quality-adjusted life expectancy
- Projected healthcare costs (NHS and societal perspectives)
- Cost-effectiveness of alternative treatment strategies
- Uncertainty ranges and scenario analyses

## Compliance and Standards

- **Economic Evaluation**: Aligned with NICE guidelines for health technology assessment
- **Reporting**: CHEERS 2022 (Consolidated Health Economic Evaluation Reporting Standards)
- **Cost Year**: 2024 GBP (HM Treasury GDP deflators)
- **Utility Values**: EQ-5D derived from UK population norms

## Data Sources

- NHS England Primary Care Dementia Data
- UK Biobank
- Office for National Statistics (ONS) population projections and life tables
- NHS National Cost Collection
- Adult Oral Health Survey 2021
- English Longitudinal Study of Ageing (ELSA)

## Contributing

This is an active research project. Contributions from health economists, epidemiologists, periodontal researchers, data scientists, and software developers are welcome.

**Getting Started:**
1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines
2. Check [TODO.md](TODO.md) for available tasks
3. Review the implementation plan at `.claude/plans/distributed-discovering-balloon.md`
4. Choose a task that matches your skills and interests
5. Create a branch, implement your changes, and submit a pull request

**Areas for Contribution:**
- **Backend Development:** Python, FastAPI, Markov modeling, PSA implementation
- **Frontend Development:** React, Redux, data visualization (Chart.js, Plotly)
- **Testing & QA:** Unit tests, integration tests, validation against Excel
- **Research & Validation:** Model calibration, NICE guidelines compliance, literature review
- **Documentation:** User guides, API documentation, tutorials
- **DevOps:** Deployment setup, Docker, CI/CD pipelines

**Skills Needed:**
- Python 3.11+ (backend developers)
- React 18+ / JavaScript (frontend developers)
- Health economics or epidemiology (research validation)
- Scientific computing (NumPy, pandas, statistical distributions)
- Software testing (pytest, Jest)

## Citation

If you use this work in academic research, please cite:

```
[Citation to be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Kwaai, Personal AI Lab

## Contact

For questions, collaboration inquiries, or access to model code:

[Contact information to be added]

## Acknowledgments

This research draws on extensive epidemiological evidence from the UK Biobank, NHS England, and numerous systematic reviews and meta-analyses conducted by the global periodontal and cardiovascular research communities.

---

**Project Status**: 🟡 Active Development
**Last Updated**: November 2025
