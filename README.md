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

**Current Status**: Methodology complete, Excel model built, results pending

**Files**:
- `Intro_CVD.docx` - Background and literature review
- `Methodology_CVD.docx` - Model structure and parameters
- `Supplementary_Material_CVD.docx` - Technical appendix (partial)
- `PD_CVD_markov - PSA On.xlsm` - Excel-based Markov model

## Repository Structure

```
periodontal/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── Intro_AD.docx                      # AD study introduction
├── Intro_CVD.docx                     # CVD study introduction
├── Methodology_AD.docx                # AD model methodology
├── Methodology_CVD.docx               # CVD model methodology
├── Results_AD.docx                    # AD results (pending)
├── Supplementary_Material_AD.docx     # AD technical appendix
├── Supplementary_Material_CVD.docx    # CVD technical appendix
└── PD_CVD_markov - PSA On.xlsm       # CVD Markov model
```

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

This is an active research project. Contributions from health economists, epidemiologists, periodontal researchers, and data scientists are welcome.

**Areas for Contribution**:
- Model validation and calibration
- Additional disease pathway modeling
- Software development for simulator
- Clinical data for model validation
- Literature review and evidence synthesis

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
