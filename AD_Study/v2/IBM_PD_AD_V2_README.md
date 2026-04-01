# IBM_PD_AD_v2 Model Documentation

## Overview

**IBM_PD_AD_v2** is a modified version of the periodontal disease and dementia individual-based model (IBM_PD_AD), specifically designed to simulate the **65+ population only**. This version includes several important modifications to focus exclusively on older adults and adjust model behavior for sensitivity analyses.

## Key Differences from Original Model

### 1. Population Scope: 65+ Only

| Parameter | Original (IBM_PD_AD) | V2 (IBM_PD_AD_v2) |
|-----------|---------------------|-------------------|
| **Population Size** | 33,167,098 (full UK) | 10,787,479 (65+ only) |
| **Age Range** | 35-100+ years | 65-100+ years |
| **New Entrants/Year** | 800,000 (distributed across ages) | 600,000 (all at age 65) |
| **Entry Age** | Distributed (35-100+) | Fixed at 65 years |

### 2. Model Behavior Modifications

#### a. Severe-to-Death Relative Risks
All severe-to-death relative risk interactions from risk factors are **set to 1.00** (no effect):

| Risk Factor | Original RR | V2 RR |
|-------------|-------------|-------|
| Smoking | Female: 1.75, Male: 1.75 | Female: 1.00, Male: 1.00 |
| Periodontal Disease | Female: 1.36, Male: 1.36 | Female: 1.00, Male: 1.00 |
| Cerebrovascular Disease | Female: 1.18, Male: 1.18 | Female: 1.00, Male: 1.00 |
| CVD Disease | Female: 1.34, Male: 1.34 | Female: 1.00, Male: 1.00 |
| Diabetes | Female: 1.79, Male: 1.79 | Female: 1.00, Male: 1.00 |

**Rationale**: This modification isolates the effect of risk factors on dementia onset and progression, removing their mortality effects in the severe stage.

#### b. QALY Accrual
- **Original**: All individuals accrue QALYs (baseline utility for cognitively normal, stage-specific for dementia)
- **V2**: Only individuals with dementia (mild, moderate, severe) accrue QALYs
- **Cognitively normal individuals**: QALY weight = 0.0

**Rationale**: This focuses the quality-of-life analysis exclusively on the burden experienced by those with dementia.

#### c. Incidence Growth Rate
- **Original**: 2% annual growth in baseline onset probability
- **V2**: 0% annual growth (rate = 0.0)

**Rationale**: Maintains constant baseline hazard over time for clearer interpretation of model outputs.

### 3. Initial Dementia Prevalence

The model is calibrated to achieve approximately **800,000 prevalent dementia cases** at initialization (t=0, year 2023):

| Age Band | Female Prevalence | Male Prevalence |
|----------|-------------------|-----------------|
| 65-79 | 3.12% (0.031152) | 2.92% (0.029153) |
| 80-100 | 22.06% (0.220598) | 15.98% (0.159755) |

**Verification**: Expected ~799,913 cases (within 0.01% of target)

### 4. Configuration Simplifications

#### Removed Age Bands (<65)
The following configuration sections have been updated to remove references to age bands below 65:

- **REPORTING_AGE_BANDS**: Now only includes (65-79) and (80+)
- **INCIDENCE_AGE_BANDS**: Starts at 65-69 (removed 0-64)
- **living_setting_transition_probabilities**: Removed (35, 65) age band
- **utility_norms_by_age**: Only includes ages 65 and 75
- **age_risk_multipliers**: Removed age 60 reference points

## Model Configuration

### Core Parameters (general_config)

```python
'number_of_timesteps': 17          # 2023-2040 (18 years)
'population': 10787479             # 65+ UK population
'base_year': 2023                  # t=0 corresponds to 2023
'time_step_years': 1               # Annual cycles
```

### Open Population Settings

```python
'open_population': {
    'use': True,
    'entrants_per_year': 600000,   # All enter at age 65
    'fixed_entry_age': 65,         # Fixed entry point
    'entrants_growth': {
        'use': True,
        'annual_rate': 0.0059088616,  # ~0.59% growth
        'reference_year': 2023
    }
}
```

### Incidence Growth (Disabled)

```python
'incidence_growth': {
    'use': True,
    'annual_rate': 0.0,            # No growth in onset hazard
    'reference_year': 2023
}
```

## Running the Model

### Basic Model Run

```python
import IBM_PD_AD_v2 as model

# Run with default configuration
results = model.run_model(model.general_config)

# Access results
print(f"Total QALYs (patients): {results['total_qalys_patient']:,.0f}")
print(f"Total costs (NHS): £{results['total_costs_nhs']:,.0f}")
print(f"Incident onsets: {results['incident_onsets']:,.0f}")
```

### Configuration Verification

Use the provided test script to verify configuration:

```bash
python test_v2_config.py
```

Expected output:
```
Population: 10,787,479
Entrants per year: 600,000
Fixed entry age: 65
Incidence growth rate: 0.0
Reporting age bands: [(65, 79), (80, None)]

Severe_to_death Relative Risks (all should be 1.00):
  smoking: Female=1.0, Male=1.0
  periodontal_disease: Female=1.0, Male=1.0
  ...

Total Expected Prevalent Cases: 799,913
Target: 800,000
```

## Probabilistic Sensitivity Analysis (PSA)

### Using run_psa_direct_v2.py

The v2 model includes a dedicated PSA script that runs analyses for multiple prevalence levels:

```bash
python run_psa_direct_v2.py
```

#### PSA Configuration

```python
PSA_ITERATIONS = 500              # Iterations per prevalence level
SCALE_FACTOR = 0.01               # 1% population (107,874 individuals)
PREVALENCE_LEVELS = [0.25, 0.50, 0.75]  # 25%, 50%, 75% PD prevalence
SEED = 42                         # Reproducibility
```

#### Output Directories

Results are saved to separate directories for each prevalence level:
- `psa_results_25_v2/` - 25% periodontal disease prevalence
- `psa_results_50_v2/` - 50% periodontal disease prevalence
- `psa_results_75_v2/` - 75% periodontal disease prevalence

Each directory contains:
1. **psa_results_XXpct.pkl.gz** - Compressed PSA results object
2. **PSA_Results_XXpct.xlsx** - Excel workbook with:
   - Summary_Scaled: Mean and 95% CIs (scaled to full population)
   - Metadata: PSA configuration and methods
   - Key_Results: Main outcomes formatted for manuscript
   - Validation: Methodology validation checks
   - PSA_Draws: Individual iteration results (first 10,000)
3. **methods_justification.txt** - Text for manuscript methods section

#### Estimated Runtime

- **Per prevalence level**: 3-6 hours (single core)
- **Total (3 levels)**: 9-18 hours
- **Population per iteration**: 107,874 (1% of 10,787,479)
- **Computational efficiency**: 100× reduction vs full population

### PSA Methodology

The PSA uses an efficient two-level nested design (O'Hagan et al., 2007):
1. Simulates 1% of the population per iteration (107,874 individuals)
2. Runs 500 iterations with parameter uncertainty
3. Scales absolute counts by 100× to full population
4. Validates that rates remain invariant after scaling

## File Structure

```
periodontal/
├── IBM_PD_AD.py                    # Original full-population model
├── IBM_PD_AD_v2.py                 # 65+ only model (this version)
├── run_psa_direct.py               # PSA script for original model
├── run_psa_direct_v2.py            # PSA script for v2 model
├── test_v2_config.py               # Configuration verification script
├── IBM_PD_AD_V2_README.md          # This file
├── psa_results_25_v2/              # PSA outputs (25% prevalence)
├── psa_results_50_v2/              # PSA outputs (50% prevalence)
└── psa_results_75_v2/              # PSA outputs (75% prevalence)
```

## Model Validation

### Initialization Checks

At t=0 (year 2023), the model should initialize with:
- **Total population**: 10,787,479 individuals
- **Age range**: All individuals aged ≥65
- **Prevalent dementia cases**: ~800,000
- **Age distribution**:
  - 72.73% in 65-79 age band
  - 27.27% in 80+ age band

### Runtime Checks

During simulation:
- **New entrants**: All enter at exactly age 65
- **QALY accrual**: Only for stages mild, moderate, severe
- **Incidence rate**: Constant over time (no growth)
- **Mortality**: No excess mortality from risk factors in severe stage

## Use Cases

### 1. Sensitivity Analysis
Test the impact of removing severe-to-death risk factor effects:
```python
# Compare original vs v2 for same prevalence
results_original = run_model(IBM_PD_AD.general_config)
results_v2 = run_model(IBM_PD_AD_v2.general_config)

# Analyze differences in mortality, QALYs, costs
```

### 2. Older Adult Focus
Analyze dementia burden specifically in the 65+ population:
```python
# V2 provides cleaner age-stratified outputs
# No need to filter out <65 age groups
```

### 3. QALY Burden Quantification
Isolate quality-of-life burden to dementia patients only:
```python
# V2: QALYs only reflect dementia stages
# Easier interpretation of QALY loss
```

### 4. Prevalence Sensitivity
Run PSA across multiple periodontal disease prevalence scenarios:
```bash
python run_psa_direct_v2.py
# Automatically runs 25%, 50%, 75% prevalence
```

## Important Notes

### Differences in Interpretation

1. **Population Denominators**:
   - V2 uses 65+ population (10.8M) vs full UK (33.2M)
   - Per-capita rates are specific to older adults only

2. **QALY Interpretation**:
   - V2 QALYs represent burden on dementia patients only
   - Cannot be directly compared to full-population QALYs

3. **Mortality**:
   - V2 removes excess mortality from risk factors in severe dementia
   - Deaths driven by age-specific background mortality only

4. **Incidence**:
   - V2 maintains constant incidence rate over time
   - Easier to attribute changes to age structure vs temporal trends

### When to Use V2 vs Original

**Use IBM_PD_AD_v2 when:**
- Focusing on older adult population (65+)
- Testing sensitivity to severe-to-death risk factor effects
- Isolating dementia-specific QALY burden
- Avoiding confounding from incidence growth over time
- Running prevalence sensitivity analyses

**Use IBM_PD_AD (original) when:**
- Modeling full population across all ages
- Including younger adults at risk
- Estimating population-wide burden
- Including risk factor effects on mortality
- Matching ONS full-population projections

## References

- **Original Model**: IBM_PD_AD.py (full UK population, ages 35-100+)
- **PSA Methodology**: O'Hagan A, Stevenson M, Madan J. Monte Carlo probabilistic sensitivity analysis for patient level simulation models: efficient estimation of mean and variance using ANOVA. Health Economics. 2007;16(10):1009-1023.

## Version History

- **Version 2.0** (2026-01-21): Initial release
  - 65+ population only
  - Severe-to-death RRs set to 1.00
  - QALYs for dementia patients only
  - Zero incidence growth
  - 800k initial prevalent cases
  - 600k entrants/year at age 65

## Contact & Support

For questions about model implementation or results interpretation, refer to:
- Configuration verification: `test_v2_config.py`
- PSA execution: `run_psa_direct_v2.py`
- Original documentation: IBM_PD_AD.py docstrings

---

**Model Type**: Individual-Based Model (IBM) / Microsimulation
**Target Population**: UK adults aged 65 and over
**Time Horizon**: 2023-2040 (18 years, annual cycles)
**Perspective**: NHS and societal (informal care costs included)
**Currency**: GBP (British Pounds)
**Discount Rate**: 3.5% annual (end-of-cycle)
