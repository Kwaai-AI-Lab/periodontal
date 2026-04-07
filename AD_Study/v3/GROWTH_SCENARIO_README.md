# Growth Scenario Implementation - README

## Overview

This implementation adds time-varying periodontal disease prevalence to model the projected growth in periodontal disease burden based on Elamin & Anash (2023).

## Changes Made

### 1. IBM_PD_AD_v3.py Modifications

**Added prevalence schedule configuration** (lines ~1751-1764):
- `prevalence_schedule['use']`: Toggle for growth scenario (default: False)
- `baseline_year`: 2023
- `target_year`: 2040
- `baseline_prevalence`: 50% (both sexes)
- `target_prevalence`: 61.25% (both sexes)
- `interpolation`: Linear

**Updated functions**:
- `get_prevalence_for_person()`: Now calculates time-varying prevalence when enabled
- `assign_risk_factors()`: Now accepts `calendar_year` parameter
- `initialize_population()`: Passes `base_year` for initial prevalence
- `add_new_entrants()`: Passes `calendar_year` for new entrant prevalence

### 2. New Script: run_growth_scenario_analysis.py

Complete analysis pipeline for growth scenario:
- **Step 1**: Main model run (full population)
- **Step 2**: PSA - 500 iterations at 1% population
- **Step 3**: Tornado analysis - varies HR bounds (1.07 to 1.38)

## Scientific Basis

**Source**: Elamin & Anash (2023) - Periodontal pocketing projections for UK population 60+

**Original projection**:
- 39.7% relative increase in periodontal pocketing (2020-2050, 30 years)

**Adapted to our model**:
- Time period: 2023-2040 (17 years instead of 30)
- Calculation: 39.7% ÷ 30 years × 17 years = 22.5% relative increase
- Age group: 65+ (within their 60+ cohort, minimal adjustment needed)
- Baseline (2023): 50% prevalence
- Target (2040): 61.25% prevalence (50% × 1.225)
- Interpolation: Linear growth year-by-year

## Two-Scenario Comparison

### Scenario 1: Stable (Baseline) - Already completed
- Constant 50% prevalence throughout 2023-2040
- Represents status quo / no change in disease burden

### Scenario 2: Growth (New) - To be run
- Dynamic prevalence: 50% → 61.25% (2023-2040)
- Represents projected epidemiological trend
- Based on Elamin & Anash (2023) projections

## How to Run

### Quick Start - Run Everything
```bash
cd AD_Model_v3
python run_growth_scenario_analysis.py
```

This runs all three steps sequentially:
1. Main model (full population, ~2-4 hours)
2. PSA (500 iterations at 1%, ~3-5 hours)
3. Tornado (varies HR, ~1-2 hours)

**Total estimated time**: 6-11 hours depending on CPU

### Output Files

**Main model**:
- `growth_scenario_results/main_model_run/results_growth_scenario.pkl.gz`
- `results/Growth_Scenario_PD_Trend.xlsx`

**PSA**:
- `growth_scenario_results/psa/PSA_Growth_Scenario.xlsx`

**Tornado**:
- `growth_scenario_results/tornado/tornado_growth_scenario.xlsx`

**Log file**:
- `growth_scenario_results/growth_scenario_log.txt`

## Tornado Analysis Details

The tornado analysis varies the **PD onset hazard ratio** while keeping the growth scenario enabled:

- **Baseline**: HR = 1.21 (point estimate)
- **Low**: HR = 1.07 (lower 95% CI)
- **High**: HR = 1.38 (upper 95% CI)

Each scenario runs with:
- 10 replicates
- 1% population (~107,875 agents)
- Results scaled back to full population

**Purpose**: Assess sensitivity of results to uncertainty in the PD-dementia association strength.

## Prevalence Schedule Verification

Test the time-varying prevalence:
```python
from IBM_PD_AD_v3 import general_config, get_prevalence_for_person
import copy

# Enable growth scenario
cfg = copy.deepcopy(general_config)
cfg['risk_factors']['periodontal_disease']['prevalence_schedule']['use'] = True
pd_meta = cfg['risk_factors']['periodontal_disease']

# Check prevalence at different years
for year in [2023, 2030, 2040]:
    prev = get_prevalence_for_person(pd_meta, age=65, sex='female', calendar_year=year)
    print(f'{year}: {prev:.2%}')
```

Expected output:
```
2023: 50.00%
2030: 54.63%
2040: 61.25%
```

## Comparison with Baseline

After running both scenarios, compare:

### Health Outcomes
- Dementia incidence (total cases)
- QALYs lost (patient + caregiver)
- Years of life lost

### Economic Outcomes
- NHS costs
- Informal care costs
- Societal costs

### Interpretation
The **difference** between growth and stable scenarios shows the **additional burden attributable to the projected increase in periodontal disease prevalence**.

## Technical Notes

### Why Linear Interpolation?
- Conservative approach
- Aligns with epidemiological modeling conventions
- More defensible than compound growth for a 17-year window

### Why 22.5% Instead of 39.7%?
- 39.7% is over 30 years (2020-2050)
- Our model runs 17 years (2023-2040)
- Linear pro-rating: (39.7% / 30) × 17 = 22.5%
- This is a conservative adjustment

### Population Scaling in PSA & Tornado
- PSA runs at 1% population (107,875 agents)
- Count metrics are scaled back up by 100x
- Rate/ratio metrics are NOT scaled (inherently intensive)
- This maintains statistical validity while reducing runtime

## Next Steps

1. **Run the analysis**:
   ```bash
   python run_growth_scenario_analysis.py
   ```

2. **Compare with baseline results** (already completed):
   - Load both Excel files
   - Calculate deltas
   - Focus on 2040 outcomes

3. **Update manuscript**:
   - Replace arbitrary prevalence levels (25%, 50%, 75%)
   - Present two scenarios: stable vs. growth
   - Cite Elamin & Anash (2023) for projections
   - Emphasize policy implications of disease trend

4. **Generate comparison figures**:
   - Time series: Growth vs. Stable
   - Tornado comparison: Growth vs. Stable
   - Cost trajectories

## Troubleshooting

**If the script fails**:
1. Check the log file: `growth_scenario_results/growth_scenario_log.txt`
2. Verify IBM_PD_AD_v3.py has all modifications
3. Ensure sufficient disk space (~5-10 GB for results)
4. Check memory (PSA requires ~8-16 GB RAM)

**If prevalence doesn't vary**:
- Verify `prevalence_schedule['use'] = True`
- Check calendar_year is being passed correctly
- Test with the verification script above

## References

Elamin A, Anash R. Projections of periodontal disease burden in the United Kingdom. *Journal Name*. 2023;XX(X):XXX-XXX.

---

**Implementation date**: 2026-03-11
**Model version**: IBM_PD_AD_v3 (65+ cohort, hazard-based)
