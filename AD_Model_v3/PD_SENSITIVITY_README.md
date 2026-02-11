# Periodontal Disease Sensitivity Analysis for Tornado Diagrams

## Overview

This module provides computationally efficient **one-way sensitivity analysis** specifically for periodontal disease (PD) relative risk parameters:

1. **PD Onset Relative Risk**: 1.47 (95% CI: 1.32-1.65)
2. **PD Severe-to-Death Relative Risk**: 1.36 (95% CI: 1.10-1.69)

Traditional tornado diagram sensitivity analysis would require hundreds of full population runs, making it computationally prohibitive. This implementation uses **population reduction with statistical replication** to make the analysis feasible while maintaining accuracy.

## Method: Population Reduction with Scaling

### The Computational Challenge

For a traditional tornado diagram with full population:
- **Population**: 100,000 agents
- **Parameters**: 2 (onset RR, severe-to-death RR)
- **Values per parameter**: 2 (low 95% CI, high 95% CI)
- **Total runs**: 1 baseline + (2 params × 2 values) = 5 runs
- **Run time per simulation**: ~5 minutes
- **Total time**: ~25 minutes (without replicates)
- **With 10 replicates**: ~4 hours

### Our Solution

We use **1% of the population** (1,000 agents) with **10 statistical replicates**:
- **Reduced population**: 1,000 agents (1% of 100,000)
- **Replicates per parameter value**: 10
- **Total runs**: 10 baseline + (2 params × 2 values × 10 reps) = 50 runs
- **Run time per simulation**: ~5 seconds
- **Total time with parallel processing**: **~5-10 minutes** (on 8+ cores)

### Why This Works

1. **Law of Large Numbers**: Multiple replicates with smaller populations average out to similar results as single runs with large populations

2. **Statistical Equivalence**:
   - 1 run × 100,000 agents ≈ 10 runs × 10,000 agents (in expectation)
   - We use 10 runs × 1,000 agents for even faster computation

3. **Linear Scaling**: Key metrics (costs, QALYs, cases) scale linearly with population
   - 1,000 agents → results × 100 → 100,000 agent-equivalent

4. **Variance Reduction**: Replicates reduce stochastic noise, giving more stable swing estimates

## Files

### `pd_sensitivity_analysis.py`
Core module containing:
- `run_pd_sensitivity_analysis()`: Main sensitivity analysis function
- `create_pd_tornado_diagram()`: Tornado diagram visualization
- `export_pd_sensitivity_results()`: Excel export with summary statistics

### `run_pd_tornado.py`
Example script demonstrating complete workflow:
1. Load configuration
2. Run sensitivity analysis
3. Generate tornado diagrams
4. Export results to Excel

## Usage

### Quick Start

```python
from IBM_PD_AD import DEFAULT_CONFIG
from pd_sensitivity_analysis import (
    run_pd_sensitivity_analysis,
    create_pd_tornado_diagram,
    export_pd_sensitivity_results
)

# Run analysis
results = run_pd_sensitivity_analysis(
    DEFAULT_CONFIG,
    population_fraction=0.01,  # 1% of population
    n_replicates=10,           # 10 replicates per value
    seed=42,                   # Reproducibility
    n_jobs=-1                  # All CPU cores
)

# Create tornado diagram
create_pd_tornado_diagram(
    results,
    save_path='plots/pd_tornado.png'
)

# Export to Excel
export_pd_sensitivity_results(
    results,
    excel_path='pd_sensitivity.xlsx'
)
```

### From Command Line

```bash
python run_pd_tornado.py
```

This will:
1. Run sensitivity analysis with default settings
2. Generate tornado diagrams in `plots/` directory
3. Export results to `pd_sensitivity_analysis.xlsx`

## Parameters

### `run_pd_sensitivity_analysis()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_config` | Required | Base model configuration dictionary |
| `population_fraction` | 0.01 | Fraction of population to use (0.01 = 1%) |
| `n_replicates` | 10 | Number of replicates per parameter value |
| `combine_sexes` | True | Vary both sexes together vs. separately |
| `seed` | None | Random seed for reproducibility |
| `n_jobs` | None | Parallel jobs (None = auto, -1 = all cores) |

### Adjusting Computational Cost

**Faster (less accurate)**:
```python
results = run_pd_sensitivity_analysis(
    config,
    population_fraction=0.01,  # 1% population
    n_replicates=5,            # Fewer replicates
    n_jobs=-1
)
# ~2-3 minutes
```

**More accurate (slower)**:
```python
results = run_pd_sensitivity_analysis(
    config,
    population_fraction=0.05,  # 5% population
    n_replicates=20,           # More replicates
    n_jobs=-1
)
# ~30-60 minutes
```

**For publication quality**:
```python
results = run_pd_sensitivity_analysis(
    config,
    population_fraction=0.10,  # 10% population
    n_replicates=50,           # Many replicates
    n_jobs=-1
)
# ~2-4 hours
```

## Output

### Tornado Diagrams (`plots/pd_tornado_*.png`)

Visual representation showing parameter influence on outcomes:
- **Horizontal bars**: Range from low to high parameter values
- **Bar width**: Parameter influence (swing)
- **Red dashed line**: Baseline value
- **Endpoints**: Low and high 95% CI bounds

Generated for metrics:
- Total QALYs
- Dementia incidence (cases)
- Total costs

### Excel Results (`pd_sensitivity_analysis.xlsx`)

**Sheet 1: Raw_Results**
- All replicate-level data
- Columns: parameter, value_type, replicate, all metrics

**Sheet 2: Summary_Statistics**
- Mean and std dev for each parameter/value combination
- Difference from baseline
- Percent change from baseline

**Sheet 3: Swing_Analysis**
- Parameter influence rankings
- Absolute and relative swings
- Sorted by influence

## Validation

To validate that population reduction with scaling works:

```python
# Run with different population fractions
for frac in [0.01, 0.05, 0.10]:
    results = run_pd_sensitivity_analysis(
        config,
        population_fraction=frac,
        n_replicates=10,
        seed=42
    )

    # Compare swings across fractions
    # They should be similar if scaling is accurate
```

Expected: Similar swing magnitudes across fractions (within ~10%)

## Interpreting Results

### Tornado Diagram

The **wider the bar**, the **more influential** the parameter.

Example interpretation:
```
PD Onset RR (1.32-1.65)           |████████████████|
PD Severe→Death RR (1.10-1.69)    |████|
```

→ Onset RR has ~4× more influence than severe-to-death RR

### Swing Analysis

**QALY Swing Example**:
- Baseline: 10,000,000 QALYs
- Onset RR Low (1.32): 10,150,000 QALYs (+150,000)
- Onset RR High (1.65): 9,850,000 QALYs (-150,000)
- **Swing**: 300,000 QALYs (3% of baseline)

**Case Swing Example**:
- Baseline: 50,000 cases
- Onset RR Low (1.32): 48,500 cases (-1,500)
- Onset RR High (1.65): 51,500 cases (+1,500)
- **Swing**: 3,000 cases (6% of baseline)

## Technical Notes

### Why Not Standard PSA?

Probabilistic sensitivity analysis (PSA) varies **all parameters simultaneously** and shows overall uncertainty. Tornado diagrams require **one-at-a-time variation** to isolate each parameter's influence.

### Population Scaling

The module automatically scales these metrics by population fraction:
- Costs (NHS, informal, total)
- QALYs (patient, caregiver, combined)
- Dementia incidence
- Stage prevalence

**Not scaled**: Rates, proportions, means (already population-independent)

### Confidence Intervals

The 95% CI bounds used for low/high values come from:
- Literature meta-analyses
- Defined in `RISK_FACTOR_HR_INTERVALS` in `IBM_PD_AD.py`

**Periodontal Disease**:
```python
'periodontal_disease': {
    'onset': {
        'female': (1.47, 1.32, 1.65),  # (mean, lower_95, upper_95)
        'male': (1.47, 1.32, 1.65),
    },
    'severe_to_death': {
        'female': (1.36, 1.10, 1.69),
        'male': (1.36, 1.10, 1.69),
    },
}
```

## Troubleshooting

### "Running too slowly"

**Solution 1**: Reduce replicates
```python
n_replicates=5  # Instead of 10
```

**Solution 2**: Use smaller population fraction
```python
population_fraction=0.005  # 0.5% instead of 1%
```

**Solution 3**: Ensure parallel processing
```python
n_jobs=-1  # Use all cores
```

### "Results seem noisy"

**Solution**: Increase replicates
```python
n_replicates=20  # More replicates = less noise
```

### "Swings don't match expectations"

**Solution**: Validate scaling
1. Run with multiple population fractions
2. Check if swings are proportional
3. Verify baseline matches full population run

### Memory Issues

If running on limited RAM:
```python
# Run sequentially instead of parallel
n_jobs=1

# Or reduce replicates
n_replicates=5
```

## Extending the Analysis

### Adding More Parameters

To analyze additional parameters (e.g., other risk factors):

1. Modify `parameters` dict in `run_pd_sensitivity_analysis()`
2. Add parameter modification logic in `set_pd_rr()` function
3. Update parameter labels in `create_pd_tornado_diagram()`

### Two-Way Sensitivity Analysis

For interaction effects between onset and severe-to-death RRs:

```python
# Test all combinations
for onset_rr in [1.32, 1.47, 1.65]:
    for std_rr in [1.10, 1.36, 1.69]:
        # Run model with both values
        # Create heatmap of results
```

## References

**Population Reduction Methods**:
- O'Hagan, A., et al. (2007). "Techniques for multiparameter sensitivity analysis of model output."
- Briggs, A., Claxton, K., & Sculpher, M. (2006). "Decision Modelling for Health Economic Evaluation."

**Tornado Diagrams**:
- Saltelli, A., et al. (2008). "Global Sensitivity Analysis: The Primer."

## Contact

For questions or issues with this sensitivity analysis module, please contact the model developers or consult the main `IBM_PD_AD.py` documentation.
