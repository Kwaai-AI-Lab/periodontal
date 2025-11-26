# O'Hagan Two-Level PSA Method - Implementation Guide

## Overview

This document explains how to use the **O'Hagan et al. (2007) ANOVA-based PSA method** implemented in your Alzheimer's disease microsimulation model. This method dramatically reduces computational cost for probabilistic sensitivity analysis by using variance decomposition to determine optimal sample sizes.

## The Problem

Traditional PSA for your model would require:
- **1000 parameter sets** × **33,167,098 individuals** = **33+ billion individual simulations**
- Estimated runtime: **Days to weeks** even with parallelization

## The Solution: Two-Level Nested Design

The O'Hagan method recognizes that outcome variance comes from two sources:

1. **Parameter uncertainty (σ²_between)**: What PSA aims to capture
2. **Stochastic noise (σ²_within)**: Random variation from individual patient sampling

**Key insight**: You don't need 33M individuals per PSA iteration! You only need enough to accurately estimate the mean outcome for each parameter set.

### Variance Decomposition

```
Total Variance = σ²_between + σ²_within/n
```

Where:
- **σ²_between**: Between-parameter-set variance (the PSA signal)
- **σ²_within**: Within-parameter-set variance (Monte Carlo noise)
- **n**: Number of individuals per parameter set

### Optimal Allocation

For a given computational budget, the optimal strategy is:
- **Maximize N** (number of parameter sets)
- **Minimize n** (individuals per set)
- Subject to: `σ²_within/n ≤ threshold × σ²_between`

Typical threshold: 0.1 to 0.25 (MC noise should be 10-25% of PSA signal)

## Two-Step Implementation

### Step 1: Run Pilot Study to Estimate Variance Components

```python
# Run pilot to determine optimal sample size
pilot_results = estimate_variance_components_pilot(
    general_config,
    general_config['psa'],
    n_outer=10,  # Test with 10 parameter sets
    n_inner_list=[1000, 5000, 10000, 50000],  # Test different population sizes
    seed=42
)

# The pilot will print:
# - Variance components (σ²_between and σ²_within) for each outcome
# - Optimal n recommendations
# - Example: "RECOMMENDATION: Use n ≈ 8,000 individuals per PSA iteration"
```

**What the pilot does:**
1. Runs your model with 10 different parameter sets
2. Tests each with different population sizes (1K, 5K, 10K, 50K)
3. Fits the variance model: `Var(ȳ) = σ²_between + σ²_within/n`
4. Calculates optimal n where MC noise is negligible compared to parameter uncertainty

**Typical runtime**: 10-30 minutes (depends on n_inner_list)

### Step 2: Run Full Two-Level PSA

```python
# Run efficient PSA using pilot recommendations
psa_results = run_two_level_psa(
    general_config,
    general_config['psa'],
    n_outer=1000,  # Full 1000 parameter sets
    variance_pilot_results=pilot_results,  # Uses recommended n automatically
    seed=42,
    n_jobs=None  # Use all CPU cores for parallelization
)

# Access results
summary = psa_results['summary']
print(f"Total QALYs: {summary['total_qalys']['mean']:.0f}")
print(f"  95% CI: [{summary['total_qalys']['lower_95']:.0f}, "
      f"{summary['total_qalys']['upper_95']:.0f}]")
```

**What this does:**
- Runs 1000 PSA iterations with optimized population size (e.g., 8K instead of 33M)
- Uses parallelization for speed
- Produces accurate 95% confidence intervals
- Saves **massive** computational time

## Complete Example Workflow

```python
if __name__ == "__main__":
    run_seed = 42

    # ========================================
    # 1. Run baseline model (full population)
    # ========================================
    print("Step 1: Running baseline model...")
    model_results = run_model(general_config, seed=run_seed)
    save_results_compressed(model_results, 'results/baseline_model.pkl.gz')
    export_results_to_excel(model_results)  # Excel export still works!

    # ========================================
    # 2. Run pilot study to find optimal n
    # ========================================
    print("\nStep 2: Running variance components pilot study...")
    pilot_results = estimate_variance_components_pilot(
        general_config,
        general_config['psa'],
        n_outer=10,
        n_inner_list=[1000, 5000, 10000, 50000],
        seed=run_seed
    )

    # Save pilot results for future reference
    save_results_compressed(pilot_results, 'results/pilot_study.pkl.gz')

    # ========================================
    # 3. Run efficient two-level PSA
    # ========================================
    print("\nStep 3: Running two-level PSA...")
    psa_results = run_two_level_psa(
        general_config,
        general_config['psa'],
        n_outer=1000,  # Full PSA with 1000 iterations
        variance_pilot_results=pilot_results,
        collect_draw_level=True,  # Keep all draw data
        seed=run_seed,
        n_jobs=None  # Parallel processing
    )

    # Save results
    save_results_compressed(psa_results, 'results/two_level_psa.pkl.gz')

    # Export PSA results to Excel
    export_psa_results_to_excel(
        psa_results,
        path='results/PSA_Results.xlsx',
        include_draws=False  # Set True to include all iterations (up to 10K rows)
    )

    # Save draw-level data as compressed Parquet (for full data)
    if 'draws' in psa_results:
        save_dataframe_compressed(
            psa_results['draws'],
            'results/psa_draws.parquet',
            format='parquet'
        )

    # ========================================
    # 4. Print summary
    # ========================================
    print("\n" + "="*60)
    print("PSA RESULTS SUMMARY")
    print("="*60)

    summary = psa_results['summary']
    for metric, stats in summary.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:,.2f}")
        print(f"  95% CI: [{stats['lower_95']:,.2f}, {stats['upper_95']:,.2f}]")

    # Print efficiency gains
    two_level_info = psa_results['two_level_psa']
    print(f"\n{'='*60}")
    print("COMPUTATIONAL EFFICIENCY")
    print(f"{'='*60}")
    print(f"Original population: {two_level_info['original_population']:,}")
    print(f"Reduced population (n): {two_level_info['n_inner']:,}")
    print(f"Reduction factor: {two_level_info['reduction_factor']:.1f}x")
    print(f"Parameter sets (N): {two_level_info['n_outer']}")
    print(f"Total individuals simulated: {two_level_info['n_outer'] * two_level_info['n_inner']:,}")
    print(f"vs. traditional PSA: {two_level_info['n_outer'] * two_level_info['original_population']:,}")
```

## Expected Computational Savings

### Example Scenario

Assumptions:
- Original population: 33,167,098
- Pilot study finds optimal n = 10,000
- PSA iterations: 1000

**Traditional PSA:**
- Total individuals: 1000 × 33,167,098 = 33+ billion
- Estimated runtime: ~10-14 days (parallelized on 8 cores)

**Two-Level PSA (O'Hagan method):**
- Total individuals: 1000 × 10,000 = 10 million
- Estimated runtime: ~8-12 hours (parallelized on 8 cores)
- **Speedup: ~3,300x fewer individuals, ~30x faster**

### Accuracy Verification

The 95% confidence intervals from the two-level PSA should be nearly identical to traditional PSA because:
- Parameter uncertainty dominates the CI width
- Stochastic noise (σ²_within/n) is kept small relative to parameter variance

## Exporting PSA Results

### Excel Export

Use the dedicated PSA Excel export function to save your results in a structured workbook:

```python
# Export PSA results to Excel
export_psa_results_to_excel(
    psa_results,
    path='results/PSA_Results.xlsx',
    include_draws=False  # Set True to include individual iterations (up to 10K)
)
```

**Excel file contains multiple sheets:**

1. **PSA_Summary**: Mean and 95% CI for all outcome metrics
   - Metric name
   - Mean value
   - Lower 95% CI
   - Upper 95% CI
   - CI width

2. **PSA_Metadata**: Run configuration details
   - PSA iterations
   - Parallel jobs used
   - Method (Standard or Two-Level)
   - Population sizes (N and n if two-level)
   - Reduction factor

3. **Variance_Components** (two-level PSA only):
   - σ²_between (parameter uncertainty)
   - σ²_within (stochastic noise)
   - Signal-to-noise ratio for each metric

4. **All_Draws** (optional, if include_draws=True):
   - Individual results from each PSA iteration
   - Limited to first 10,000 rows (Excel limits)
   - For complete data, use `save_dataframe_compressed(..., format='parquet')`

### Note on Model vs. PSA Exports

There are now **two separate Excel export functions**:

1. **`export_results_to_excel(model_results)`** - For single baseline model runs
   - Time series data
   - Age distributions
   - Prevalence by stage
   - Incidence rates
   - Use after: `run_model()`

2. **`export_psa_results_to_excel(psa_results)`** - For PSA uncertainty analysis
   - Mean and 95% confidence intervals
   - Variance decomposition
   - PSA metadata
   - Use after: `run_probabilistic_sensitivity_analysis()` or `run_two_level_psa()`

**Example using both:**

```python
# Run baseline model
baseline_results = run_model(general_config, seed=42)
export_results_to_excel(baseline_results, path='Baseline_Model.xlsx')

# Run PSA
psa_results = run_two_level_psa(general_config, general_config['psa'], ...)
export_psa_results_to_excel(psa_results, path='PSA_Results.xlsx')
```

## Advanced Usage

### Manual n Specification

If you already know the optimal n or want to test a specific value:

```python
psa_results = run_two_level_psa(
    general_config,
    general_config['psa'],
    n_outer=1000,
    n_inner=10000,  # Explicitly set n
    seed=42
)
```

### Testing Different n Values

```python
# Test sensitivity to n
for n_test in [5000, 10000, 20000]:
    results = run_two_level_psa(
        general_config,
        general_config['psa'],
        n_outer=100,  # Smaller N for testing
        n_inner=n_test,
        seed=42
    )
    print(f"\nn={n_test:,}:")
    print(f"  Total QALYs 95% CI width: "
          f"{results['summary']['total_qalys']['upper_95'] - results['summary']['total_qalys']['lower_95']:.0f}")
```

### Re-using Pilot Results

Save time on subsequent runs:

```python
# Load previously saved pilot results
pilot_results = load_results_compressed('results/pilot_study.pkl.gz')

# Run PSA with different configurations using same optimal n
psa_scenario_1 = run_two_level_psa(
    config_scenario_1,
    config_scenario_1['psa'],
    n_outer=1000,
    variance_pilot_results=pilot_results,  # Re-use pilot
    seed=42
)

psa_scenario_2 = run_two_level_psa(
    config_scenario_2,
    config_scenario_2['psa'],
    n_outer=1000,
    variance_pilot_results=pilot_results,  # Re-use pilot
    seed=43
)
```

## Interpreting Pilot Study Output

### Variance Components

```
total_qalys:
  σ²_between (parameter uncertainty): 3.456e+12
  σ²_within (stochastic noise): 8.234e+10
  Ratio (signal/noise): 41.98
```

**Interpretation:**
- Parameter uncertainty is 42x larger than stochastic noise
- This is ideal - most variance comes from parameters (what PSA should capture)
- High ratio → can use smaller n

### Optimal n Recommendations

```
For total_qalys:
  Conservative n (MC noise < 10% of PSA signal): 23,811
  Moderate n (MC noise < 25% of PSA signal): 9,524
```

**Interpretation:**
- **Conservative**: MC noise contributes <10% of total variance (very accurate)
- **Moderate**: MC noise contributes <25% of total variance (usually sufficient)
- Use moderate for most analyses
- Use conservative if you need very precise CI estimates

## When to Use This Method

### ✅ Use Two-Level PSA when:
- Model runtime is substantial (>10 minutes per full run)
- You need 500+ PSA iterations for robust CIs
- You're simulating large populations (millions of individuals)
- Parameter uncertainty exists (PSA makes sense)

### ❌ Don't use when:
- Model runs very fast (<1 minute)
- You only need 100-200 PSA iterations
- Population size is already small (<10,000)
- No parameter uncertainty (deterministic model)

## Validation

### Compare Two-Level vs. Traditional PSA

```python
# Run traditional PSA (if feasible)
traditional_psa = run_probabilistic_sensitivity_analysis(
    general_config,
    general_config['psa'],
    seed=42
)

# Run two-level PSA
two_level_psa = run_two_level_psa(
    general_config,
    general_config['psa'],
    n_outer=1000,
    variance_pilot_results=pilot_results,
    seed=42
)

# Compare CI widths
for metric in ['total_qalys', 'total_costs']:
    trad_width = (traditional_psa['summary'][metric]['upper_95'] -
                  traditional_psa['summary'][metric]['lower_95'])
    two_level_width = (two_level_psa['summary'][metric]['upper_95'] -
                       two_level_psa['summary'][metric]['lower_95'])

    print(f"{metric}:")
    print(f"  Traditional CI width: {trad_width:,.0f}")
    print(f"  Two-level CI width: {two_level_width:,.0f}")
    print(f"  Difference: {abs(trad_width - two_level_width)/trad_width * 100:.1f}%")
```

Expect <5% difference in CI widths if n is chosen appropriately.

## Troubleshooting

### Issue: Pilot study recommends very large n

**Possible causes:**
1. Stochastic noise dominates parameter uncertainty
2. Not enough parameter uncertainty in the PSA distributions
3. Model outcomes are highly variable

**Solutions:**
- Check PSA distributions are appropriate
- Consider if PSA is necessary (low parameter uncertainty)
- May need to use larger n or increase parameter uncertainty ranges

### Issue: PSA results differ significantly from traditional PSA

**Likely cause**: n is too small (stochastic noise affecting results)

**Solution:**
- Use conservative n recommendation instead of moderate
- Run pilot with larger n_inner_list to verify variance estimates
- Check variance ratio (signal/noise) - should be >10 for most outcomes

### Issue: Pilot study takes too long

**Solutions:**
- Reduce n_outer (can use as few as 5-8 parameter sets)
- Use smaller values in n_inner_list
- The pilot only needs to run once - can re-use results

## References

**O'Hagan, A., Stevenson, M., & Madan, J. (2007).** "Monte Carlo probabilistic sensitivity analysis for patient level simulation models: efficient estimation of mean and variance using ANOVA." *Health Economics*, 16(10), 1009-1023.

**Key concepts:**
- Two-level nested Monte Carlo design
- ANOVA variance decomposition
- Optimal sample size allocation for PSA
- Reducing computational burden of microsimulation PSA

## Summary

The O'Hagan two-level PSA method:

1. **Runs a small pilot study** to estimate variance components
2. **Determines optimal population size** (n) per PSA iteration
3. **Runs efficient PSA** with N parameter sets × n individuals
4. **Produces accurate 95% CIs** at fraction of computational cost

**Expected savings for your model:**
- **~3,000x fewer individuals** to simulate
- **~30x faster runtime**
- **Identical accuracy** for confidence intervals

This makes 1000-iteration PSA feasible for your 33M-person Alzheimer's disease microsimulation!
