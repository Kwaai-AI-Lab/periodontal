# Performance Optimizations Guide

This document describes the performance optimizations implemented for the Alzheimer's Disease microsimulation model.

## Overview

The model simulates 33+ million individuals over 17 years with 1000 Monte Carlo iterations for probabilistic sensitivity analysis (PSA). Without optimization, this can take days to complete. The optimizations below can reduce runtime significantly.

## 1. Parallel Processing for PSA

### What Changed
The 1000 PSA iterations now run in parallel across all available CPU cores by default.

### Expected Speed-Up
- **4-core CPU**: ~4x faster
- **8-core CPU**: ~8x faster
- **16-core CPU**: ~12-15x faster (with some overhead)

For example, if 1000 iterations took 24 hours serially, on an 8-core machine they might complete in ~3-4 hours.

### Configuration

In `general_config['psa']`:

```python
'psa': {
    'use': True,
    'iterations': 1000,
    'n_jobs': None,  # Use all CPU cores (default)
    # 'n_jobs': 8,   # Explicitly use 8 cores
    # 'n_jobs': 1,   # Disable parallelization (run serially)
    ...
}
```

**Options:**
- `n_jobs: None` (default) - Uses all available CPU cores
- `n_jobs: N` - Uses N cores (useful to limit resource usage)
- `n_jobs: 1` - Runs serially (for debugging or memory-constrained systems)

### Memory Considerations

Parallel processing uses more memory since multiple model runs execute simultaneously. Each worker process needs memory for:
- The full model configuration (~few MB)
- Population state during simulation (~GB range per process)

**Recommendations:**
- Systems with **<16 GB RAM**: Set `n_jobs: 2` or `4`
- Systems with **16-32 GB RAM**: Use default (all cores)
- Systems with **>32 GB RAM**: Use default (all cores)

If you encounter memory errors, reduce `n_jobs`.

## 2. Progress Tracking

### Installation (Optional but Recommended)
```bash
pip install tqdm
```

### What You'll See

**With tqdm installed:**
```
Running PSA with 1000 iterations using 8 parallel job(s)...
Progress tracking enabled (tqdm installed)
PSA iterations: 45%|████████▌         | 450/1000 [02:15<02:45, 3.3it/s]
```

**Without tqdm:**
```
Running PSA with 1000 iterations using 8 parallel job(s)...
Install tqdm for progress tracking: pip install tqdm
(no progress bar, but still works)
```

Progress tracking works in both serial (`n_jobs=1`) and parallel modes.

## 3. Output Compression

### New Functions

Three new functions are available to compress outputs:

#### 3.1 `save_results_compressed()`
Saves entire model results dictionaries in compressed format.

```python
# Run model
model_results = run_model(general_config, seed=42)

# Save compressed (typically 5-10x smaller than uncompressed pickle)
save_results_compressed(model_results, 'results/model_output.pkl.gz')

# Load later
loaded_results = load_results_compressed('results/model_output.pkl.gz')
```

**Compression levels:**
- `compression_level=1`: Fast, larger files (~3x compression)
- `compression_level=6`: Balanced (default, ~5-7x compression)
- `compression_level=9`: Maximum compression, slower (~8-10x compression)

#### 3.2 `save_dataframe_compressed()`
Saves DataFrames in compressed Parquet or CSV format.

```python
# For PSA results DataFrame
psa_results = run_probabilistic_sensitivity_analysis(...)
draws_df = psa_results['draws']  # If collect_draw_level=True

# Parquet format (recommended - better compression + preserves dtypes)
save_dataframe_compressed(draws_df, 'results/psa_draws.parquet', format='parquet')

# Or compressed CSV
save_dataframe_compressed(draws_df, 'results/psa_draws.csv.gz', format='csv')

# Read back
import pandas as pd
df = pd.read_parquet('results/psa_draws.parquet')
# or
df = pd.read_csv('results/psa_draws.csv.gz')
```

**Format comparison:**
- **Parquet**: Best compression, preserves dtypes, fast read/write, requires `pyarrow`
- **CSV.gz**: Good compression, human-readable when unzipped, universally compatible

### Example: Compressed Workflow

```python
if __name__ == "__main__":
    run_seed = 42

    # Run initial model
    print("Running initial model...")
    model_results = run_model(general_config, seed=run_seed)
    save_results_compressed(model_results, 'results/baseline_model.pkl.gz')

    # Run PSA with parallelization
    psa_cfg = general_config.get('psa', {})
    if psa_cfg.get('use', False):
        print("\nRunning PSA...")
        psa_results = run_probabilistic_sensitivity_analysis(
            general_config,
            psa_cfg,
            collect_draw_level=True,  # Keep all draw-level data
            seed=run_seed,
        )

        # Save compressed outputs
        save_results_compressed(psa_results, 'results/psa_results.pkl.gz')

        # If you collected draw-level data, save it separately
        if 'draws' in psa_results:
            save_dataframe_compressed(
                psa_results['draws'],
                'results/psa_draws.parquet',
                format='parquet'
            )
```

## 4. Installation Requirements

### Required (Already Installed)
- `numpy`
- `pandas`
- `matplotlib`

### New Dependencies (for full optimization)
```bash
pip install tqdm          # For progress bars
pip install pyarrow       # For Parquet compression (optional but recommended)
```

**Note:** The code works without these packages but with reduced functionality:
- Without `tqdm`: No progress bars
- Without `pyarrow`: Can't use Parquet format (CSV.gz still works)

## 5. Benchmarking Your System

To test the performance improvement on your system:

```python
import time

# Test serial execution (1000 iterations might be too many for testing)
general_config['psa']['iterations'] = 10  # Use fewer iterations for testing
general_config['psa']['n_jobs'] = 1

start = time.time()
run_probabilistic_sensitivity_analysis(general_config, general_config['psa'])
serial_time = time.time() - start
print(f"Serial (1 core): {serial_time:.1f} seconds")

# Test parallel execution
general_config['psa']['n_jobs'] = None  # Use all cores

start = time.time()
run_probabilistic_sensitivity_analysis(general_config, general_config['psa'])
parallel_time = time.time() - start
print(f"Parallel (all cores): {parallel_time:.1f} seconds")
print(f"Speedup: {serial_time/parallel_time:.1f}x")

# Restore full iteration count
general_config['psa']['iterations'] = 1000
```

## 6. Troubleshooting

### Issue: "Out of Memory" errors with parallel processing
**Solution:** Reduce `n_jobs`:
```python
general_config['psa']['n_jobs'] = 4  # or 2, depending on RAM
```

### Issue: No speedup observed
**Possible causes:**
1. CPU-bound system (already at 100% on all cores)
2. I/O bottleneck (slow disk)
3. Very fast single runs (parallelization overhead dominates)

### Issue: "Cannot pickle" error
**Solution:** Ensure all objects in `general_config` are picklable. Lambda functions and some complex objects may cause issues.

### Issue: Results differ slightly between serial and parallel runs
**Note:** This is expected due to floating-point arithmetic order. Results should be statistically equivalent. Use the same `seed` for reproducibility.

## 7. Summary of Changes Made

1. **Added imports** (IBM_PD_AD.py, lines 6-26):
   - `multiprocessing.Pool` and `cpu_count`
   - `pickle` and `gzip` for compression
   - Optional `tqdm` for progress bars

2. **Added `_run_single_psa_iteration()` helper** (lines 2782-2801):
   - Wrapper function for parallel execution

3. **Modified `run_probabilistic_sensitivity_analysis()`** (lines 2804-2886):
   - Added `n_jobs` parameter
   - Parallel execution using multiprocessing.Pool
   - Progress tracking with tqdm
   - Pre-generates seeds for reproducibility

4. **Added compression utilities** (lines 2889-2951):
   - `save_results_compressed()` / `load_results_compressed()`
   - `save_dataframe_compressed()`

5. **Updated configuration** (line 322):
   - Added `n_jobs` parameter to PSA config

## 8. Best Practices

1. **Always save outputs compressed** to save disk space
2. **Use progress tracking** (`pip install tqdm`) for long runs
3. **Start with a small test run** (e.g., 10 iterations) to verify everything works
4. **Monitor memory usage** when using all cores
5. **Use Parquet format** for DataFrames (better than CSV.gz)
6. **Keep results organized**:
   ```
   results/
   ├── baseline_model.pkl.gz
   ├── psa_results.pkl.gz
   └── psa_draws.parquet
   ```

## Questions?

For further optimization opportunities (e.g., vectorizing population loops), please consult a performance profiling analysis of specific bottlenecks in your runs.
