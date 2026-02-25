# Full Analysis Pipeline

## Overview

This directory contains a comprehensive master script (`run_full_analysis.py`) that orchestrates all analysis steps for the periodontal disease and dementia model. The script is designed for weekend batch processing and runs all analyses sequentially.

## Quick Start

### Before Your Weekend Run: Test First!

**Highly recommended:** Test the pipeline first with small populations (~10-20 minutes):

```bash
python test_pipeline.py
```

This will:
- Verify all imports work
- Check configuration is valid
- Run a quick model test
- Execute the pipeline with 1% population to catch any errors

### Full Weekend Run

Once the test passes, run the complete analysis pipeline:

```bash
python run_full_analysis.py
```

This will execute all enabled steps (configured at the top of the script) and save results to `full_analysis_results/`.

## Pipeline Steps

### Step 1: Main Model Runs (25%, 50%, 75% PD)
**Script:** Embedded in `run_full_analysis.py`
**Purpose:** Run the main IBM model at three different periodontal disease prevalence levels
**Output:**
- `full_analysis_results/main_model_runs/results_pd_XX_percent.pkl.gz` (compressed results)
- `results/Baseline_Model_PD_25.xlsx` (25% PD scenario - all summary data)
- `results/Baseline_Model_PD_50.xlsx` (50% PD scenario - all summary data)
- `results/Baseline_Model_PD_75.xlsx` (75% PD scenario - all summary data)
**Runtime:** ~30-60 minutes per prevalence level with full population
**Note:** Each scenario exports to a unique Excel file to avoid overwriting. Automatic plot generation is disabled during batch processing.

### Step 2: Probabilistic Sensitivity Analysis (PSA)
**Script:** `run_psa_direct_v3.py`
**Purpose:** 500 iterations of PSA at 25%, 50%, 75% PD prevalence
**Output:** Excel files and plots in current directory
**Runtime:** ~2-4 hours with 1% population
**Configuration:**
- PSA_ITERATIONS = 500
- SCALE_FACTOR = 0.01 (1% population)

### Step 3: One-Way Sensitivity Analysis (Tornado)
**Script:** `run_pd_tornado.py`
**Purpose:** One-way sensitivity analysis for PD hazard ratio uncertainty
**Output:**
- `pd_sensitivity_analysis.xlsx` - Detailed results with Baseline, Low, and High scenarios
**Runtime:** ~1-2 hours with 10 replicates
**Note:** Automatic tornado plot generation is disabled (plots don't render correctly). All data is exported to Excel for manual plotting.

### Step 4: External Validation
**Script:** `external_validation.py`
**Purpose:** Validate model against NHS England data
**Validates:**
- Population counts by age band (ONS 2024)
- Dementia prevalence by age/sex (NHS 2024, 2025)
**Output:** `full_analysis_results/validation/`
- Population validation plots and tables
- Prevalence validation plots and tables for 2024 and 2025
**Runtime:** ~5-10 minutes per validation year

### Step 5: Age Distribution Validation
**Script:** `validate_age_distribution.py`
**Purpose:** Validate age distribution dynamics against ONS projections (2025, 2030, 2035, 2040)
**Output:** `full_analysis_results/age_validation/`
**Runtime:** ~1-2 hours with full population

### Step 6: Counterfactual Analysis (Non-Preventable Risk)
**Script:** `run_non_preventable_risk_analysis.py`
**Purpose:** Calculate incidence/prevalence not attributable to preventable risk factors
**Output:** `results/non_preventable_risk_analysis.xlsx`
**Runtime:** ~30-60 minutes

### Step 7: Generate Manuscript Figures
**Script:** `generate_figures_2_3_4_from_model.py`
**Purpose:** Create publication-ready figures and export source data for journal reproducibility
**Generates:**
- `AD_Model_v3/images_regenerated_model/figure_2.png` - Risk factor landscape (2040, 50% PD)
- `AD_Model_v3/images_regenerated_model/figure_3.png` - Annual societal costs by scenario (2024-2040)
- `AD_Model_v3/images_regenerated_model/figure_4.png` - Cumulative QALY differences vs baseline (2024-2040)
- `AD_Model_v3/images_regenerated_model/figure_2_3_4_data.xlsx` - **Figure source data for journal reproducibility**
- `results/Figure_Generation_PD_25.xlsx` - Complete results for 25% PD scenario (used for figures)
- `results/Figure_Generation_PD_50.xlsx` - Complete results for 50% PD scenario (used for figures)
- `results/Figure_Generation_PD_75.xlsx` - Complete results for 75% PD scenario (used for figures)
**Runtime:** ~30-90 minutes (runs 3 full model scenarios)
**Note:** All Excel exports use unique filenames to avoid overwriting. The `figure_2_3_4_data.xlsx` contains extracted data for the figures, while the `Figure_Generation_PD_XX.xlsx` files contain complete model results.

## Optional Additional Analyses

### Hazard Ratio Meta-Analysis Utility
**Script:** `combine_hazard_ratios.py`
**Purpose:** Utility for combining hazard ratios from multiple studies
**Note:** This is a utility script, not part of the analysis pipeline.

## Configuration

Edit the `CONFIG` dictionary at the top of `run_full_analysis.py` to customize:

```python
CONFIG = {
    'seed': 42,
    'output_dir': Path('full_analysis_results'),

    # Enable/disable steps
    'run_main_model': True,
    'run_psa': True,
    'run_tornado': True,
    'run_external_validation': True,
    'run_age_validation': True,
    'run_counterfactual': True,
    'run_generate_figures': True,

    # Step-specific settings
    'main_model_prevalences': [0.25, 0.50, 0.75],
    'psa_iterations': 500,
    'tornado_n_replicates': 10,
    'validation_years': [2024, 2025],
    # ... etc
}
```

## Data Outputs for Journal Reproducibility

The pipeline exports all data needed for journal submissions and reproducibility:

### Primary Data Exports:

1. **`results/Baseline_Model_PD_25.xlsx`, `Baseline_Model_PD_50.xlsx`, `Baseline_Model_PD_75.xlsx`** (from Step 1)
   - Complete summary data (all timesteps, all metrics)
   - Yearly flows (QALYs, costs by year)
   - Risk factor prevalence over time
   - Severity distribution (mild/moderate/severe cases)

2. **`AD_Model_v3/images_regenerated_model/figure_2_3_4_data.xlsx`** (from Step 7)
   - `figure_2_risk_landscape` - Risk factor prevalence data for 2040
   - `figure_3_annual_costs` - Annual societal costs by scenario (2024-2040)
   - `figure_4_qaly_differences` - Cumulative QALY differences vs baseline (2024-2040)
   - This file contains ALL source data needed to reproduce manuscript figures

3. **PSA Results** (from Step 2)
   - Multiple Excel files with probabilistic sensitivity analysis results
   - Includes mean, median, and 95% confidence intervals for all outcomes

4. **Tornado Analysis** (from Step 3)
   - `pd_sensitivity_analysis.xlsx` - One-way sensitivity results
   - Detailed swing analysis for QALYs, costs, and incidence

5. **Validation Data** (from Step 4)
   - `full_analysis_results/validation/*.csv` - Population and prevalence validation tables
   - Includes observed vs predicted with R² and calibration statistics

**Note:** All Excel files include raw data that journals can use to verify and reproduce your analyses. The main IBM script no longer generates plots automatically to avoid clutter during batch processing.

## Output Structure

```
full_analysis_results/
├── full_analysis_log.txt              # Complete execution log
├── main_model_runs/                   # Step 1 outputs
│   ├── results_pd_25_percent.pkl.gz
│   ├── results_pd_50_percent.pkl.gz
│   └── results_pd_75_percent.pkl.gz
├── validation/                        # Step 4 outputs
│   ├── population_validation.png
│   ├── population_validation_table.csv
│   ├── prevalence_validation_2024.png
│   ├── prevalence_validation_2024.csv
│   ├── prevalence_validation_2025.png
│   └── prevalence_validation_2025.csv
└── age_validation/                    # Step 5 outputs (if enabled)
    ├── age_distribution_validation_2025.csv
    ├── age_distribution_validation_2030.csv
    ├── age_distribution_validation_2035.csv
    ├── age_distribution_validation_2040.csv
    └── age_distribution_validation_all_years.png

plots/                                 # Step 3 outputs
├── pd_tornado_main.png
├── pd_tornado_qalys.png
├── pd_tornado_incidence.png
├── pd_tornado_costs.png
├── figure_2.png                       # Step 6 outputs
├── figure_3.png
└── figure_4.png

[Current directory also contains PSA Excel outputs and other files from Steps 2-6]
```

## Estimated Total Runtime

With default settings:

| Component | Runtime | Notes |
|-----------|---------|-------|
| Step 1: Main Model (3 runs) | 1-3 hours | Full population |
| Step 2: PSA | 2-4 hours | 1% population, 500 iterations |
| Step 3: Tornado | 1-2 hours | 1% population, 10 replicates |
| Step 4: Validation | 10-20 minutes | Quick validation runs |
| Step 5: Age Validation | 1-2 hours | Full population, 4 projection years |
| Step 6: Counterfactual | 30-60 minutes | Full population |
| Step 7: Figures | 1-2 hours | Full population, 3 scenarios |
| **TOTAL** | **~7-14 hours** | Suitable for overnight/weekend run |

## Error Handling

The pipeline is designed to be robust:
- If a step fails, it logs the error and continues to the next step
- All output is logged to both console and `full_analysis_log.txt`
- Intermediate results are saved, so failed steps can be re-run independently
- Final summary shows which steps succeeded/failed

## Tips for Weekend Runs

1. **Test first:** Run with small populations to verify everything works
   ```python
   'main_model_population_fraction': 0.01,  # Test with 1%
   'figures_population_fraction': 0.01,     # Test figures
   ```

2. **Monitor progress:** Check the log file periodically
   ```bash
   tail -f full_analysis_results/full_analysis_log.txt
   ```

3. **Disable long steps for testing:** Set `'run_age_validation': False` and `'run_counterfactual': False` in CONFIG for faster test runs

4. **Use full population for publication:** Set fractions to 1.0 for final results

5. **Clean up duplicate PSA directories:** If you have old PSA results outside AD_Model_v3/, run:
   ```bash
   python cleanup_duplicate_psa.py
   ```

## Requirements

All required packages should already be installed. The pipeline uses:
- IBM_PD_AD_v3.py (main model)
- numpy, pandas, matplotlib
- joblib (for parallel processing)
- openpyxl (for Excel output)

## Questions?

- **Model configuration:** Edit `IBM_PD_AD_v3.py` → `general_config`
- **Pipeline configuration:** Edit `run_full_analysis.py` → `CONFIG`
- **Individual scripts:** Each can be run standalone for testing
