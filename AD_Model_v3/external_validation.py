"""
External Validation Script for IBM Dementia Model
Validates model predictions against observed 2024 prevalence data

RECOMMENDED WORKFLOW:
1. Run generate_validation_data.py ONCE to create 2024 results (uses full population)
2. Run this script to load those results and perform validation

This two-step approach allows you to:
- Run the full population model once and save results
- Quickly re-run validation with different parameters/plots
- Avoid re-running the simulation every time

The script uses pre-aggregated age-banded prevalence data from the model's
'incidence_by_year_sex_df' output, which tracks prevalence by age band, sex,
and year. This allows validation with R² and slope coefficients at the
aggregated level.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IBM_PD_AD_v3 import (
    run_model,
    general_config,
    save_results_compressed,
    load_results_compressed,
)
import copy
import os


# Observed prevalence data for 2024 (from surveillance data)
OBSERVED_DATA_2024 = [
    [35, 49, "F", 0.0001],
    [50, 64, "F", 0.0012],
    [65, 79, "F", 0.0178],
    [80, None, "F", 0.1244],
    [35, 49, "M", 0.0001],
    [50, 64, "M", 0.0013],
    [65, 79, "M", 0.0168],
    [80, None, "M", 0.0910],
]

# ONS England 2024 population (both sexes) by 5-year band (counts, not prevalence)
# Source: user-provided table
OBSERVED_POP_2024 = pd.DataFrame({
    "age_lower": [65, 70, 75, 80, 85, 90],
    "age_upper": [69, 74, 79, 84, 89, None],
    "population_obs": [3_137_193, 2_735_296, 2_588_675, 1_640_827, 1_007_996, 563_609],
})

# Observed recorded dementia diagnoses (counts) by 5-year age band and sex
OBSERVED_DEMENTIA_COUNTS = {
    2024: pd.DataFrame({
        "age_lower": [65, 70, 75, 80, 85, 90] * 2,
        "age_upper": [69, 74, 79, 84, 89, None] * 2,
        "sex": ["M"] * 6 + ["F"] * 6,
        "cases_obs": [
            9532, 17908, 37002, 46968, 43787, 26883,    # male
            8898, 19412, 46934, 70271, 80208, 75087     # female
        ],
    }),
    2025: pd.DataFrame({
        "age_lower": [65, 70, 75, 80, 85, 90] * 2,
        "age_upper": [69, 74, 79, 84, 89, None] * 2,
        "sex": ["M"] * 6 + ["F"] * 6,
        "cases_obs": [
            9715, 18364, 38543, 48785, 44962, 28111,    # male
            9154, 19783, 48774, 72931, 81670, 76593     # female
        ],
    }),
}


def run_model_to_year(config, target_year=2024, seed=42):
    """
    Run model to target_year and return results

    NOTE: This function runs with the FULL population from config.
    For a less memory-intensive approach, first run generate_validation_data.py
    to create cached results, then load them with the results_file parameter.

    Parameters:
    -----------
    config : dict
        Model configuration (uses population size as-is from config)
    seed : int
        Random seed for reproducibility
    """
    print(f"Running model to {target_year}...")
    print(f"  Population: {config.get('population', 0):,}")

    # Create a copy of config to modify
    validation_config = copy.deepcopy(config)

    # Ensure model runs to 2024 by calculating number of timesteps needed
    base_year = int(validation_config.get('base_year', 2023))
    timesteps_needed = target_year - base_year
    validation_config['number_of_timesteps'] = timesteps_needed

    print(f"  Running {timesteps_needed} timestep(s): {base_year} -> {target_year}")

    # Run the model (same way IBM_PD_AD runs it)
    results = run_model(validation_config, seed=seed, return_agents=False)

    print(f"Model run complete. End year: {base_year + timesteps_needed}")
    return results


def extract_population_by_age(results, year=2024):
    """
    Extract population alive by 5-year age band (65+) combining sexes.
    Returns DataFrame with columns: age_lower, age_upper, population_pred
    """
    df = results['incidence_by_year_sex_df']
    year_df = df[(df['calendar_year'] == year) & (df['sex'].isin(['male', 'female']))].copy()
    if year_df.empty:
        print(f"No incidence_by_year_sex_df rows for {year}")
        return pd.DataFrame()

    # group by age_lower (each 5-year band) summing both sexes
    grouped = (
        year_df.groupby(['age_lower', 'age_upper'], as_index=False)['population_alive_in_band']
        .sum()
        .rename(columns={'population_alive_in_band': 'population_pred'})
    )
    return grouped


def extract_prevalence_by_age_sex(results, year=2024):
    """
    Extract prevalence from model results by age group and sex using aggregated data

    Returns DataFrame with columns: age_lower, age_upper, sex, prevalence

    NOTE: This function uses pre-aggregated data from incidence_by_year_sex_df,
    avoiding the need to store individual agent data for 33+ million people.

    The data is aggregated to match the observed age bands:
    - 35-49, 50-64, 65-79, 80+
    """
    print(f"\nExtracting prevalence for year {year}...")

    # Get the pre-aggregated incidence dataframe
    incidence_df = results['incidence_by_year_sex_df']

    # Filter to target year and exclude 'all' sex category
    year_data = incidence_df[
        (incidence_df['calendar_year'] == year) &
        (incidence_df['sex'].isin(['male', 'female']))
    ].copy()

    if year_data.empty:
        print(f"  Warning: No data found for year {year}")
        return pd.DataFrame()

    # Map sex from 'male'/'female' to 'M'/'F' for consistency
    year_data['sex_code'] = year_data['sex'].map({'male': 'M', 'female': 'F'})

    # Define target age bands matching observed data
    target_bands = [
        (35, 49),
        (50, 64),
        (65, 79),
        (80, None)  # 80+
    ]

    prevalence_data = []

    for sex_code in ['F', 'M']:
        sex_data = year_data[year_data['sex_code'] == sex_code]

        for age_lower, age_upper in target_bands:
            # Filter model bands whose age_lower falls within target range
            # Note: age_upper in target bands is exclusive (e.g., 35-49 means 35 <= age < 50)
            if age_upper is None:  # 80+ case
                mask = (sex_data['age_lower'] >= age_lower)
            else:
                # Include all bands that start within [age_lower, age_upper)
                # For 35-49: include bands starting at 35,40,45 (NOT 50)
                mask = (sex_data['age_lower'] >= age_lower) & (sex_data['age_lower'] < age_upper)

            band_data = sex_data[mask]

            if band_data.empty:
                print(f"  Warning: No data for {age_lower}-{age_upper if age_upper else '+'} {sex_code}")
                continue

            # Aggregate across the sub-bands
            total_population = band_data['population_alive_in_band'].sum()
            total_dementia = band_data['prevalent_dementia_cases_in_band'].sum()

            if total_population > 0:
                prevalence = total_dementia / total_population
            else:
                prevalence = 0.0

            age_band_str = f"{age_lower}-{age_upper}" if age_upper else f"{age_lower}+"

            print(f"  {age_band_str:8s} {sex_code}: {total_dementia:6d}/{total_population:6d} = {prevalence:.6f}")

            prevalence_data.append({
                'age_lower': age_lower,
                'age_upper': age_upper,
                'sex': sex_code,
                'prevalence': prevalence,
                'age_band': age_band_str,
                'n_total': int(total_population),
                'n_with_dementia': int(total_dementia)
            })

    return pd.DataFrame(prevalence_data)


def extract_prevalence_five_by_sex(results, year=2024):
    """
    Extract prevalence by 5-year bands (65+) and sex.
    Returns columns: age_lower, age_upper, sex, prevalence, n_total, n_with_dementia, age_band
    """
    df = results['incidence_by_year_sex_df']
    year_df = df[(df['calendar_year'] == year) & (df['sex'].isin(['male', 'female']))].copy()
    if year_df.empty:
        print(f"No data for year {year}")
        return pd.DataFrame()

    records = []
    for sex_code, sex_name in [('M', 'male'), ('F', 'female')]:
        sex_df = year_df[year_df['sex'] == sex_name]
        grouped = (
            sex_df.groupby(['age_lower', 'age_upper'], as_index=False)[
                ['population_alive_in_band', 'prevalent_dementia_cases_in_band']
            ].sum()
        )
        grouped['sex'] = sex_code
        grouped['prevalence'] = grouped['prevalent_dementia_cases_in_band'] / grouped['population_alive_in_band']
        grouped['age_band'] = grouped.apply(
            lambda r: f"{int(r['age_lower'])}-{int(r['age_upper'])}" if pd.notna(r['age_upper']) else f"{int(r['age_lower'])}+", axis=1
        )
        grouped = grouped.rename(columns={
            'population_alive_in_band': 'n_total',
            'prevalent_dementia_cases_in_band': 'n_with_dementia'
        })
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def ols_fit(x, y):
    """Perform OLS regression and return fit statistics"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Add intercept
    X = np.column_stack((np.ones_like(x), x))

    # Calculate beta coefficients
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    # Calculate predictions and residuals
    y_hat = X @ beta
    resid = y - y_hat

    # Calculate R²
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'alpha': beta[0],  # intercept
        'beta': beta[1],   # slope
        'r2': r2
    }


def create_calibration_plot(predicted_df, observed_df, sex, save_dir):
    """Create calibration plot comparing predicted vs observed prevalence"""

    # Merge predicted and observed data
    merged = predicted_df[predicted_df['sex'] == sex].copy()
    obs_subset = observed_df[observed_df['sex'] == sex].copy()

    merged = merged.merge(
        obs_subset[['age_lower', 'age_upper', 'observed']],
        on=['age_lower', 'age_upper'],
        how='inner'  # Use inner join to only keep matching rows
    )

    # Remove any rows with missing values
    merged = merged.dropna(subset=['prevalence', 'observed'])

    if len(merged) < 2:
        print(f"  Warning: Insufficient data points ({len(merged)}) for {sex} calibration")
        return {'alpha': 0.0, 'beta': 0.0, 'r2': 0.0}

    # Fit OLS
    fit_stats = ols_fit(merged['prevalence'], merged['observed'])
    alpha, beta, r2 = fit_stats['alpha'], fit_stats['beta'], fit_stats['r2']

    # Print statistics
    sex_label = "Female" if sex == "F" else "Male"
    print(f"\n{sex_label} Calibration Statistics:")
    print(f"  Intercept (alpha): {alpha:.6f}")
    print(f"  Slope (beta):      {beta:.6f}")
    print(f"  R²:                {r2:.4f}")

    # Create plot
    plt.figure(figsize=(7, 7))
    plt.scatter(merged['prevalence'], merged['observed'], s=100, alpha=0.7, edgecolors='black')

    # Plot 1:1 line
    max_val = max(merged['prevalence'].max(), merged['observed'].max()) * 1.1
    x_line = np.linspace(0, max_val, 100)
    plt.plot(x_line, x_line, 'k--', label='Perfect calibration (1:1)', linewidth=2)

    # Plot fitted line
    plt.plot(x_line, alpha + beta * x_line, 'r-',
             label=f'Fitted (slope={beta:.2f}, R²={r2:.3f})', linewidth=2)

    # Annotate points with age bands
    for _, row in merged.iterrows():
        plt.annotate(row['age_band'],
                    (row['prevalence'], row['observed']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.xlabel('Predicted Prevalence (2024)', fontsize=12)
    plt.ylabel('Observed Prevalence (2024)', fontsize=12)
    plt.title(f'External Validation - {sex_label}\nDementia Prevalence 2024', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save plot
    file_path = save_dir / f"external_validation_{sex.lower()}.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {file_path}")

    return fit_stats


def run_population_validation(seed=42, save_dir="validation_outputs_pop"):
    """
    Validate model total population counts (65+) by 5-year age band against ONS 2024 estimates.
    Runs IBM_PD_AD_v3 one timestep (2023->2024) on full population,
    extracts population_alive_in_band, compares to ONS.
    Returns dict with alpha, beta, r2 and writes a CSV comparison table.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run full population
    results = run_model_to_year(general_config, target_year=2024, seed=seed)

    predicted = extract_population_by_age(results, year=2024)
    predicted['age_band'] = predicted.apply(
        lambda r: f"{int(r['age_lower'])}-{int(r['age_upper'])}" if pd.notna(r['age_upper']) else f"{int(r['age_lower'])}+", axis=1
    )

    observed = OBSERVED_POP_2024.copy()
    observed['age_band'] = observed.apply(
        lambda r: f"{int(r['age_lower'])}-{int(r['age_upper'])}" if pd.notna(r['age_upper']) else f"{int(r['age_lower'])}+", axis=1
    )

    merged = predicted.merge(observed, on=['age_lower', 'age_upper', 'age_band'], how='inner')
    merged['abs_error'] = merged['population_pred'] - merged['population_obs']
    merged['relative_error_pct'] = merged['abs_error'] / merged['population_obs'] * 100

    fit_stats = ols_fit(merged['population_pred'], merged['population_obs'])

    # Save table
    table_path = save_dir / "population_validation_table.csv"
    merged[['age_band', 'population_obs', 'population_pred', 'abs_error', 'relative_error_pct']].to_csv(table_path, index=False)
    print(f"Population comparison saved to {table_path}")

    print("\nPopulation calibration statistics (counts, 2024):")
    print(f"  Intercept (alpha): {fit_stats['alpha']:.2f}")
    print(f"  Slope (beta):      {fit_stats['beta']:.6f}")
    print(f"  R²:                {fit_stats['r2']:.4f}")

    # Optional quick plot
    plt.figure(figsize=(7,7))
    plt.scatter(merged['population_pred'], merged['population_obs'], s=120, edgecolors='black', alpha=0.7)
    max_val = max(merged['population_pred'].max(), merged['population_obs'].max()) * 1.1
    x_line = np.linspace(0, max_val, 100)
    plt.plot(x_line, x_line, 'k--', label='Perfect agreement')
    plt.plot(x_line, fit_stats['alpha'] + fit_stats['beta'] * x_line, 'r-', label=f"Fit (β={fit_stats['beta']:.3f}, R²={fit_stats['r2']:.3f})")
    for _, row in merged.iterrows():
        plt.annotate(row['age_band'], (row['population_pred'], row['population_obs']), xytext=(5,5), textcoords='offset points')
    plt.xlabel('Model population (2024)')
    plt.ylabel('ONS population (2024)')
    plt.title('Population Validation (65+, England 2024)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot_path = save_dir / "population_validation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Population plot saved to {plot_path}")

    return {'alpha': fit_stats['alpha'], 'beta': fit_stats['beta'], 'r2': fit_stats['r2'], 'table': table_path, 'plot': plot_path}


def run_prevalence_validation(year: int, seed=42, save_dir="validation_outputs_prev"):
    """
    Validate dementia prevalence by 5-year age band and sex using observed recorded diagnoses as proxy.
    Observed prevalence = observed_cases / model_population_in_band (by sex).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if year not in OBSERVED_DEMENTIA_COUNTS:
        raise ValueError(f"No observed counts available for year {year}")

    results = run_model_to_year(general_config, target_year=year, seed=seed)

    predicted = extract_prevalence_five_by_sex(results, year=year)
    if predicted.empty:
        raise RuntimeError("No predicted prevalence extracted.")

    observed = OBSERVED_DEMENTIA_COUNTS[year].copy()
    observed['age_band'] = observed.apply(
        lambda r: f"{int(r['age_lower'])}-{int(r['age_upper'])}" if pd.notna(r['age_upper']) else f"{int(r['age_lower'])}+",
        axis=1
    )

    merged = predicted.merge(
        observed,
        on=['age_lower', 'age_upper', 'sex', 'age_band'],
        how='inner'
    )

    # Observed prevalence using model denominator (proxy)
    merged['observed_prevalence'] = merged['cases_obs'] / merged['n_total']
    merged['predicted_prevalence'] = merged['prevalence']
    merged['abs_error'] = merged['predicted_prevalence'] - merged['observed_prevalence']
    merged['relative_error_pct'] = merged['abs_error'] / merged['observed_prevalence'] * 100

    fit_stats = ols_fit(merged['predicted_prevalence'], merged['observed_prevalence'])

    # Save table
    table_path = save_dir / f"prevalence_validation_{year}.csv"
    merged[['age_band', 'sex', 'cases_obs', 'n_total', 'observed_prevalence',
            'predicted_prevalence', 'abs_error', 'relative_error_pct']].to_csv(table_path, index=False)
    print(f"Prevalence comparison saved to {table_path}")

    # Plot
    plt.figure(figsize=(7,7))
    plt.scatter(merged['predicted_prevalence'], merged['observed_prevalence'],
                s=120, edgecolors='black', alpha=0.7)
    max_val = max(merged['predicted_prevalence'].max(), merged['observed_prevalence'].max()) * 1.1
    x_line = np.linspace(0, max_val, 100)
    plt.plot(x_line, x_line, 'k--', label='Perfect agreement')
    plt.plot(x_line, fit_stats['alpha'] + fit_stats['beta'] * x_line, 'r-',
             label=f"Fit (β={fit_stats['beta']:.3f}, R²={fit_stats['r2']:.3f})")
    for _, row in merged.iterrows():
        plt.annotate(f"{row['age_band']} {row['sex']}",
                     (row['predicted_prevalence'], row['observed_prevalence']),
                     xytext=(5,5), textcoords='offset points', fontsize=8)
    plt.xlabel('Model prevalence')
    plt.ylabel('Observed prevalence (proxy)')
    plt.title(f'Dementia Prevalence Validation {year}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot_path = save_dir / f"prevalence_validation_{year}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prevalence plot saved to {plot_path}")

    print(f"\nPrevalence calibration {year}: alpha={fit_stats['alpha']:.6f}, beta={fit_stats['beta']:.6f}, R²={fit_stats['r2']:.4f}")

    return {'alpha': fit_stats['alpha'], 'beta': fit_stats['beta'], 'r2': fit_stats['r2'],
            'table': table_path, 'plot': plot_path, 'merged': merged}

def create_comparison_table(predicted_df, observed_df, save_dir):
    """Create a comparison table showing predicted vs observed"""

    # Merge data
    comparison = predicted_df.merge(
        observed_df[['age_lower', 'age_upper', 'sex', 'observed']],
        on=['age_lower', 'age_upper', 'sex'],
        how='left'
    )

    # Calculate metrics
    comparison['absolute_error'] = comparison['prevalence'] - comparison['observed']
    comparison['relative_error_pct'] = (comparison['absolute_error'] / comparison['observed']) * 100

    # Format for display
    comparison_display = comparison[[
        'age_band', 'sex', 'observed', 'prevalence',
        'absolute_error', 'relative_error_pct', 'n_total', 'n_with_dementia'
    ]].copy()

    comparison_display.columns = [
        'Age Band', 'Sex', 'Observed', 'Predicted',
        'Abs Error', 'Rel Error (%)', 'N Total', 'N Dementia'
    ]

    # Save to CSV
    csv_path = save_dir / "validation_comparison_table.csv"
    comparison_display.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nComparison table saved: {csv_path}")

    # Print to console
    print("\n" + "="*90)
    print("VALIDATION COMPARISON TABLE")
    print("="*90)
    print(comparison_display.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print("="*90)

    return comparison_display


def run_external_validation(seed=42, save_dir=None, results_file=None, save_results=False):
    """
    Main function to run external validation

    RECOMMENDED: Use results_file parameter to load pre-computed results from
    generate_validation_data.py rather than running the model fresh.

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility (only used if running model fresh)
    save_dir : Path or str
        Directory to save outputs (default: plots/)
    results_file : Path or str, optional
        Path to pre-computed model results (.pkl.gz file)
        If provided, will load results instead of running model
        If None, will run the model fresh with FULL population from general_config
    save_results : bool
        If True and running fresh, save model results to 'validation_results_2024.pkl.gz'
        for future reuse (default: False - use generate_validation_data.py instead)
    """
    print("="*80)
    print("EXTERNAL VALIDATION - IBM DEMENTIA MODEL")
    print("="*80)

    # Set up save directory
    if save_dir is None:
        save_dir = Path("plots")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get model results (either load or run)
    if results_file is not None and os.path.exists(results_file):
        print(f"\nLoading pre-computed results from: {results_file}")
        results = load_results_compressed(results_file)
        print("Results loaded successfully!")
    else:
        if results_file is not None:
            print(f"\nWarning: Results file not found: {results_file}")
            print("Running model fresh instead...\n")

        # Run model to 2024 with FULL population
        results = run_model_to_2024(general_config, seed=seed)

        # Save results for future reuse if requested
        if save_results:
            results_path = Path("validation_results_2024.pkl.gz")
            print(f"\nSaving results to {results_path} for future reuse...")
            save_results_compressed(results, results_path)
            print(f"  To reuse: run_external_validation(results_file='{results_path}')")

    # 2. Extract predicted prevalence
    predicted_df = extract_prevalence_by_age_sex(results, year=2024)

    # 3. Load observed data
    observed_df = pd.DataFrame(
        OBSERVED_DATA_2024,
        columns=['age_lower', 'age_upper', 'sex', 'observed']
    )
    observed_df['age_band'] = observed_df.apply(
        lambda r: f"{int(r.age_lower)}+" if pd.isna(r.age_upper)
        else f"{int(r.age_lower)}-{int(r.age_upper)}",
        axis=1
    )

    # 4. Create comparison table
    comparison = create_comparison_table(predicted_df, observed_df, save_dir)

    # 5. Create calibration plots for each sex
    female_stats = create_calibration_plot(predicted_df, observed_df, 'F', save_dir)
    male_stats = create_calibration_plot(predicted_df, observed_df, 'M', save_dir)

    # 6. Combined plot
    print("\nCreating combined calibration plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, sex, sex_label in [(ax1, 'F', 'Female'), (ax2, 'M', 'Male')]:
        merged = predicted_df[predicted_df['sex'] == sex].merge(
            observed_df[observed_df['sex'] == sex][['age_lower', 'age_upper', 'observed']],
            on=['age_lower', 'age_upper'],
            how='inner'
        ).dropna(subset=['prevalence', 'observed'])

        stats = female_stats if sex == 'F' else male_stats

        ax.scatter(merged['prevalence'], merged['observed'], s=100, alpha=0.7, edgecolors='black')
        max_val = max(merged['prevalence'].max(), merged['observed'].max()) * 1.1
        x_line = np.linspace(0, max_val, 100)
        ax.plot(x_line, x_line, 'k--', label='1:1 line', linewidth=2)
        ax.plot(x_line, stats['alpha'] + stats['beta'] * x_line, 'r-',
                label=f"Fitted (R²={stats['r2']:.3f})", linewidth=2)

        for _, row in merged.iterrows():
            ax.annotate(row['age_band'], (row['prevalence'], row['observed']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Predicted Prevalence', fontsize=11)
        ax.set_ylabel('Observed Prevalence', fontsize=11)
        ax.set_title(f'{sex_label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('External Validation - Dementia Prevalence 2024', fontsize=14, fontweight='bold')
    plt.tight_layout()
    combined_path = save_dir / "external_validation_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved: {combined_path}")

    # 7. Summary statistics
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Female: R² = {female_stats['r2']:.4f}, Slope = {female_stats['beta']:.4f}")
    print(f"Male:   R² = {male_stats['r2']:.4f}, Slope = {male_stats['beta']:.4f}")
    print("="*80)

    # Return results
    return {
        'predicted': predicted_df,
        'observed': observed_df,
        'comparison': comparison,
        'female_stats': female_stats,
        'male_stats': male_stats,
        'model_results': results
    }


if __name__ == "__main__":
    # Population validation (counts)
    pop_stats = run_population_validation(seed=42, save_dir="plots")
    print("\n[SUCCESS] Population validation complete!")
    print(f"  alpha={pop_stats['alpha']:.2f}, beta={pop_stats['beta']:.6f}, R²={pop_stats['r2']:.4f}")
    print(f"  Table: {pop_stats['table']}")
    print(f"  Plot:  {pop_stats['plot']}")

    # Prevalence validation for 2024 and 2025
    prev24 = run_prevalence_validation(year=2024, seed=42, save_dir="plots")
    prev25 = run_prevalence_validation(year=2025, seed=42, save_dir="plots")
    print("\n[SUCCESS] Prevalence validation complete!")
    print(f"  2024: alpha={prev24['alpha']:.6f}, beta={prev24['beta']:.6f}, R²={prev24['r2']:.4f}")
    print(f"        Table: {prev24['table']} | Plot: {prev24['plot']}")
    print(f"  2025: alpha={prev25['alpha']:.6f}, beta={prev25['beta']:.6f}, R²={prev25['r2']:.4f}")
    print(f"        Table: {prev25['table']} | Plot: {prev25['plot']}")
