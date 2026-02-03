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
from IBM_PD_AD import run_model, general_config, save_results_compressed, load_results_compressed
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


def run_model_to_2024(config, seed=42):
    """
    Run model to 2024 and return results

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
    print(f"Running model to 2024...")
    print(f"  Population: {config.get('population', 0):,}")

    # Create a copy of config to modify
    validation_config = copy.deepcopy(config)

    # Ensure model runs to 2024 by calculating number of timesteps needed
    base_year = int(validation_config.get('base_year', 2023))
    target_year = 2024
    timesteps_needed = target_year - base_year
    validation_config['number_of_timesteps'] = timesteps_needed

    print(f"  Running {timesteps_needed} timestep(s): {base_year} -> {target_year}")

    # Run the model (same way IBM_PD_AD runs it)
    results = run_model(validation_config, seed=seed, return_agents=False)

    print(f"Model run complete. End year: {base_year + timesteps_needed}")
    return results


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
    # RECOMMENDED WORKFLOW:
    # Step 1: Run generate_validation_data.py ONCE to create 2024 results (full population)
    # Step 2: Run this script to load results and perform validation

    # MODE 1: Load pre-computed results (RECOMMENDED - fast, uses full population)
    # First run: python generate_validation_data.py
    # Then uncomment the line below:
    validation_results = run_external_validation(
        seed=42,
        results_file="validation_results_2024.pkl.gz"
    )

    # MODE 2: Run model fresh (will use FULL population from general_config)
    # Uncomment the line below to run without pre-computed data:
    # validation_results = run_external_validation(seed=42, save_results=True)

    print("\n[SUCCESS] External validation complete!")
    print(f"  Check the 'plots/' directory for output files")
