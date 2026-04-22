"""
Validate Model Age Distribution Against ONS Projections

This script validates that the model's age distribution dynamics (aging + mortality)
correctly reproduce ONS population projections for 2025, 2030, 2035, and 2040.

The age_band_multiplier_schedule from the config represents ONS-projected changes
in age distribution relative to the 2023 baseline. This validation checks if:
1. New entrants entering at age 65 (realistic)
2. Natural aging through age bands
3. Calibrated mortality rates

...combine to produce age distributions that match ONS projections.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from IBM_PD_AD_v3 import run_model, general_config
from ons_projection_data import (
    BASELINE_AGE_DISTRIBUTION_2023,
    ONS_AGE_BAND_MULTIPLIER_SCHEDULE,
    get_ons_projected_distribution
)


def extract_ons_projections():
    """
    Extract ONS population projections from ons_projection_data module.

    Returns DataFrame with columns: year, age_lower, age_upper, age_band, ons_proportion
    """
    records = []

    # Add baseline year (2023)
    for (age_lower, age_upper), weight in BASELINE_AGE_DISTRIBUTION_2023.items():
        age_band = f"{age_lower}-{age_upper if age_upper != 100 else '+'}"
        records.append({
            'year': 2023,
            'age_lower': age_lower,
            'age_upper': age_upper,
            'age_band': age_band,
            'ons_proportion': weight
        })

    # Add projection years
    for year in sorted(ONS_AGE_BAND_MULTIPLIER_SCHEDULE.keys()):
        distribution = get_ons_projected_distribution(year)
        for (age_lower, age_upper), proportion in distribution.items():
            age_band = f"{age_lower}-{age_upper if age_upper != 100 else '+'}"
            records.append({
                'year': year,
                'age_lower': age_lower,
                'age_upper': age_upper,
                'age_band': age_band,
                'ons_proportion': proportion
            })

    return pd.DataFrame(records)


def extract_model_age_distribution(results, year):
    """
    Extract age distribution from model results for a specific year.

    Returns DataFrame with columns: age_lower, age_upper, age_band,
                                     population_count, model_proportion
    """
    df = results['incidence_by_year_sex_df']
    year_df = df[(df['calendar_year'] == year) & (df['sex'].isin(['male', 'female']))].copy()

    if year_df.empty:
        print(f"Warning: No data for year {year}")
        return pd.DataFrame()

    # Group by age band (sum across sexes)
    grouped = (
        year_df.groupby(['age_lower', 'age_upper'], as_index=False)['population_alive_in_band']
        .sum()
        .rename(columns={'population_alive_in_band': 'population_count'})
    )

    # Calculate proportions
    total_pop = grouped['population_count'].sum()
    grouped['model_proportion'] = grouped['population_count'] / total_pop if total_pop > 0 else 0.0

    # Add age_band label
    grouped['age_band'] = grouped.apply(
        lambda r: f"{int(r['age_lower'])}-{int(r['age_upper']) if pd.notna(r['age_upper']) and r['age_upper'] != 100 else '+'}",
        axis=1
    )

    return grouped


def calculate_fit_statistics(observed, predicted):
    """Calculate R² and slope from OLS regression"""
    x = np.array(predicted)
    y = np.array(observed)

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
        'alpha': beta[0],
        'beta': beta[1],
        'r2': r2
    }


def validate_age_distribution_for_year(config, year, seed=42, save_dir="plots"):
    """
    Run model to specific year and validate age distribution against ONS projections.
    """
    print(f"\n{'='*80}")
    print(f"Validating Age Distribution for {year}")
    print(f"{'='*80}")

    # Run model
    validation_config = copy.deepcopy(config)
    base_year = int(validation_config.get('base_year', 2023))
    timesteps_needed = year - base_year
    validation_config['number_of_timesteps'] = timesteps_needed

    print(f"Running model: {base_year} -> {year} ({timesteps_needed} timesteps)")
    print(f"Population: {validation_config.get('population', 0):,}")

    results = run_model(validation_config, seed=seed, return_agents=False)

    # Extract ONS projections
    ons_df = extract_ons_projections()
    ons_year = ons_df[ons_df['year'] == year].copy()

    if ons_year.empty:
        print(f"Warning: No ONS projection data for year {year}")
        return None

    # Extract model results
    model_df = extract_model_age_distribution(results, year)

    if model_df.empty:
        print(f"Warning: No model data for year {year}")
        return None

    # Merge
    merged = model_df.merge(
        ons_year[['age_lower', 'age_upper', 'age_band', 'ons_proportion']],
        on=['age_lower', 'age_upper', 'age_band'],
        how='inner'
    )

    # Calculate errors
    merged['abs_error'] = merged['model_proportion'] - merged['ons_proportion']
    merged['rel_error_pct'] = (merged['abs_error'] / merged['ons_proportion']) * 100

    # Calculate fit statistics
    fit_stats = calculate_fit_statistics(merged['ons_proportion'], merged['model_proportion'])

    # Print summary
    print(f"\nAge Distribution Comparison ({year}):")
    print(f"{'Age Band':<12} {'ONS %':<10} {'Model %':<10} {'Abs Error':<12} {'Rel Error %':<12}")
    print('-' * 60)
    for _, row in merged.iterrows():
        print(f"{row['age_band']:<12} {row['ons_proportion']*100:>8.2f}% "
              f"{row['model_proportion']*100:>8.2f}% {row['abs_error']*100:>10.2f}% "
              f"{row['rel_error_pct']:>10.1f}%")

    print(f"\nFit Statistics:")
    print(f"  R² = {fit_stats['r2']:.4f}")
    print(f"  Slope (β) = {fit_stats['beta']:.4f}")
    print(f"  Intercept (α) = {fit_stats['alpha']:.6f}")

    # Save comparison table
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    table_path = save_dir / f"age_distribution_validation_{year}.csv"
    merged[['age_band', 'ons_proportion', 'model_proportion', 'abs_error', 'rel_error_pct']].to_csv(
        table_path, index=False, float_format='%.6f'
    )
    print(f"\nTable saved: {table_path}")

    return {
        'year': year,
        'merged': merged,
        'fit_stats': fit_stats,
        'table_path': table_path
    }


def create_validation_plots(validation_results, save_dir="plots"):
    """
    Create visualization of age distribution validation across all years.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Number of years to plot
    n_years = len(validation_results)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, result in enumerate(validation_results):
        if result is None:
            continue

        ax = axes[idx]
        merged = result['merged']
        year = result['year']
        fit_stats = result['fit_stats']

        # Scatter plot
        ax.scatter(merged['model_proportion'], merged['ons_proportion'],
                  s=150, alpha=0.7, edgecolors='black', linewidths=1.5)

        # Perfect agreement line
        max_val = max(merged['model_proportion'].max(), merged['ons_proportion'].max()) * 1.1
        x_line = np.linspace(0, max_val, 100)
        ax.plot(x_line, x_line, 'k--', label='Perfect agreement', linewidth=2)

        # Fitted line
        ax.plot(x_line, fit_stats['alpha'] + fit_stats['beta'] * x_line, 'r-',
               label=f"Fit (β={fit_stats['beta']:.3f}, R²={fit_stats['r2']:.3f})", linewidth=2)

        # Annotate points
        for _, row in merged.iterrows():
            ax.annotate(row['age_band'],
                       (row['model_proportion'], row['ons_proportion']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('Model Proportion', fontsize=11)
        ax.set_ylabel('ONS Projected Proportion', fontsize=11)
        ax.set_title(f'{year}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Age Distribution Validation Against ONS Projections',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    plot_path = save_dir / "age_distribution_validation_all_years.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCombined plot saved: {plot_path}")

    return plot_path


def run_full_age_distribution_validation(seed=42, save_dir="plots"):
    """
    Run complete age distribution validation for all ONS projection years.
    """
    print("="*80)
    print("AGE DISTRIBUTION VALIDATION - ONS PROJECTIONS")
    print("="*80)
    print("\nThis validates that model aging + mortality dynamics correctly")
    print("reproduce ONS-projected age distributions for the 65+ population.")

    # Get projection years from ONS data
    projection_years = sorted(ONS_AGE_BAND_MULTIPLIER_SCHEDULE.keys())

    if not projection_years:
        print("\nError: No ONS projection data found")
        return None

    print(f"\nValidation years: {', '.join(map(str, projection_years))}")

    # Run validation for each year
    results = []
    for year in projection_years:
        result = validate_age_distribution_for_year(general_config, year, seed=seed, save_dir=save_dir)
        results.append(result)

    # Create combined plots
    create_validation_plots(results, save_dir=save_dir)

    # Summary table
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Year':<8} {'R²':<10} {'Slope (β)':<12} {'Mean Abs Error':<18}")
    print('-' * 80)

    for result in results:
        if result is None:
            continue
        year = result['year']
        fit_stats = result['fit_stats']
        mean_abs_error = result['merged']['abs_error'].abs().mean() * 100

        print(f"{year:<8} {fit_stats['r2']:<10.4f} {fit_stats['beta']:<12.4f} {mean_abs_error:<18.2f}%")

    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = run_full_age_distribution_validation(seed=42, save_dir="plots")
    print("\n[SUCCESS] Age distribution validation complete!")
