"""
External Validation Script for IBM Dementia Model
Validates model predictions against observed 2024 prevalence data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IBM_PD_AD import run_model, general_config
import copy


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
    """Run model to 2024 and return results"""
    print("Running model to 2024...")

    # Create a copy of config to modify
    validation_config = copy.deepcopy(config)

    # Ensure model runs to 2024
    validation_config['end_year'] = 2024

    # Run the model
    results = run_model(validation_config, seed=seed)

    print(f"Model run complete. End year: {validation_config['end_year']}")
    return results


def extract_prevalence_by_age_sex(results, year=2024):
    """
    Extract prevalence from model results by age group and sex

    Returns DataFrame with columns: age_lower, age_upper, sex, prevalence
    """
    print(f"\nExtracting prevalence for year {year}...")

    # Get the agents dataframe
    agents = results['agents']

    # Filter to agents alive in the target year
    alive_agents = agents[agents['year_of_death'] >= year].copy()

    print(f"Total agents alive in {year}: {len(alive_agents)}")

    # Calculate age in target year
    alive_agents['age_in_year'] = year - alive_agents['year_of_birth']

    # Define age bands
    age_bands = [
        (35, 49),
        (50, 64),
        (65, 79),
        (80, None),
    ]

    prevalence_data = []

    for age_lower, age_upper in age_bands:
        for sex in ['F', 'M']:
            # Filter by age and sex
            if age_upper is None:
                age_mask = (alive_agents['age_in_year'] >= age_lower)
                age_label = f"{age_lower}+"
            else:
                age_mask = (alive_agents['age_in_year'] >= age_lower) & (alive_agents['age_in_year'] <= age_upper)
                age_label = f"{age_lower}-{age_upper}"

            sex_mask = alive_agents['sex'] == sex
            subset = alive_agents[age_mask & sex_mask]

            # Count total in age-sex group
            n_total = len(subset)

            if n_total == 0:
                print(f"  Warning: No agents in {age_label}, {sex} group")
                prevalence = 0.0
            else:
                # Count those with dementia (year_of_onset <= year and year_of_death >= year)
                n_with_dementia = len(subset[subset['year_of_onset'] <= year])
                prevalence = n_with_dementia / n_total

                print(f"  {age_label:8s} {sex}: {n_with_dementia:6d}/{n_total:6d} = {prevalence:.6f}")

            prevalence_data.append({
                'age_lower': age_lower,
                'age_upper': age_upper,
                'sex': sex,
                'prevalence': prevalence,
                'age_band': age_label,
                'n_total': n_total,
                'n_with_dementia': n_with_dementia if n_total > 0 else 0
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
        obs_subset[['age_lower', 'observed']],
        on='age_lower',
        how='left'
    )

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


def run_external_validation(seed=42, save_dir=None):
    """
    Main function to run external validation

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    save_dir : Path or str
        Directory to save outputs (default: plots/)
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

    # 1. Run model to 2024
    results = run_model_to_2024(general_config, seed=seed)

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
            observed_df[observed_df['sex'] == sex][['age_lower', 'observed']],
            on='age_lower'
        )

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
    # Run validation
    validation_results = run_external_validation(seed=42)

    print("\n✓ External validation complete!")
    print(f"  Check the 'plots/' directory for output files")
