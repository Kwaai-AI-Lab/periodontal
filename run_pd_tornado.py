"""
Example script to run periodontal disease sensitivity analysis for tornado diagrams.

This script:
1. Loads the base configuration
2. Runs sensitivity analysis on PD onset RR
3. Generates tornado diagrams
4. Exports results to Excel

Computational note:
- Uses 1% of population by default (107,875 agents from base of 10,787,479)
- Runs 10 replicates per parameter value
- Total: ~30 model runs (baseline + 1 param × 2 values × 10 reps each)
- Estimated time: Several hours depending on CPU cores and model complexity
"""

from IBM_PD_AD_v2 import general_config
from pd_sensitivity_analysis import (
    run_pd_sensitivity_analysis,
    create_pd_tornado_diagram,
    export_pd_sensitivity_results
)


def main():
    """Run complete PD sensitivity analysis workflow."""

    print("\n" + "="*70)
    print("PERIODONTAL DISEASE TORNADO DIAGRAM SENSITIVITY ANALYSIS")
    print("="*70 + "\n")

    # Use default config or load from file
    config = general_config.copy()

    # Optional: Customize config
    # config['population'] = 10787479  # This will be reduced to 1% automatically
    # config['number_of_timesteps'] = 17
    # config['base_year'] = 2023

    print("Configuration:")
    print(f"  Base population: {config.get('population', 10787479):,} agents")
    print(f"  Time horizon: {config.get('number_of_timesteps', 17)} timesteps")
    print(f"  Base year: {config.get('base_year', 2023)}")
    print()

    # Step 1: Run sensitivity analysis
    print("STEP 1: Running sensitivity analysis...")
    print("-" * 70)

    results = run_pd_sensitivity_analysis(
        config,
        population_fraction=0.01,   # Use 1% of population
        n_replicates=10,            # 10 replicates per parameter value
        combine_sexes=True,         # Vary both sexes together
        seed=42,                    # For reproducibility
        n_jobs=1                    # Use 4 cores (Windows resource limits)
    )

    print(f"\nOK Generated {len(results)} result rows")
    print()

    # Step 2: Create tornado diagrams
    print("STEP 2: Creating tornado diagrams...")
    print("-" * 70)

    # Create diagram for key metrics
    create_pd_tornado_diagram(
        results,
        metrics=['total_qalys_combined', 'incident_onsets_total', 'total_costs_all'],
        save_path='plots/pd_tornado_main.png',
        show=False
    )

    # Optional: Create separate diagrams for specific metrics
    for metric, label in [
        ('total_qalys_combined', 'QALYs'),
        ('incident_onsets_total', 'Incidence'),
        ('total_costs_all', 'Costs')
    ]:
        create_pd_tornado_diagram(
            results,
            metrics=[metric],
            save_path=f'plots/pd_tornado_{label.lower()}.png',
            show=False
        )

    print("\nOK All tornado diagrams created in plots/")
    print()

    # Step 3: Export to Excel
    print("STEP 3: Exporting results to Excel...")
    print("-" * 70)

    export_pd_sensitivity_results(
        results,
        excel_path='pd_sensitivity_analysis.xlsx'
    )

    print()

    # Step 4: Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    # Calculate and display swing analysis
    import pandas as pd

    baseline = results[results['value_type'] == 'baseline']
    baseline_qalys = baseline['total_qalys_combined'].mean()
    baseline_cases = baseline['incident_onsets_total'].mean()

    print(f"\nBaseline Results (mean of {len(baseline)} replicates):")
    print(f"  Total QALYs: {baseline_qalys:,.0f}")
    print(f"  Dementia Cases: {baseline_cases:,.0f}")

    print("\nParameter Swings (High - Low):")

    for param in ['onset_rr']:
        param_data = results[results['parameter'] == param]

        low_qalys = param_data[param_data['value_type'] == 'low']['total_qalys_combined'].mean()
        high_qalys = param_data[param_data['value_type'] == 'high']['total_qalys_combined'].mean()
        swing_qalys = high_qalys - low_qalys

        low_cases = param_data[param_data['value_type'] == 'low']['incident_onsets_total'].mean()
        high_cases = param_data[param_data['value_type'] == 'high']['incident_onsets_total'].mean()
        swing_cases = high_cases - low_cases

        param_name = "PD Onset RR"

        print(f"\n  {param_name}:")
        print(f"    QALY swing: {swing_qalys:+,.0f} ({abs(swing_qalys/baseline_qalys*100):.2f}%)")
        print(f"    Case swing: {swing_cases:+,.0f} ({abs(swing_cases/baseline_cases*100):.2f}%)")

    print("\n" + "="*70)
    print("OK ANALYSIS COMPLETE!")
    print("="*70)

    print("\nOutput files:")
    print("  📊 plots/pd_tornado_main.png - Combined tornado diagram")
    print("  📊 plots/pd_tornado_qalys.png - QALYs tornado diagram")
    print("  📊 plots/pd_tornado_incidence.png - Incidence tornado diagram")
    print("  📊 plots/pd_tornado_costs.png - Costs tornado diagram")
    print("  📄 pd_sensitivity_analysis.xlsx - Detailed results")
    print()


if __name__ == "__main__":
    main()
