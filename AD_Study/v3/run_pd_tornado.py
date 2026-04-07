"""
One-way sensitivity analysis for periodontal disease hazard ratio - Baseline (50% stable).

This script:
1. Loads the base configuration
2. Runs deterministic sensitivity analysis on PD onset HR (1.07, 1.21, 1.38)
3. Exports results to Excel (tornado plots disabled - create manually from Excel data)

Note: Automatic tornado plot generation is disabled as plots do not render correctly.
      All data is exported to Excel for manual plotting.

Analysis approach:
- Scenario: Baseline (50% stable PD prevalence)
- Population: Full population (10,787,479 agents)
- Runs: Single deterministic run per HR value (no replicates)
- HR values: 1.07 (low), 1.21 (baseline), 1.38 (high)
- Total: 3 model runs
- Estimated time: ~3 hours total
"""

# Use the v3 configuration (hazard-ratio based risk factors)
from IBM_PD_AD_v3 import general_config
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
        population_fraction=1.0,    # Full population for deterministic comparison
        n_replicates=1,             # Single run per scenario (no replicates)
        combine_sexes=True,
        seed=42,                    # Fixed seed for reproducibility across scenarios
        paired_seeds=True,          # Reuse seed across baseline/low/high
        prevalence_values=[0.50],   # Baseline scenario: 50% stable prevalence
        n_jobs=1                    # Sequential runs
    )

    print(f"\nOK Generated {len(results)} result rows")
    print()

    # Step 2: Create tornado diagrams
    # DISABLED: Tornado plots do not render correctly and will be created manually
    # after Excel export. All data is available in the Excel file for manual plotting.
    print("STEP 2: Creating tornado diagrams...")
    print("-" * 70)
    print("  Tornado plot generation DISABLED (plots do not render correctly)")
    print("  All data is available in Excel for manual plotting")
    print()

    # # Create diagram for key metrics
    # create_pd_tornado_diagram(
    #     results,
    #     metrics=['total_qalys_combined', 'incident_onsets_total', 'total_costs_all'],
    #     save_path='plots/pd_tornado_main.png',
    #     show=False
    # )

    # # Optional: Create separate diagrams for specific metrics
    # for metric, label in [
    #     ('total_qalys_combined', 'QALYs'),
    #     ('incident_onsets_total', 'Incidence'),
    #     ('total_costs_all', 'Costs')
    # ]:
    #     create_pd_tornado_diagram(
    #         results,
    #         metrics=[metric],
    #         save_path=f'plots/pd_tornado_{label.lower()}.png',
    #         show=False
    #     )

    # print("\nOK All tornado diagrams created in plots/")
    # print()

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
    print("  📄 pd_sensitivity_analysis.xlsx - Detailed results (use this for manual plotting)")
    print()
    print("Note: Tornado plots are disabled. All data is available in the Excel file.")
    print("      Create tornado diagrams manually using the 'Baseline', 'Low', and 'High' sheets.")
    print()


if __name__ == "__main__":
    main()
