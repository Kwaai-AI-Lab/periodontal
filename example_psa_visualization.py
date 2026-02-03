# PSA Visualization Example Script

"""
This script demonstrates how to create visualizations for PSA results.

Current capabilities:
1. Baseline time series (incidence, prevalence, costs, QALYs over time)
2. PSA summary metrics with 95% CI error bars
3. Excel exports with all results

Limitation: Time-series plots with shaded CIs require storing time-series data
from each PSA draw, which is computationally expensive. The current implementation
shows baseline trends and PSA uncertainty for summary metrics.
"""

from IBM_PD_AD import *

if __name__ == "__main__":
    run_seed = 42

    # ========================================
    # 1. Run Baseline Model (for time series)
    # ========================================
    print("="*60)
    print("STEP 1: Running Baseline Model")
    print("="*60)

    baseline_results = run_model(general_config, seed=run_seed)

    # Export baseline to Excel
    export_results_to_excel(baseline_results, path='results/Baseline_Model.xlsx')

    # ========================================
    # 2. Plot Baseline Time Series
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: Creating Baseline Time Series Plots")
    print("="*60)

    # These functions use existing plotting capabilities
    plot_ad_prevalence(baseline_results, save_path="results/plots/baseline_prevalence.png")
    plot_ad_incidence(baseline_results, save_path="results/plots/baseline_incidence.png")
    plot_costs_per_person_over_time(baseline_results, save_path="results/plots/baseline_costs.png")
    plot_qalys_per_person_over_time(baseline_results, save_path="results/plots/baseline_qalys.png")

    print("Baseline time series plots saved to results/plots/")

    # ========================================
    # 3. Run Pilot Study (optional but recommended)
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: Running Variance Components Pilot Study")
    print("="*60)

    pilot_results = estimate_variance_components_pilot(
        general_config,
        general_config['psa'],
        n_outer=10,
        n_inner_list=[1000, 5000, 10000],
        seed=run_seed
    )

    save_results_compressed(pilot_results, 'results/pilot_study.pkl.gz')

    # ========================================
    # 4. Run Two-Level PSA
    # ========================================
    print("\n" + "="*60)
    print("STEP 4: Running Two-Level PSA (O'Hagan Method)")
    print("="*60)

    psa_results = run_two_level_psa(
        general_config,
        general_config['psa'],
        n_outer=1000,  # Full 1000 iterations
        variance_pilot_results=pilot_results,
        collect_draw_level=True,
        seed=run_seed,
        n_jobs=None  # Use all cores
    )

    # Save PSA results
    save_results_compressed(psa_results, 'results/two_level_psa.pkl.gz')

    # Export PSA to Excel
    export_psa_results_to_excel(
        psa_results,
        path='results/PSA_Results.xlsx',
        include_draws=False
    )

    # ========================================
    # 5. Plot PSA Summary Metrics with 95% CI
    # ========================================
    print("\n" + "="*60)
    print("STEP 5: Creating PSA Summary Plots")
    print("="*60)

    plot_psa_summary_metrics(
        psa_results,
        save_path="results/plots/psa_summary_with_ci.png",
        show=False
    )

    print("PSA summary plot saved to results/plots/psa_summary_with_ci.png")

    # ========================================
    # 6. Print Summary Results
    # ========================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    summary = psa_results['summary']

    print("\nKey Metrics (Mean with 95% CI):")
    print("-" * 60)

    for metric in ['total_costs_all', 'total_qalys_combined', 'incident_onsets_total']:
        if metric in summary:
            stats = summary[metric]
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:,.0f}")
            print(f"  95% CI: [{stats['lower_95']:,.0f}, {stats['upper_95']:,.0f}]")
            ci_width = stats['upper_95'] - stats['lower_95']
            rel_width = (ci_width / stats['mean']) * 100 if stats['mean'] > 0 else 0
            print(f"  CI Width: {ci_width:,.0f} ({rel_width:.1f}% of mean)")

    # Efficiency summary
    if 'two_level_psa' in psa_results:
        two_level_info = psa_results['two_level_psa']
        print("\n" + "="*60)
        print("COMPUTATIONAL EFFICIENCY")
        print("="*60)
        print(f"Original population: {two_level_info['original_population']:,}")
        print(f"Optimized population (n): {two_level_info['n_inner']:,}")
        print(f"Reduction factor: {two_level_info['reduction_factor']:.0f}x")
        print(f"PSA iterations (N): {two_level_info['n_outer']:,}")
        print(f"\nTotal individuals simulated:")
        print(f"  With optimization: {two_level_info['n_outer'] * two_level_info['n_inner']:,}")
        print(f"  Without optimization: {two_level_info['n_outer'] * two_level_info['original_population']:,}")
        savings = two_level_info['n_outer'] * two_level_info['original_population'] / (two_level_info['n_outer'] * two_level_info['n_inner'])
        print(f"  Computational savings: {savings:.0f}x fewer individuals")

    print("\n" + "="*60)
    print("OUTPUTS GENERATED")
    print("="*60)
    print("Excel files:")
    print("  - results/Baseline_Model.xlsx (time series, distributions)")
    print("  - results/PSA_Results.xlsx (mean + 95% CI for all metrics)")
    print("\nPlots:")
    print("  - results/plots/baseline_prevalence.png")
    print("  - results/plots/baseline_incidence.png")
    print("  - results/plots/baseline_costs.png")
    print("  - results/plots/baseline_qalys.png")
    print("  - results/plots/psa_summary_with_ci.png")
    print("\nCompressed data:")
    print("  - results/pilot_study.pkl.gz")
    print("  - results/two_level_psa.pkl.gz")

    print("\n✅ Analysis complete!")

    # ========================================
    # Note on Time-Series CI Plots
    # ========================================
    print("\n" + "="*60)
    print("NOTE: Time-Series Plots with Shaded CIs")
    print("="*60)
    print("""
    The current implementation shows:
    - Baseline time series (single run)
    - PSA uncertainty for SUMMARY metrics (totals)

    To create line graphs with shaded CIs over time for incidence/prevalence,
    you would need to:
    1. Store time-series data from each PSA draw (expensive)
    2. Calculate mean and 95% CI for each time point across draws

    This can be implemented as a future enhancement if needed for your analysis.
    The summary metrics with CIs (bar charts) capture the overall uncertainty
    in total costs, QALYs, and dementia onsets.
    """)
