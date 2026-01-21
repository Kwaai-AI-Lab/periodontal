"""
Periodontal Disease Sensitivity Analysis for Tornado Diagrams

This module provides efficient one-way sensitivity analysis specifically for:
1. Periodontal disease onset relative risk
2. Periodontal disease severe-to-death relative risk

Uses reduced population with scaling to make tornado diagram generation computationally feasible.
"""

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from IBM_PD_AD import run_model, extract_psa_metrics


def run_pd_sensitivity_analysis(
    base_config: dict,
    *,
    population_fraction: float = 0.01,
    n_replicates: int = 10,
    combine_sexes: bool = True,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """
    One-way sensitivity analysis for periodontal disease relative risks.

    Varies only:
    - PD onset RR (95% CI: 1.32-1.65, base: 1.47)
    - PD severe_to_death RR (95% CI: 1.10-1.69, base: 1.36)

    Args:
        base_config: Base model configuration
        population_fraction: Fraction of population to use (default 0.01 = 1%)
        n_replicates: Number of replicate runs per parameter value (reduces stochastic noise)
        combine_sexes: If True, vary both sexes together; if False, vary separately
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (None = auto-detect)

    Returns:
        DataFrame with columns: parameter, value_type (baseline/low/high),
                               metric values, replicate number
    """
    # Get original and reduced population sizes
    original_pop = base_config.get('population', 33167098)
    reduced_pop = int(original_pop * population_fraction)

    print("=" * 70)
    print("PERIODONTAL DISEASE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Population: {reduced_pop:,} agents ({population_fraction:.1%} of {original_pop:,})")
    print(f"Replicates per parameter value: {n_replicates}")
    print(f"Parallel jobs: {n_jobs or 'auto-detect'}")
    print()

    # Create reduced population config
    working_config = copy.deepcopy(base_config)
    working_config['population'] = reduced_pop

    # Store original population for potential scaling by run_model
    if 'psa' not in working_config:
        working_config['psa'] = {}
    working_config['psa']['original_population'] = original_pop

    # Define parameters to test based on confidence intervals
    # Format: (low_95CI, high_95CI)
    parameters = {
        'onset_rr': (1.32, 1.65),           # 95% CI for onset
        'severe_to_death_rr': (1.10, 1.69)  # 95% CI for severe_to_death
    }

    results_list = []
    rng = np.random.default_rng(seed)

    # Helper function to modify config
    def set_pd_rr(config: dict, onset_rr: Optional[float] = None,
                  std_rr: Optional[float] = None) -> dict:
        """Set periodontal disease relative risks."""
        cfg = copy.deepcopy(config)

        if onset_rr is not None:
            cfg['risk_factors']['periodontal_disease']['relative_risks']['onset']['female'] = onset_rr
            cfg['risk_factors']['periodontal_disease']['relative_risks']['onset']['male'] = onset_rr

        if std_rr is not None:
            cfg['risk_factors']['periodontal_disease']['relative_risks']['severe_to_death']['female'] = std_rr
            cfg['risk_factors']['periodontal_disease']['relative_risks']['severe_to_death']['male'] = std_rr

        return cfg

    # Helper to run a single replicate
    def run_replicate(config: dict, param_name: str, value_type: str,
                     rep_num: int, rep_seed: int) -> dict:
        """Run a single model replicate."""
        result = run_model(config, seed=rep_seed)
        metrics = extract_psa_metrics(result)
        metrics['parameter'] = param_name
        metrics['value_type'] = value_type
        metrics['replicate'] = rep_num
        return metrics

    # 1. Run baseline
    print("Running baseline...")
    baseline_seeds = [int(rng.integers(0, 2**32-1)) for _ in range(n_replicates)]

    if n_jobs and n_jobs != 1:
        baseline_results = Parallel(n_jobs=n_jobs)(
            delayed(run_replicate)(working_config, 'baseline', 'baseline', i, s)
            for i, s in enumerate(baseline_seeds)
        )
    else:
        baseline_results = []
        for i, s in enumerate(baseline_seeds):
            print(f"  Replicate {i+1}/{n_replicates}", end='\r')
            baseline_results.append(run_replicate(working_config, 'baseline', 'baseline', i, s))
        print()

    results_list.extend(baseline_results)
    baseline_mean = pd.DataFrame(baseline_results)['total_qalys_combined'].mean()
    print(f"  Baseline mean QALYs: {baseline_mean:,.0f}")
    print()

    # 2. Test onset RR
    print("Testing PD Onset RR (95% CI: 1.32-1.65)...")
    for value_type, rr_value in [('low', parameters['onset_rr'][0]),
                                  ('high', parameters['onset_rr'][1])]:
        test_config = set_pd_rr(working_config, onset_rr=rr_value)
        rep_seeds = [int(rng.integers(0, 2**32-1)) for _ in range(n_replicates)]

        if n_jobs and n_jobs != 1:
            reps = Parallel(n_jobs=n_jobs)(
                delayed(run_replicate)(test_config, 'onset_rr', value_type, i, s)
                for i, s in enumerate(rep_seeds)
            )
        else:
            reps = []
            for i, s in enumerate(rep_seeds):
                print(f"  {value_type.capitalize()}: Replicate {i+1}/{n_replicates}", end='\r')
                reps.append(run_replicate(test_config, 'onset_rr', value_type, i, s))
            print()

        results_list.extend(reps)
        mean_qalys = pd.DataFrame(reps)['total_qalys_combined'].mean()
        print(f"  {value_type.capitalize()} (RR={rr_value}): {mean_qalys:,.0f} QALYs "
              f"(Δ={mean_qalys-baseline_mean:+,.0f})")
    print()

    # 3. Test severe_to_death RR
    print("Testing PD Severe-to-Death RR (95% CI: 1.10-1.69)...")
    for value_type, rr_value in [('low', parameters['severe_to_death_rr'][0]),
                                  ('high', parameters['severe_to_death_rr'][1])]:
        test_config = set_pd_rr(working_config, std_rr=rr_value)
        rep_seeds = [int(rng.integers(0, 2**32-1)) for _ in range(n_replicates)]

        if n_jobs and n_jobs != 1:
            reps = Parallel(n_jobs=n_jobs)(
                delayed(run_replicate)(test_config, 'severe_to_death_rr', value_type, i, s)
                for i, s in enumerate(rep_seeds)
            )
        else:
            reps = []
            for i, s in enumerate(rep_seeds):
                print(f"  {value_type.capitalize()}: Replicate {i+1}/{n_replicates}", end='\r')
                reps.append(run_replicate(test_config, 'severe_to_death_rr', value_type, i, s))
            print()

        results_list.extend(reps)
        mean_qalys = pd.DataFrame(reps)['total_qalys_combined'].mean()
        print(f"  {value_type.capitalize()} (RR={rr_value}): {mean_qalys:,.0f} QALYs "
              f"(Δ={mean_qalys-baseline_mean:+,.0f})")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(results_list)

    # Scale metrics by population fraction
    print(f"Scaling results by factor of {1/population_fraction:.0f}x...")
    scale_factor = 1.0 / population_fraction
    metrics_to_scale = [
        'total_costs_nhs', 'total_costs_informal', 'total_costs_all',
        'total_qalys_patient', 'total_qalys_caregiver', 'total_qalys_combined',
        'incident_onsets_total', 'stage_mild', 'stage_moderate', 'stage_severe'
    ]

    for metric in metrics_to_scale:
        if metric in df.columns:
            df[metric] = df[metric] * scale_factor

    print("✓ Sensitivity analysis complete!")
    print("=" * 70)
    print()

    return df


def create_pd_tornado_diagram(
    sensitivity_results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    save_path: str = 'plots/pd_tornado_diagram.png',
    show: bool = False
) -> None:
    """
    Create tornado diagram for periodontal disease sensitivity analysis.

    Args:
        sensitivity_results: DataFrame from run_pd_sensitivity_analysis
        metrics: List of metrics to plot (if None, plots key metrics)
        save_path: Output file path
        show: Whether to display plot
    """
    if metrics is None:
        metrics = [
            'total_qalys_combined',
            'incident_onsets_total',
            'total_costs_all'
        ]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in sensitivity_results.columns]

    if not available_metrics:
        print("No valid metrics found in results.")
        return

    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    # Process each metric
    for ax, metric in zip(axes, available_metrics):
        # Calculate mean for each parameter/value_type
        summary = sensitivity_results.groupby(['parameter', 'value_type'])[metric].mean().unstack()

        # Calculate swing (high - low) for sorting
        summary['swing'] = abs(summary['high'] - summary['low'])
        summary = summary.sort_values('swing', ascending=True)

        # Get baseline
        baseline_val = sensitivity_results[
            sensitivity_results['value_type'] == 'baseline'
        ][metric].mean()

        # Parameter labels
        param_labels = {
            'onset_rr': 'PD Onset RR\n(1.32-1.65)',
            'severe_to_death_rr': 'PD Severe→Death RR\n(1.10-1.69)'
        }

        # Plot bars
        y_pos = np.arange(len(summary))
        for i, (param, row) in enumerate(summary.iterrows()):
            low = row['low']
            high = row['high']

            # Horizontal bar from low to high
            ax.barh(i, high - low, left=low, height=0.6,
                   color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)

            # Mark low and high endpoints
            ax.plot([low, low], [i-0.3, i+0.3], 'k-', linewidth=2.5)
            ax.plot([high, high], [i-0.3, i+0.3], 'k-', linewidth=2.5)

        # Baseline reference line
        ax.axvline(baseline_val, color='red', linestyle='--', linewidth=2,
                  label=f'Baseline', zorder=10)

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([param_labels.get(p, p) for p in summary.index])

        # X-axis label based on metric
        xlabel_map = {
            'total_qalys_combined': 'Total QALYs',
            'incident_onsets_total': 'Dementia Cases',
            'total_costs_all': 'Total Costs (£)',
            'total_costs_nhs': 'NHS Costs (£)',
        }
        ax.set_xlabel(xlabel_map.get(metric, metric.replace('_', ' ').title()),
                     fontsize=11, fontweight='bold')

        # Title
        ax.set_title(xlabel_map.get(metric, metric.replace('_', ' ').title()),
                    fontsize=12, fontweight='bold', pad=10)

        # Grid and legend
        ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(fontsize=10)

        # Format x-axis
        ax.ticklabel_format(style='plain', axis='x')
        if metric in ['total_costs_all', 'total_costs_nhs']:
            # Format as currency
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, p: f'£{x/1e9:.1f}B' if abs(x) >= 1e9 else f'£{x/1e6:.1f}M'
            ))
        elif metric in ['total_qalys_combined', 'total_qalys_patient']:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, p: f'{x/1e6:.2f}M'
            ))
        elif metric == 'incident_onsets_total':
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, p: f'{x/1e3:.0f}K'
            ))

    # Overall title
    fig.suptitle('Periodontal Disease Sensitivity Analysis - Tornado Diagram',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    print(f"\n✓ Saved tornado diagram to {save_path}")


def export_pd_sensitivity_results(
    sensitivity_results: pd.DataFrame,
    excel_path: str = 'pd_sensitivity_analysis.xlsx'
) -> None:
    """
    Export sensitivity analysis results to Excel with summary statistics.

    Args:
        sensitivity_results: DataFrame from run_pd_sensitivity_analysis
        excel_path: Output Excel file path
    """
    print(f"\nExporting results to {excel_path}...")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Raw results
        sensitivity_results.to_excel(writer, sheet_name='Raw_Results', index=False)

        # Sheet 2: Summary statistics
        summary_list = []

        for param in sensitivity_results['parameter'].unique():
            for metric in ['total_qalys_combined', 'incident_onsets_total',
                          'total_costs_all', 'total_costs_nhs']:
                if metric not in sensitivity_results.columns:
                    continue

                param_data = sensitivity_results[sensitivity_results['parameter'] == param]

                baseline_mean = sensitivity_results[
                    sensitivity_results['value_type'] == 'baseline'
                ][metric].mean()

                for value_type in ['low', 'high']:
                    data = param_data[param_data['value_type'] == value_type][metric]

                    if len(data) > 0:
                        mean_val = data.mean()
                        std_val = data.std()

                        summary_list.append({
                            'Parameter': param,
                            'Value_Type': value_type,
                            'Metric': metric,
                            'Mean': mean_val,
                            'Std': std_val,
                            'Baseline': baseline_mean,
                            'Difference': mean_val - baseline_mean,
                            'Pct_Change': ((mean_val - baseline_mean) / baseline_mean * 100)
                                         if baseline_mean != 0 else 0
                        })

        summary_df = pd.DataFrame(summary_list)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        # Sheet 3: Swing analysis
        swing_list = []

        for param in sensitivity_results['parameter'].unique():
            for metric in ['total_qalys_combined', 'incident_onsets_total',
                          'total_costs_all']:
                if metric not in sensitivity_results.columns:
                    continue

                param_data = sensitivity_results[sensitivity_results['parameter'] == param]

                low_mean = param_data[param_data['value_type'] == 'low'][metric].mean()
                high_mean = param_data[param_data['value_type'] == 'high'][metric].mean()

                swing_list.append({
                    'Parameter': param,
                    'Metric': metric,
                    'Low_Value': low_mean,
                    'High_Value': high_mean,
                    'Swing': abs(high_mean - low_mean),
                    'Swing_Pct': abs((high_mean - low_mean) / low_mean * 100)
                                if low_mean != 0 else 0
                })

        swing_df = pd.DataFrame(swing_list).sort_values('Swing', ascending=False)
        swing_df.to_excel(writer, sheet_name='Swing_Analysis', index=False)

    print(f"✓ Exported results to {excel_path}")
    print(f"  - Raw_Results: All replicate data")
    print(f"  - Summary_Statistics: Mean/std for each parameter value")
    print(f"  - Swing_Analysis: Parameter influence (high-low)")


# Example usage
if __name__ == "__main__":
    from IBM_PD_AD import general_config

    # Load your base configuration
    config = general_config  # Use the built-in configuration

    # Run sensitivity analysis with 1% population
    results = run_pd_sensitivity_analysis(
        config,
        population_fraction=0.01,
        n_replicates=10,
        seed=42,
        n_jobs=-1
    )

    # Create tornado diagram
    create_pd_tornado_diagram(
        results,
        save_path='plots/pd_tornado_diagram.png',
        show=False
    )

    # Export to Excel
    export_pd_sensitivity_results(
        results,
        excel_path='pd_sensitivity_analysis.xlsx'
    )
