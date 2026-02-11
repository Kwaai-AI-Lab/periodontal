"""
Periodontal Disease Sensitivity Analysis for Tornado Diagrams

One-way deterministic sensitivity analysis for PD onset hazard ratio,
using reduced-population runs with scaling.
"""

import copy
from pathlib import Path
from typing import Optional, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed

from IBM_PD_AD_v3 import (
    run_model,
    extract_psa_metrics,
    _with_scaled_population_and_entrants,
)


def run_pd_sensitivity_analysis(
    base_config: dict,
    *,
    population_fraction: float = 0.01,
    n_replicates: int = 10,
    combine_sexes: bool = True,  # currently unused, kept for API compatibility
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    prevalence_values: Optional[Iterable[float]] = None  # e.g., [0.25, 0.50, 0.75]
) -> pd.DataFrame:
    """
    One-way sensitivity analysis for periodontal disease hazard ratio.

    Varies only:
    - PD onset HR (95% CI: 1.07-1.38)

    Runs each prevalence in `prevalence_values` (or the base config prevalence if None)
    with 10 replicates on 1% of the population, then scales count metrics back up.
    """
    original_pop = base_config.get('population', 10787479)
    rng = np.random.default_rng(seed)

    # Default prevalence list = whatever is in the input config (female value)
    if prevalence_values is None:
        pd_meta = base_config.get('risk_factors', {}).get('periodontal_disease', {})
        base_prev = (pd_meta.get('prevalence', {}) or {}).get('female', 0.50)
        prevalence_values = [base_prev]

    def set_pd_prevalence(config: dict, prevalence: float) -> dict:
        cfg = copy.deepcopy(config)
        pd_meta = cfg.setdefault('risk_factors', {}).setdefault('periodontal_disease', {})
        pd_meta.setdefault('prevalence', {})
        pd_meta['prevalence']['female'] = prevalence
        pd_meta['prevalence']['male'] = prevalence
        return cfg

    def set_pd_hr(config: dict, onset_hr: Optional[float] = None) -> dict:
        cfg = copy.deepcopy(config)
        if onset_hr is None:
            return cfg
        pd_meta = cfg.setdefault('risk_factors', {}).setdefault('periodontal_disease', {})
        hr_map = pd_meta.get('hazard_ratios')
        if isinstance(hr_map, dict):
            hr_map.setdefault('onset', {})
            hr_map['onset']['female'] = onset_hr
            hr_map['onset']['male'] = onset_hr
            return cfg
        rr_map = pd_meta.get('relative_risks')
        if isinstance(rr_map, dict):
            rr_map.setdefault('onset', {})
            rr_map['onset']['female'] = onset_hr
            rr_map['onset']['male'] = onset_hr
            return cfg
        pd_meta['hazard_ratios'] = {'onset': {'female': onset_hr, 'male': onset_hr}}
        return cfg

    def run_replicate(config: dict, param_name: str, value_type: str,
                      rep_num: int, rep_seed: int, prevalence: float) -> dict:
        result = run_model(config, seed=rep_seed)
        metrics = extract_psa_metrics(result)
        metrics['parameter'] = param_name
        metrics['value_type'] = value_type
        metrics['replicate'] = rep_num
        metrics['prevalence'] = prevalence
        return metrics

    parameters = {'onset_hr': (1.07, 1.38)}  # updated bounds
    results_list = []

    for prevalence in prevalence_values:
        # Build a scaled config per prevalence so populations aren’t cross-contaminated
        cfg_prev = set_pd_prevalence(base_config, prevalence)
        working_config = _with_scaled_population_and_entrants(
            cfg_prev,
            new_population=int(original_pop * population_fraction),
            original_population=original_pop
        )
        working_config.setdefault('psa', {})['original_population'] = original_pop

        print("=" * 70)
        print(f"PERIODONTAL DISEASE SENSITIVITY ANALYSIS | prevalence={prevalence:.0%}")
        print("=" * 70)
        print(f"Population: {int(original_pop * population_fraction):,} "
              f"agents ({population_fraction:.1%} of {original_pop:,})")
        print(f"Replicates per parameter value: {n_replicates}")
        print(f"Parallel jobs: {n_jobs or 'auto-detect'}")
        print()

        # Baseline
        print("Running baseline...")
        baseline_seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(n_replicates)]
        if n_jobs and n_jobs != 1:
            baseline_results = Parallel(n_jobs=n_jobs)(
                delayed(run_replicate)(working_config, 'baseline', 'baseline', i, s, prevalence)
                for i, s in enumerate(baseline_seeds)
            )
        else:
            baseline_results = []
            for i, s in enumerate(baseline_seeds):
                print(f"  Replicate {i+1}/{n_replicates}", end='\r')
                baseline_results.append(run_replicate(
                    working_config, 'baseline', 'baseline', i, s, prevalence))
            print()
        results_list.extend(baseline_results)
        baseline_mean = pd.DataFrame(baseline_results)['total_qalys_combined'].mean()
        print(f"  Baseline mean QALYs: {baseline_mean:,.0f}\n")

        # Onset HR low/high
        print("Testing PD Onset HR (95% CI: 1.07-1.38)...")
        for value_type, hr_value in [('low', parameters['onset_hr'][0]),
                                     ('high', parameters['onset_hr'][1])]:
            test_config = set_pd_hr(working_config, onset_hr=hr_value)
            rep_seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(n_replicates)]
            if n_jobs and n_jobs != 1:
                reps = Parallel(n_jobs=n_jobs)(
                    delayed(run_replicate)(test_config, 'onset_hr', value_type, i, s, prevalence)
                    for i, s in enumerate(rep_seeds)
                )
            else:
                reps = []
                for i, s in enumerate(rep_seeds):
                    print(f"  {value_type.capitalize()}: Replicate {i+1}/{n_replicates}", end='\r')
                    reps.append(run_replicate(
                        test_config, 'onset_hr', value_type, i, s, prevalence))
                print()
            results_list.extend(reps)
            mean_qalys = pd.DataFrame(reps)['total_qalys_combined'].mean()
            print(f"  {value_type.capitalize()} (HR={hr_value}): {mean_qalys:,.0f} QALYs "
                  f"(Delta={mean_qalys - baseline_mean:+,.0f})")
        print()

    df = pd.DataFrame(results_list)

    # Scale count metrics back up
    print(f"Scaling results by factor of {1/population_fraction:.0f}x...")
    scale_factor = 1.0 / population_fraction
    skip_keywords = ['rate', 'ratio', 'per_', 'prevalence', 'prob', 'hazard', 'mean', 'median']
    protected_cols = {'parameter', 'value_type', 'replicate', 'prevalence'}

    for col in df.columns:
        if col in protected_cols or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        name = col.lower()
        if any(key in name for key in skip_keywords):
            continue
        df[col] = df[col] * scale_factor

    print("OK Sensitivity analysis complete!")
    print("=" * 70)
    print()
    return df


def export_pd_sensitivity_results(
    sensitivity_results: pd.DataFrame,
    excel_path: str = 'pd_sensitivity_analysis.xlsx'
) -> None:
    """
    Export periodontal sensitivity analysis results to Excel.

    Sheets:
      - Raw_Results: all replicate-level rows
      - Summary: mean and standard deviation by prevalence, parameter, value_type
    """
    if sensitivity_results is None or sensitivity_results.empty:
        print("No results to export.")
        return

    # Identify numeric metric columns
    protected = {'parameter', 'value_type', 'replicate', 'prevalence'}
    metric_cols = [
        c for c in sensitivity_results.columns
        if c not in protected and pd.api.types.is_numeric_dtype(sensitivity_results[c])
    ]

    summary = (
        sensitivity_results
        .groupby(['prevalence', 'parameter', 'value_type'])[metric_cols]
        .agg(['mean', 'std'])
    )

    Path(excel_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        sensitivity_results.to_excel(writer, sheet_name='Raw_Results', index=False)
        summary.to_excel(writer, sheet_name='Summary')

    print(f"OK Exported sensitivity results to {excel_path}")
