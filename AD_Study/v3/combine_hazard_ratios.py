"""
Combine Hazard Ratios from the Same Study

This script combines multiple hazard ratios (with 95% CIs) from the same study
using fixed-effect meta-analysis methods.

References:
-----------
1. Tierney JF, et al. (2007). Practical methods for incorporating summary time-to-event
   data into meta-analysis. Trials, 8:16. DOI: 10.1186/1745-6215-8-16

2. Parmar MK, et al. (1998). Extracting summary statistics to perform meta-analyses
   of the published literature for survival endpoints. Statistics in Medicine, 17(24):2815-34.
   DOI: 10.1002/(SICI)1097-0258(19981230)17:24<2815::AID-SIM110>3.0.CO;2-8

3. Cochrane Handbook for Systematic Reviews of Interventions (2023). Chapter 6:
   Choosing effect measures and computing estimates of effect.

4. Borenstein M, et al. (2009). Introduction to Meta-Analysis. Wiley.
   (Chapter 4: Effect Sizes Based on Means; applicable principles for HRs)

5. Deeks JJ, et al. (2001). Statistical methods for examining heterogeneity and
   combining results from several studies in meta-analysis. In: Egger M, et al.,
   Systematic Reviews in Health Care: Meta-Analysis in Context, 2nd Edition.

Methods:
--------
- Hazard ratios are transformed to log scale: ln(HR)
- Standard errors are calculated from 95% CIs: SE = [ln(UCI) - ln(LCI)] / 3.92
- Inverse-variance weighted fixed-effect meta-analysis is used
- Combined HR = exp(Sum[w_i * ln(HR_i)] / Sum[w_i]), where w_i = 1/SE_i^2
- Combined 95% CI uses SE_combined = 1/sqrt(Sum[w_i])
"""

import numpy as np
from typing import List, Tuple, Dict
import warnings


def combine_hazard_ratios(
    hrs: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    labels: List[str] = None,
    method: str = 'fixed'
) -> Dict[str, float]:
    """
    Combine multiple hazard ratios from the same study using meta-analysis.

    Parameters:
    -----------
    hrs : List[float]
        List of hazard ratios to combine
    ci_lower : List[float]
        List of lower bounds of 95% confidence intervals
    ci_upper : List[float]
        List of upper bounds of 95% confidence intervals
    labels : List[str], optional
        Labels for each HR (for display purposes)
    method : str, default='fixed'
        Meta-analysis method: 'fixed' for fixed-effect

    Returns:
    --------
    Dict containing:
        - 'combined_hr': Combined hazard ratio
        - 'ci_lower': Lower bound of 95% CI
        - 'ci_upper': Upper bound of 95% CI
        - 'se': Standard error of log(HR)
        - 'individual_results': List of dicts with individual calculations

    Notes:
    ------
    - Uses inverse-variance weighting (Cochrane Handbook, Section 10.3)
    - Assumes HRs are from same study (homogeneous population)
    - For combining across studies, consider random-effects methods
    """

    if not (len(hrs) == len(ci_lower) == len(ci_upper)):
        raise ValueError("All input lists must have the same length")

    if len(hrs) == 0:
        raise ValueError("At least one HR must be provided")

    if labels is None:
        labels = [f"HR_{i+1}" for i in range(len(hrs))]

    # Convert to numpy arrays
    hrs = np.array(hrs)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # Check for invalid values
    if np.any(hrs <= 0) or np.any(ci_lower <= 0) or np.any(ci_upper <= 0):
        raise ValueError("All HRs and CIs must be positive")

    if np.any(ci_lower >= hrs) or np.any(ci_upper <= hrs):
        warnings.warn("Some CIs may not contain the HR point estimate")

    # Transform to log scale
    log_hrs = np.log(hrs)
    log_ci_lower = np.log(ci_lower)
    log_ci_upper = np.log(ci_upper)

    # Calculate standard errors from 95% CIs
    # SE = (ln(UCI) - ln(LCI)) / 3.92
    # 3.92 = 2 × 1.96 (for 95% CI)
    # Reference: Cochrane Handbook Section 6.3.1
    se_log_hrs = (log_ci_upper - log_ci_lower) / 3.92

    # Calculate inverse-variance weights
    # w_i = 1 / SE_i²
    variances = se_log_hrs ** 2
    weights = 1.0 / variances

    # Fixed-effect meta-analysis
    # Combined log(HR) = Sum(w_i * log(HR_i)) / Sum(w_i)
    total_weight = np.sum(weights)
    combined_log_hr = np.sum(weights * log_hrs) / total_weight

    # Standard error of combined estimate
    # SE_combined = 1 / sqrt(Sum(w_i))
    se_combined = 1.0 / np.sqrt(total_weight)

    # 95% CI for combined log(HR)
    z_95 = 1.96  # 95% CI critical value
    combined_log_ci_lower = combined_log_hr - z_95 * se_combined
    combined_log_ci_upper = combined_log_hr + z_95 * se_combined

    # Transform back to HR scale
    combined_hr = np.exp(combined_log_hr)
    combined_ci_lower = np.exp(combined_log_ci_lower)
    combined_ci_upper = np.exp(combined_log_ci_upper)

    # Prepare individual results
    individual_results = []
    for i, label in enumerate(labels):
        individual_results.append({
            'label': label,
            'hr': hrs[i],
            'ci_lower': ci_lower[i],
            'ci_upper': ci_upper[i],
            'log_hr': log_hrs[i],
            'se': se_log_hrs[i],
            'weight': weights[i],
            'weight_pct': 100 * weights[i] / total_weight
        })

    return {
        'combined_hr': combined_hr,
        'ci_lower': combined_ci_lower,
        'ci_upper': combined_ci_upper,
        'se': se_combined,
        'individual_results': individual_results
    }


def print_results(results: Dict, title: str = "Combined Hazard Ratio Results"):
    """
    Pretty-print the results of combining hazard ratios.

    Parameters:
    -----------
    results : Dict
        Output from combine_hazard_ratios()
    title : str
        Title for the output
    """
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    print()

    print("Individual Hazard Ratios:")
    print("-" * 80)
    print(f"{'Label':<20} {'HR':>8} {'95% CI':>20} {'Weight':>10} {'% Weight':>10}")
    print("-" * 80)

    for ind in results['individual_results']:
        ci_str = f"({ind['ci_lower']:.3f}, {ind['ci_upper']:.3f})"
        print(f"{ind['label']:<20} {ind['hr']:>8.3f} {ci_str:>20} "
              f"{ind['weight']:>10.3f} {ind['weight_pct']:>9.1f}%")

    print("-" * 80)
    print()

    print("Combined Result (Fixed-Effect Meta-Analysis):")
    print("-" * 80)
    ci_str = f"({results['ci_lower']:.3f}, {results['ci_upper']:.3f})"
    print(f"Combined HR: {results['combined_hr']:.3f}")
    print(f"95% CI:      {ci_str}")
    print(f"SE(log HR):  {results['se']:.4f}")
    print("=" * 80)
    print()


# Example usage
if __name__ == "__main__":
    print(__doc__)
    print()

    # ============================================================================
    # YOUR ACTUAL DATA: Lifestyle Factors
    # ============================================================================
    print("\n" + "="*80)
    print("YOUR DATA: Combining Lifestyle Factor Hazard Ratios")
    print("="*80)
    print()

    # Enter your actual HRs and 95% CIs here:
    hrs_actual = [2.2, 1.9]
    ci_lower_actual = [1.97, 1.74]
    ci_upper_actual = [2.45, 2.07]
    labels_actual = ['Low income', 'Low socioeconomic status']

    results_actual = combine_hazard_ratios(
        hrs_actual,
        ci_lower_actual,
        ci_upper_actual,
        labels_actual
    )

    print_results(results_actual, "Combined Lifestyle Factors Hazard Ratio")
