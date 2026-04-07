"""
ONS Population Projection Data for England 65+ Population

Source: Office for National Statistics (ONS)
Data represents projected changes in age distribution relative to 2023 baseline.

This data is used for validation purposes to check if the model's aging + mortality
dynamics correctly reproduce ONS-projected age distributions.

NOTE: These multipliers are applied to the 2023 baseline age distribution to get
projected distributions for future years. They are NOT used for entrant age sampling
(all entrants now correctly enter at age 65).
"""

# Baseline age distribution (2023)
# These represent the proportion of the 65+ population in each age band
# Source: ONS 2023-based population projections for England
BASELINE_AGE_DISTRIBUTION_2023 = {
    (65, 69): 0.267405,   # 26.74%
    (70, 74): 0.240115,   # 24.01%
    (75, 79): 0.221414,   # 22.14%
    (80, 84): 0.137043,   # 13.70%
    (85, 100): 0.134023,  # 13.40%
}

# ONS projection multipliers by year
# These scale the baseline distribution to project future age structures
# Calculated from ONS 2023-based population projections for England
ONS_AGE_BAND_MULTIPLIER_SCHEDULE = {
    2025: {
        (65, 69): 1.017831,   # → 27.22%
        (70, 74): 0.953498,   # → 22.89%
        (75, 79): 0.996295,   # → 22.06%
        (80, 84): 1.050225,   # → 14.39%
        (85, 100): 1.002499,  # → 13.44%
    },
    2030: {
        (65, 69): 1.054718,   # → 28.20%
        (70, 74): 0.953422,   # → 22.89%
        (75, 79): 0.831688,   # → 18.41%
        (80, 84): 1.195089,   # → 16.38%
        (85, 100): 1.052850,  # → 14.11%
    },
    2035: {
        (65, 69): 0.990478,   # → 26.49%
        (70, 74): 1.010204,   # → 24.26%
        (75, 79): 0.853345,   # → 18.89%
        (80, 84): 1.024041,   # → 14.03%
        (85, 100): 1.218417,  # → 16.33%
    },
    2040: {
        (65, 69): 0.886772,   # → 23.71%
        (70, 74): 0.986045,   # → 23.68%
        (75, 79): 0.942767,   # → 20.87%
        (80, 84): 1.101349,   # → 15.09%
        (85, 100): 1.241837,  # → 16.64%
    },
}


def get_ons_projected_distribution(year):
    """
    Get ONS projected age distribution for a specific year.

    Parameters:
    -----------
    year : int
        Target year (must be 2023 or in ONS_AGE_BAND_MULTIPLIER_SCHEDULE)

    Returns:
    --------
    dict : Age band -> proportion mapping
    """
    if year == 2023:
        return BASELINE_AGE_DISTRIBUTION_2023.copy()

    if year not in ONS_AGE_BAND_MULTIPLIER_SCHEDULE:
        raise ValueError(f"No ONS projection data for year {year}. "
                        f"Available years: 2023, {', '.join(map(str, sorted(ONS_AGE_BAND_MULTIPLIER_SCHEDULE.keys())))}")

    multipliers = ONS_AGE_BAND_MULTIPLIER_SCHEDULE[year]
    projected = {}

    for age_band, baseline_prop in BASELINE_AGE_DISTRIBUTION_2023.items():
        multiplier = multipliers.get(age_band, 1.0)
        projected[age_band] = baseline_prop * multiplier

    # Normalize to sum to 1.0
    total = sum(projected.values())
    if total > 0:
        projected = {k: v / total for k, v in projected.items()}

    return projected


if __name__ == "__main__":
    # Print projected distributions for all years
    print("ONS Projected Age Distributions (England 65+ Population)")
    print("=" * 70)

    all_years = [2023] + sorted(ONS_AGE_BAND_MULTIPLIER_SCHEDULE.keys())

    for year in all_years:
        print(f"\n{year}:")
        distribution = get_ons_projected_distribution(year)
        for (age_lower, age_upper), proportion in sorted(distribution.items()):
            age_band = f"{age_lower}-{age_upper if age_upper != 100 else '+'}"
            print(f"  {age_band:8s}: {proportion*100:5.2f}%")
