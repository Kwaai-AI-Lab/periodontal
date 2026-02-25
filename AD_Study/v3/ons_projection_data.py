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
BASELINE_AGE_DISTRIBUTION_2023 = {
    (65, 69): 0.12,
    (70, 74): 0.11,
    (75, 79): 0.23,
    (80, 84): 0.15,
    (85, 100): 0.39,
}

# ONS projection multipliers by year
# These scale the baseline distribution to project future age structures
ONS_AGE_BAND_MULTIPLIER_SCHEDULE = {
    2025: {
        (65, 69): 1.05,
        (70, 74): 0.99,
        (75, 79): 1.03,
        (80, 84): 1.09,
        (85, 100): 1.04,
    },
    2030: {
        (65, 69): 1.21,
        (70, 74): 1.09,
        (75, 79): 0.95,
        (80, 84): 1.37,
        (85, 100): 1.21,
    },
    2035: {
        (65, 69): 1.23,
        (70, 74): 1.26,
        (75, 79): 1.06,
        (80, 84): 1.28,
        (85, 100): 1.52,
    },
    2040: {
        (65, 69): 1.16,
        (70, 74): 1.29,
        (75, 79): 1.23,
        (80, 84): 1.44,
        (85, 100): 1.62,
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
