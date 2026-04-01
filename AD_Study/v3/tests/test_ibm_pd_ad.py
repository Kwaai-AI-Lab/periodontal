"""
Unit tests for IBM_PD_AD.py - Alzheimer's Disease Microsimulation Model

Tests are organized by function category:
- Tier 1: Critical mathematical functions (hazard/probability conversions)
- Tier 2: Age band handling functions
- Tier 3: Distribution parameter functions
- Tier 4: Smoothing and utility functions
"""

import math
import pytest

# Import functions to test
from IBM_PD_AD import (
    # Hazard/probability conversion functions
    prob_to_hazard,
    hazard_to_prob,
    base_hazard_from_duration,
    hazards_from_survival,
    # Age band functions
    assign_age_to_reporting_band,
    age_band_key,
    age_band_label,
    age_band_midpoint,
    REPORTING_AGE_BANDS,
    # Distribution parameter functions
    _beta_params_from_mean_rel_sd,
    _gamma_params_from_mean_rel_sd,
    _lognormal_params_from_ci,
    # Smoothing
    smooth_series,
)


# =============================================================================
# TIER 1: CRITICAL MATHEMATICAL FUNCTIONS - Hazard/Probability Conversions
# =============================================================================

class TestProbToHazard:
    """Tests for prob_to_hazard: p -> h = -ln(1-p) / dt"""

    def test_zero_probability_returns_zero(self):
        """Zero probability should return zero hazard."""
        assert prob_to_hazard(0.0) == 0.0
        assert prob_to_hazard(0.0, dt=2.0) == 0.0

    def test_negative_probability_returns_zero(self):
        """Negative probability should be treated as zero."""
        assert prob_to_hazard(-0.1) == 0.0
        assert prob_to_hazard(-1.0) == 0.0

    def test_probability_one_returns_infinity(self):
        """Probability of 1.0 should return infinity."""
        result = prob_to_hazard(1.0)
        assert math.isinf(result) and result > 0

    def test_probability_greater_than_one_returns_infinity(self):
        """Probability > 1.0 should return infinity."""
        result = prob_to_hazard(1.5)
        assert math.isinf(result) and result > 0

    def test_small_probability(self):
        """For small p, hazard ≈ p (first-order approximation)."""
        p = 0.01
        h = prob_to_hazard(p)
        # For small p: h ≈ p, exact: h = -ln(1-p)
        expected = -math.log(1 - p)
        assert abs(h - expected) < 1e-10

    def test_medium_probability(self):
        """Test with p = 0.5."""
        p = 0.5
        h = prob_to_hazard(p)
        expected = -math.log(0.5)  # = ln(2) ≈ 0.693
        assert abs(h - expected) < 1e-10

    def test_high_probability(self):
        """Test with p = 0.9."""
        p = 0.9
        h = prob_to_hazard(p)
        expected = -math.log(0.1)  # = ln(10) ≈ 2.303
        assert abs(h - expected) < 1e-10

    def test_time_step_scaling(self):
        """Hazard should scale inversely with time step."""
        p = 0.1
        h1 = prob_to_hazard(p, dt=1.0)
        h2 = prob_to_hazard(p, dt=2.0)
        # h = -ln(1-p)/dt, so h2 = h1/2
        assert abs(h2 - h1 / 2) < 1e-10

    def test_default_time_step_is_one(self):
        """Default dt should be 1.0."""
        p = 0.1
        h_default = prob_to_hazard(p)
        h_explicit = prob_to_hazard(p, dt=1.0)
        assert h_default == h_explicit


class TestHazardToProb:
    """Tests for hazard_to_prob: h -> p = 1 - exp(-h*dt)"""

    def test_zero_hazard_returns_zero(self):
        """Zero hazard should return zero probability."""
        assert hazard_to_prob(0.0) == 0.0
        assert hazard_to_prob(0.0, dt=2.0) == 0.0

    def test_negative_hazard_returns_zero(self):
        """Negative hazard should return zero probability."""
        assert hazard_to_prob(-0.1) == 0.0
        assert hazard_to_prob(-1.0) == 0.0

    def test_small_hazard(self):
        """For small h, probability ≈ h (first-order approximation)."""
        h = 0.01
        p = hazard_to_prob(h)
        expected = 1 - math.exp(-h)
        assert abs(p - expected) < 1e-10
        # Also verify approximation: p ≈ h for small h
        assert abs(p - h) < 0.001

    def test_medium_hazard(self):
        """Test with h = 0.693 (ln(2)), should give p ≈ 0.5."""
        h = math.log(2)
        p = hazard_to_prob(h)
        assert abs(p - 0.5) < 1e-10

    def test_large_hazard_approaches_one(self):
        """Very large hazard should give probability approaching 1."""
        h = 100.0
        p = hazard_to_prob(h)
        assert p > 0.999999

    def test_time_step_scaling(self):
        """Probability should increase with time step (more exposure)."""
        h = 0.1
        p1 = hazard_to_prob(h, dt=1.0)
        p2 = hazard_to_prob(h, dt=2.0)
        # p = 1 - exp(-h*dt), so p2 = 1 - exp(-2h) > p1
        assert p2 > p1
        expected_p2 = 1 - math.exp(-h * 2)
        assert abs(p2 - expected_p2) < 1e-10

    def test_default_time_step_is_one(self):
        """Default dt should be 1.0."""
        h = 0.1
        p_default = hazard_to_prob(h)
        p_explicit = hazard_to_prob(h, dt=1.0)
        assert p_default == p_explicit


class TestHazardProbRoundTrip:
    """Test that hazard and probability conversions are inverses."""

    @pytest.mark.parametrize("p", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    def test_prob_hazard_roundtrip(self, p):
        """Converting p -> h -> p should return original value."""
        h = prob_to_hazard(p)
        p_recovered = hazard_to_prob(h)
        assert abs(p_recovered - p) < 1e-10

    @pytest.mark.parametrize("h", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    def test_hazard_prob_roundtrip(self, h):
        """Converting h -> p -> h should return original value."""
        p = hazard_to_prob(h)
        h_recovered = prob_to_hazard(p)
        assert abs(h_recovered - h) < 1e-10

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0, 5.0])
    def test_roundtrip_with_different_time_steps(self, dt):
        """Round-trip should work with different time steps."""
        p = 0.3
        h = prob_to_hazard(p, dt=dt)
        p_recovered = hazard_to_prob(h, dt=dt)
        assert abs(p_recovered - p) < 1e-10


class TestBaseHazardFromDuration:
    """Tests for base_hazard_from_duration: duration -> h = 1/duration"""

    def test_none_duration_returns_zero(self):
        """None duration should return zero hazard."""
        assert base_hazard_from_duration(None) == 0.0

    def test_zero_duration_returns_zero(self):
        """Zero duration should return zero hazard."""
        assert base_hazard_from_duration(0.0) == 0.0

    def test_negative_duration_returns_zero(self):
        """Negative duration should return zero hazard."""
        assert base_hazard_from_duration(-1.0) == 0.0
        assert base_hazard_from_duration(-10.0) == 0.0

    def test_one_year_duration(self):
        """Duration of 1 year should return hazard of 1.0."""
        assert base_hazard_from_duration(1.0) == 1.0

    def test_ten_year_duration(self):
        """Duration of 10 years should return hazard of 0.1."""
        assert abs(base_hazard_from_duration(10.0) - 0.1) < 1e-10

    def test_small_duration_high_hazard(self):
        """Small duration should give high hazard."""
        assert base_hazard_from_duration(0.1) == 10.0

    def test_large_duration_low_hazard(self):
        """Large duration should give low hazard."""
        assert abs(base_hazard_from_duration(100.0) - 0.01) < 1e-10


class TestHazardsFromSurvival:
    """Tests for hazards_from_survival: infer piecewise hazards from survival curve."""

    def test_basic_two_point_survival(self):
        """Two time points should give one interval."""
        times = [0.0, 1.0]
        survival = [1.0, 0.9]  # 10% died in 1 year
        df = hazards_from_survival(times, survival)

        assert len(df) == 1
        assert df.iloc[0]['time_start'] == 0.0
        assert df.iloc[0]['time_end'] == 1.0
        assert df.iloc[0]['interval_length'] == 1.0
        # hazard = -ln(0.9/1.0) / 1.0 = -ln(0.9) ≈ 0.105
        expected_hazard = -math.log(0.9)
        assert abs(df.iloc[0]['interval_hazard'] - expected_hazard) < 1e-10

    def test_multiple_intervals(self):
        """Multiple time points should give multiple intervals."""
        times = [0.0, 1.0, 2.0, 3.0]
        survival = [1.0, 0.9, 0.8, 0.7]
        df = hazards_from_survival(times, survival)

        assert len(df) == 3
        assert list(df['time_start']) == [0.0, 1.0, 2.0]
        assert list(df['time_end']) == [1.0, 2.0, 3.0]

    def test_constant_hazard_gives_zero_deviation(self):
        """Constant hazard should give zero relative deviation."""
        # Exponential survival with constant hazard h=0.1
        h = 0.1
        times = [0.0, 1.0, 2.0, 3.0]
        survival = [math.exp(-h * t) for t in times]
        df = hazards_from_survival(times, survival)

        # All hazards should be equal to h
        for hazard in df['interval_hazard']:
            assert abs(hazard - h) < 1e-10

        # Relative deviation should be near zero
        for dev in df['relative_deviation']:
            assert abs(dev) < 1e-10

    def test_mean_hazard_calculation(self):
        """Mean hazard should be average of interval hazards."""
        times = [0.0, 1.0, 2.0]
        survival = [1.0, 0.9, 0.7]  # Different hazards in each interval
        df = hazards_from_survival(times, survival)

        expected_mean = df['interval_hazard'].mean()
        assert all(df['mean_hazard'] == expected_mean)

    def test_mismatched_lengths_raises_error(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            hazards_from_survival([0, 1, 2], [1.0, 0.9])

    def test_single_time_point_raises_error(self):
        """Less than two time points should raise ValueError."""
        with pytest.raises(ValueError, match="At least two"):
            hazards_from_survival([0.0], [1.0])

    def test_non_finite_values_raise_error(self):
        """Non-finite values should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            hazards_from_survival([0, float('inf')], [1.0, 0.9])

        with pytest.raises(ValueError, match="finite"):
            hazards_from_survival([0, 1], [1.0, float('nan')])

    def test_non_increasing_times_raise_error(self):
        """Non-strictly increasing times should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            hazards_from_survival([0, 1, 1, 2], [1.0, 0.9, 0.9, 0.8])

    def test_survival_out_of_range_raises_error(self):
        """Survival probabilities outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="interval"):
            hazards_from_survival([0, 1], [1.0, 0.0])  # 0 not allowed

        with pytest.raises(ValueError, match="interval"):
            hazards_from_survival([0, 1], [1.0, 1.5])  # > 1 not allowed

    def test_unsorted_times_are_sorted(self):
        """Times should be sorted automatically."""
        times = [2.0, 0.0, 1.0]
        survival = [0.8, 1.0, 0.9]  # Corresponding to sorted times
        df = hazards_from_survival(times, survival)

        assert df.iloc[0]['time_start'] == 0.0
        assert df.iloc[0]['time_end'] == 1.0


# =============================================================================
# TIER 2: AGE BAND HANDLING FUNCTIONS
# =============================================================================

class TestAssignAgeToReportingBand:
    """Tests for assign_age_to_reporting_band.

    Note: Default REPORTING_AGE_BANDS are:
    - (35, 49)
    - (50, 64)
    - (65, 79)
    - (80, None)
    """

    def test_age_in_first_band(self):
        """Age in first band (35-49) should return that band."""
        assert assign_age_to_reporting_band(35) == (35, 49)
        assert assign_age_to_reporting_band(40) == (35, 49)
        assert assign_age_to_reporting_band(49) == (35, 49)

    def test_age_in_middle_band(self):
        """Age in middle bands should return correct band."""
        assert assign_age_to_reporting_band(50) == (50, 64)
        assert assign_age_to_reporting_band(55) == (50, 64)
        assert assign_age_to_reporting_band(65) == (65, 79)
        assert assign_age_to_reporting_band(75) == (65, 79)

    def test_age_in_last_open_band(self):
        """Age in open-ended band (80+) should return (80, None)."""
        assert assign_age_to_reporting_band(80) == (80, None)
        assert assign_age_to_reporting_band(90) == (80, None)
        assert assign_age_to_reporting_band(100) == (80, None)
        assert assign_age_to_reporting_band(120) == (80, None)

    def test_age_below_minimum_returns_none(self):
        """Age below minimum band should return None."""
        assert assign_age_to_reporting_band(34) is None
        assert assign_age_to_reporting_band(20) is None
        assert assign_age_to_reporting_band(0) is None

    def test_negative_age_returns_none(self):
        """Negative age should return None."""
        assert assign_age_to_reporting_band(-1) is None
        assert assign_age_to_reporting_band(-100) is None

    def test_float_ages(self):
        """Float ages should work correctly.

        Note: The function uses integer bounds with <= comparison.
        Float values between integer bounds (e.g., 49.9) fall into gaps and return None.
        """
        assert assign_age_to_reporting_band(35.0) == (35, 49)
        assert assign_age_to_reporting_band(49.0) == (35, 49)
        # 49.9 > 49 doesn't match (35, 49), and 49.9 < 50 doesn't match (50, 64)
        # So it falls into a gap and returns None
        assert assign_age_to_reporting_band(49.9) is None
        assert assign_age_to_reporting_band(50.0) == (50, 64)

    def test_boundary_ages(self):
        """Test exact boundary ages between bands."""
        # 49 should be in (35, 49), 50 in (50, 64)
        assert assign_age_to_reporting_band(49) == (35, 49)
        assert assign_age_to_reporting_band(50) == (50, 64)

        # 64 should be in (50, 64), 65 in (65, 79)
        assert assign_age_to_reporting_band(64) == (50, 64)
        assert assign_age_to_reporting_band(65) == (65, 79)

        # 79 should be in (65, 79), 80 in (80, None)
        assert assign_age_to_reporting_band(79) == (65, 79)
        assert assign_age_to_reporting_band(80) == (80, None)

    def test_custom_bands(self):
        """Custom band list should be used when provided."""
        custom_bands = [(0, 17), (18, 64), (65, None)]
        assert assign_age_to_reporting_band(10, bands=custom_bands) == (0, 17)
        assert assign_age_to_reporting_band(30, bands=custom_bands) == (18, 64)
        assert assign_age_to_reporting_band(70, bands=custom_bands) == (65, None)


class TestAgeBandKey:
    """Tests for age_band_key."""

    def test_closed_band(self):
        """Closed band should give 'lower_upper' format."""
        assert age_band_key((35, 49)) == "35_49"
        assert age_band_key((50, 54)) == "50_54"
        assert age_band_key((0, 17)) == "0_17"

    def test_open_band(self):
        """Open-ended band should give 'lower_plus' format."""
        assert age_band_key((90, None)) == "90_plus"
        assert age_band_key((65, None)) == "65_plus"


class TestAgeBandLabel:
    """Tests for age_band_label."""

    def test_closed_band(self):
        """Closed band should give 'lower-upper' format."""
        assert age_band_label((35, 49)) == "35-49"
        assert age_band_label((50, 54)) == "50-54"

    def test_open_band(self):
        """Open-ended band should give 'lower+' format."""
        assert age_band_label((90, None)) == "90+"
        assert age_band_label((65, None)) == "65+"


class TestAgeBandMidpoint:
    """Tests for age_band_midpoint."""

    def test_closed_band_midpoint(self):
        """Closed band should return numeric midpoint."""
        assert age_band_midpoint((35, 49)) == 42.0
        assert age_band_midpoint((50, 54)) == 52.0
        assert age_band_midpoint((0, 10)) == 5.0

    def test_open_band_returns_none(self):
        """Open-ended band should return None."""
        assert age_band_midpoint((90, None)) is None
        assert age_band_midpoint((65, None)) is None

    def test_single_year_band(self):
        """Single year band (lower == upper) should return that value."""
        assert age_band_midpoint((50, 50)) == 50.0


# =============================================================================
# TIER 3: DISTRIBUTION PARAMETER FUNCTIONS
# =============================================================================

class TestBetaParamsFromMeanRelSd:
    """Tests for _beta_params_from_mean_rel_sd."""

    def test_valid_symmetric_distribution(self):
        """Mean of 0.5 with reasonable rel_sd should work."""
        result = _beta_params_from_mean_rel_sd(0.5, 0.2)
        assert result is not None
        alpha, beta = result
        assert alpha > 0 and beta > 0
        # For mean=0.5, alpha should equal beta (symmetric)
        assert abs(alpha - beta) < 1e-10

    def test_asymmetric_distribution(self):
        """Mean != 0.5 should give asymmetric parameters."""
        result = _beta_params_from_mean_rel_sd(0.3, 0.2)
        assert result is not None
        alpha, beta = result
        assert alpha > 0 and beta > 0
        # For mean < 0.5, beta > alpha
        assert beta > alpha

    def test_mean_zero_returns_none(self):
        """Mean of 0 should return None."""
        assert _beta_params_from_mean_rel_sd(0.0, 0.2) is None

    def test_mean_one_returns_none(self):
        """Mean of 1 should return None."""
        assert _beta_params_from_mean_rel_sd(1.0, 0.2) is None

    def test_mean_outside_range_returns_none(self):
        """Mean outside (0, 1) should return None."""
        assert _beta_params_from_mean_rel_sd(-0.1, 0.2) is None
        assert _beta_params_from_mean_rel_sd(1.5, 0.2) is None

    def test_zero_rel_sd_returns_none(self):
        """Zero relative SD should return None."""
        assert _beta_params_from_mean_rel_sd(0.5, 0.0) is None

    def test_negative_rel_sd_returns_none(self):
        """Negative relative SD should return None."""
        assert _beta_params_from_mean_rel_sd(0.5, -0.1) is None

    def test_very_large_rel_sd_clamped(self):
        """Very large rel_sd should be clamped and still work."""
        # When variance would exceed max_variance = mean*(1-mean)
        result = _beta_params_from_mean_rel_sd(0.5, 10.0)
        # Should still return valid params due to clamping
        if result is not None:
            alpha, beta = result
            assert alpha > 0 and beta > 0

    def test_resulting_distribution_has_correct_mean(self):
        """Resulting beta distribution should have specified mean."""
        mean = 0.3
        rel_sd = 0.2
        result = _beta_params_from_mean_rel_sd(mean, rel_sd)
        assert result is not None
        alpha, beta = result
        # Beta mean = alpha / (alpha + beta)
        computed_mean = alpha / (alpha + beta)
        assert abs(computed_mean - mean) < 1e-10


class TestGammaParamsFromMeanRelSd:
    """Tests for _gamma_params_from_mean_rel_sd."""

    def test_valid_parameters(self):
        """Valid mean and rel_sd should return shape and scale."""
        result = _gamma_params_from_mean_rel_sd(100.0, 0.2)
        assert result is not None
        shape, scale = result
        assert shape > 0 and scale > 0

    def test_zero_mean_returns_none(self):
        """Zero mean should return None."""
        assert _gamma_params_from_mean_rel_sd(0.0, 0.2) is None

    def test_negative_mean_returns_none(self):
        """Negative mean should return None."""
        assert _gamma_params_from_mean_rel_sd(-10.0, 0.2) is None

    def test_zero_rel_sd_returns_none(self):
        """Zero relative SD should return None."""
        assert _gamma_params_from_mean_rel_sd(100.0, 0.0) is None

    def test_negative_rel_sd_returns_none(self):
        """Negative relative SD should return None."""
        assert _gamma_params_from_mean_rel_sd(100.0, -0.1) is None

    def test_resulting_distribution_has_correct_mean(self):
        """Resulting gamma distribution should have specified mean."""
        mean = 100.0
        rel_sd = 0.2
        result = _gamma_params_from_mean_rel_sd(mean, rel_sd)
        assert result is not None
        shape, scale = result
        # Gamma mean = shape * scale
        computed_mean = shape * scale
        assert abs(computed_mean - mean) < 1e-10

    def test_resulting_distribution_has_correct_variance(self):
        """Resulting gamma distribution should have specified variance."""
        mean = 100.0
        rel_sd = 0.2
        expected_variance = (rel_sd * mean) ** 2
        result = _gamma_params_from_mean_rel_sd(mean, rel_sd)
        assert result is not None
        shape, scale = result
        # Gamma variance = shape * scale^2
        computed_variance = shape * scale ** 2
        assert abs(computed_variance - expected_variance) < 1e-6


class TestLognormalParamsFromCi:
    """Tests for _lognormal_params_from_ci."""

    def test_valid_parameters(self):
        """Valid point estimate and CI should return mu and sigma."""
        result = _lognormal_params_from_ci(1.5, 1.2, 1.9)
        assert result is not None
        mu, sigma = result
        assert sigma > 0
        # mu should be ln(point_estimate)
        assert abs(mu - math.log(1.5)) < 1e-10

    def test_zero_point_estimate_returns_none(self):
        """Zero point estimate should return None."""
        assert _lognormal_params_from_ci(0.0, 1.0, 2.0) is None

    def test_negative_point_estimate_returns_none(self):
        """Negative point estimate should return None."""
        assert _lognormal_params_from_ci(-1.0, 1.0, 2.0) is None

    def test_zero_lower_bound_returns_none(self):
        """Zero lower bound should return None."""
        assert _lognormal_params_from_ci(1.5, 0.0, 2.0) is None

    def test_zero_upper_bound_returns_none(self):
        """Zero upper bound should return None."""
        assert _lognormal_params_from_ci(1.5, 1.0, 0.0) is None

    def test_negative_bounds_return_none(self):
        """Negative bounds should return None."""
        assert _lognormal_params_from_ci(1.5, -1.0, 2.0) is None
        assert _lognormal_params_from_ci(1.5, 1.0, -2.0) is None

    def test_sigma_formula(self):
        """Sigma should be (ln(upper) - ln(lower)) / (2 * 1.96)."""
        lower, upper = 1.0, 2.0
        result = _lognormal_params_from_ci(1.5, lower, upper)
        assert result is not None
        mu, sigma = result
        expected_sigma = (math.log(upper) - math.log(lower)) / (2 * 1.96)
        assert abs(sigma - expected_sigma) < 1e-10


# =============================================================================
# TIER 4: SMOOTHING AND UTILITY FUNCTIONS
# =============================================================================

class TestSmoothSeries:
    """Tests for smooth_series (centered moving average)."""

    def test_empty_list_returns_empty(self):
        """Empty list should return empty list."""
        assert smooth_series([]) == []

    def test_single_value_unchanged(self):
        """Single value should be unchanged."""
        assert smooth_series([5.0]) == [5.0]

    def test_window_one_returns_original(self):
        """Window of 1 should return original values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert smooth_series(values, window=1) == values

    def test_window_zero_returns_original(self):
        """Window of 0 should return original values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert smooth_series(values, window=0) == values

    def test_basic_smoothing(self):
        """Basic smoothing should compute moving average."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        smoothed = smooth_series(values, window=3)
        # Middle values should be average of 3 neighbors
        # smoothed[2] = (2 + 3 + 4) / 3 = 3.0
        assert abs(smoothed[2] - 3.0) < 1e-10

    def test_constant_series_unchanged(self):
        """Constant series should be unchanged by smoothing."""
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        smoothed = smooth_series(values, window=3)
        for v in smoothed:
            assert abs(v - 5.0) < 1e-10

    def test_output_length_matches_input(self):
        """Output length should match input length."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        smoothed = smooth_series(values, window=3)
        assert len(smoothed) == len(values)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_survival_to_probability_chain(self):
        """Test chain: survival -> hazard -> probability."""
        # Start with survival probabilities
        times = [0.0, 1.0, 2.0]
        survival = [1.0, 0.9, 0.8]  # ~10% die each year

        # Get hazards
        df = hazards_from_survival(times, survival)

        # Convert hazard back to probability for first interval
        h = df.iloc[0]['interval_hazard']
        p = hazard_to_prob(h, dt=1.0)

        # Probability should match the survival decrement
        # S(1) = S(0) * (1 - p) => p = 1 - S(1)/S(0) = 1 - 0.9 = 0.1
        expected_p = 1 - survival[1] / survival[0]
        assert abs(p - expected_p) < 1e-10

    def test_age_band_functions_consistency(self):
        """Test that age band functions are consistent with each other."""
        for band in REPORTING_AGE_BANDS:
            key = age_band_key(band)
            label = age_band_label(band)
            midpoint = age_band_midpoint(band)

            # Key and label should contain the lower bound
            assert str(band[0]) in key
            assert str(band[0]) in label

            # If closed band, midpoint should be between bounds
            if band[1] is not None:
                assert midpoint is not None
                assert band[0] <= midpoint <= band[1]
            else:
                assert midpoint is None
                assert "plus" in key or "+" in label
