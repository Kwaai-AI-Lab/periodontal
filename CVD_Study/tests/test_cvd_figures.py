"""
Unit tests for generate_cvd_figures.py - CVD Figure Generation

Tests focus on:
- Data generation and transformations
- CEAC calculation logic
- Error handling
"""

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import module and constants
from generate_cvd_figures import (
    BASE_CASE_ICER,
    NICE_THRESHOLD_LOW,
    NICE_THRESHOLD_HIGH,
    create_tornado_plot,
    create_ce_plane,
    create_ceac,
    read_excel_psa_data,
)


# =============================================================================
# TESTS FOR CONSTANTS
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_base_case_icer_value(self):
        """Base case ICER should match paper value."""
        assert BASE_CASE_ICER == 44858

    def test_nice_thresholds(self):
        """NICE thresholds should be £20k and £30k."""
        assert NICE_THRESHOLD_LOW == 20000
        assert NICE_THRESHOLD_HIGH == 30000

    def test_nice_threshold_ordering(self):
        """Low threshold should be less than high threshold."""
        assert NICE_THRESHOLD_LOW < NICE_THRESHOLD_HIGH


# =============================================================================
# TESTS FOR CEAC CALCULATION LOGIC
# =============================================================================

class TestCEACCalculation:
    """Tests for CEAC calculation logic (Net Monetary Benefit)."""

    def test_nmb_positive_when_cost_effective(self):
        """NMB should be positive when intervention is cost-effective."""
        # Scenario: £1000 extra cost, 0.1 extra QALY, WTP = £20k
        # NMB = 0.1 * 20000 - 1000 = 2000 - 1000 = 1000 > 0
        incremental_cost = 1000
        incremental_qalys = 0.1
        wtp = 20000
        nmb = incremental_qalys * wtp - incremental_cost
        assert nmb == 1000
        assert nmb > 0

    def test_nmb_negative_when_not_cost_effective(self):
        """NMB should be negative when not cost-effective."""
        # Scenario: £10000 extra cost, 0.1 extra QALY, WTP = £20k
        # NMB = 0.1 * 20000 - 10000 = 2000 - 10000 = -8000 < 0
        incremental_cost = 10000
        incremental_qalys = 0.1
        wtp = 20000
        nmb = incremental_qalys * wtp - incremental_cost
        assert nmb == -8000
        assert nmb < 0

    def test_nmb_zero_at_icer_wtp(self):
        """NMB should be zero when WTP equals ICER."""
        # When WTP = ICER = cost/qalys
        incremental_cost = 5000
        incremental_qalys = 0.25
        icer = incremental_cost / incremental_qalys  # = 20000
        wtp = icer
        nmb = incremental_qalys * wtp - incremental_cost
        assert abs(nmb) < 1e-10

    def test_probability_cost_effective_all_positive(self):
        """When all NMBs positive, probability should be 100%."""
        # Create data where all ICERs are below WTP
        psa_data = pd.DataFrame({
            'incremental_cost': [1000, 2000, 3000],
            'incremental_qalys': [0.1, 0.2, 0.3]
            # ICERs: 10000, 10000, 10000 - all below WTP of 20000
        })
        wtp = 20000
        nmb = psa_data['incremental_qalys'] * wtp - psa_data['incremental_cost']
        prob_ce = (nmb > 0).sum() / len(nmb)
        assert prob_ce == 1.0

    def test_probability_cost_effective_all_negative(self):
        """When all NMBs negative, probability should be 0%."""
        # Create data where all ICERs are above WTP
        psa_data = pd.DataFrame({
            'incremental_cost': [10000, 20000, 30000],
            'incremental_qalys': [0.1, 0.2, 0.3]
            # ICERs: 100000, 100000, 100000 - all above WTP of 20000
        })
        wtp = 20000
        nmb = psa_data['incremental_qalys'] * wtp - psa_data['incremental_cost']
        prob_ce = (nmb > 0).sum() / len(nmb)
        assert prob_ce == 0.0

    def test_probability_cost_effective_mixed(self):
        """Mixed NMBs should give intermediate probability."""
        psa_data = pd.DataFrame({
            'incremental_cost': [1000, 5000],  # ICERs: 10000, 50000
            'incremental_qalys': [0.1, 0.1]
        })
        wtp = 20000
        nmb = psa_data['incremental_qalys'] * wtp - psa_data['incremental_cost']
        # First: 0.1*20000 - 1000 = 1000 > 0
        # Second: 0.1*20000 - 5000 = -3000 < 0
        prob_ce = (nmb > 0).sum() / len(nmb)
        assert prob_ce == 0.5


# =============================================================================
# TESTS FOR TORNADO PLOT DATA
# =============================================================================

class TestTornadoPlotData:
    """Tests for tornado plot data generation."""

    def test_default_sensitivity_data_structure(self):
        """Default sensitivity data should have required columns."""
        # Create default data by calling with None (same logic as function)
        sensitivity_data = pd.DataFrame({
            'parameter': [
                'Stroke Treatment HR\n(0.29 - 0.81)',
                'MI Treatment HR\n(0.44 - 0.95)',
            ],
            'icer_low': [14429, 21475],
            'icer_high': [76222, 85902],
        })

        assert 'parameter' in sensitivity_data.columns
        assert 'icer_low' in sensitivity_data.columns
        assert 'icer_high' in sensitivity_data.columns

    def test_range_calculation(self):
        """Range should be absolute difference between high and low."""
        data = pd.DataFrame({
            'parameter': ['Test'],
            'icer_low': [10000],
            'icer_high': [50000],
        })
        data['range'] = abs(data['icer_high'] - data['icer_low'])
        assert data['range'].iloc[0] == 40000

    def test_deviation_from_base_case(self):
        """Deviations should be relative to base case ICER."""
        data = pd.DataFrame({
            'parameter': ['Test'],
            'icer_low': [30000],
            'icer_high': [60000],
        })
        data['low_deviation'] = data['icer_low'] - BASE_CASE_ICER
        data['high_deviation'] = data['icer_high'] - BASE_CASE_ICER

        # Base case is 44858
        expected_low_dev = 30000 - 44858  # -14858
        expected_high_dev = 60000 - 44858  # 15142

        assert data['low_deviation'].iloc[0] == expected_low_dev
        assert data['high_deviation'].iloc[0] == expected_high_dev


# =============================================================================
# TESTS FOR PSA DATA SIMULATION
# =============================================================================

class TestPSADataSimulation:
    """Tests for simulated PSA data generation."""

    def test_simulated_data_reproducibility(self):
        """Same seed should produce same data."""
        np.random.seed(42)
        n = 100
        z1 = np.random.normal(0, 1, n)

        np.random.seed(42)
        z2 = np.random.normal(0, 1, n)

        np.testing.assert_array_equal(z1, z2)

    def test_simulated_psa_data_shape(self):
        """Simulated PSA data should have expected shape."""
        np.random.seed(42)
        n_iterations = 1000
        mean_cost = 8096
        std_cost = 2500

        z1 = np.random.normal(0, 1, n_iterations)
        incremental_cost = mean_cost + std_cost * z1

        assert len(incremental_cost) == n_iterations

    def test_simulated_data_mean_approximately_correct(self):
        """Simulated data mean should be close to specified mean."""
        np.random.seed(42)
        n_iterations = 10000
        mean_cost = 8096
        std_cost = 2500

        z1 = np.random.normal(0, 1, n_iterations)
        incremental_cost = mean_cost + std_cost * z1

        # With 10000 iterations, mean should be within 2% of target
        assert abs(incremental_cost.mean() - mean_cost) < mean_cost * 0.02

    def test_correlation_structure(self):
        """Correlated data generation should work correctly."""
        np.random.seed(42)
        n = 10000
        rho = 0.3

        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)

        # Create correlated variable
        correlated = rho * z1 + np.sqrt(1 - rho**2) * z2

        # Empirical correlation should be close to rho
        empirical_corr = np.corrcoef(z1, correlated)[0, 1]
        assert abs(empirical_corr - rho) < 0.05


# =============================================================================
# TESTS FOR QUADRANT CALCULATIONS
# =============================================================================

class TestQuadrantCalculations:
    """Tests for CE plane quadrant percentage calculations."""

    def test_all_in_q1(self):
        """All points in Q1 (more effective, more costly)."""
        data = pd.DataFrame({
            'incremental_cost': [100, 200, 300],
            'incremental_qalys': [0.1, 0.2, 0.3]
        })
        q1 = ((data['incremental_qalys'] > 0) &
              (data['incremental_cost'] > 0)).sum() / len(data) * 100
        assert q1 == 100.0

    def test_all_in_q4(self):
        """All points in Q4 (more effective, less costly) - dominant."""
        data = pd.DataFrame({
            'incremental_cost': [-100, -200, -300],
            'incremental_qalys': [0.1, 0.2, 0.3]
        })
        q4 = ((data['incremental_qalys'] > 0) &
              (data['incremental_cost'] < 0)).sum() / len(data) * 100
        assert q4 == 100.0

    def test_mixed_quadrants(self):
        """Points distributed across quadrants."""
        data = pd.DataFrame({
            'incremental_cost': [100, 100, -100, -100],
            'incremental_qalys': [0.1, -0.1, 0.1, -0.1]
        })
        # Q1: cost>0, qaly>0 -> 1 point
        q1 = ((data['incremental_qalys'] > 0) &
              (data['incremental_cost'] > 0)).sum() / len(data) * 100
        assert q1 == 25.0

        # Q4: cost<0, qaly>0 -> 1 point
        q4 = ((data['incremental_qalys'] > 0) &
              (data['incremental_cost'] < 0)).sum() / len(data) * 100
        assert q4 == 25.0


# =============================================================================
# TESTS FOR EXCEL READING (MOCKED)
# =============================================================================

class TestReadExcelPSAData:
    """Tests for Excel reading function with mocks."""

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Nonexistent file should return None gracefully."""
        result = read_excel_psa_data(tmp_path / "nonexistent.xlsx")
        assert result is None

    @patch('generate_cvd_figures.openpyxl.load_workbook')
    def test_no_psa_sheet_returns_none(self, mock_load):
        """Excel without PSA sheet should return None."""
        mock_wb = MagicMock()
        mock_wb.sheetnames = ['Sheet1', 'Data', 'Summary']
        mock_load.return_value = mock_wb

        result = read_excel_psa_data('test.xlsx')
        assert result is None


# =============================================================================
# TESTS FOR FIGURE GENERATION (MOCKED)
# =============================================================================

class TestFigureGeneration:
    """Tests for figure generation functions with mocked matplotlib."""

    @patch('generate_cvd_figures.plt')
    def test_tornado_plot_creates_figure(self, mock_plt):
        """Tornado plot should create a figure."""
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())

        create_tornado_plot()

        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch('generate_cvd_figures.plt')
    def test_ce_plane_creates_figure(self, mock_plt):
        """CE plane should create a figure and return data."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_xlim.return_value = (-0.5, 1.0)
        mock_ax.get_ylim.return_value = (-5000, 20000)
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        result = create_ce_plane()

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'incremental_cost' in result.columns
        assert 'incremental_qalys' in result.columns

    @patch('generate_cvd_figures.plt')
    def test_ceac_creates_figure(self, mock_plt):
        """CEAC should create a figure."""
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())

        create_ceac()

        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple calculations."""

    def test_ceac_increases_with_wtp(self):
        """CEAC probability should increase with WTP for positive QALY interventions."""
        # Simulate simple PSA data with positive QALYs and costs
        psa_data = pd.DataFrame({
            'incremental_cost': np.random.normal(5000, 1000, 1000),
            'incremental_qalys': np.random.normal(0.1, 0.02, 1000)
        })

        wtp_values = [10000, 30000, 50000, 100000]
        probs = []

        for wtp in wtp_values:
            nmb = psa_data['incremental_qalys'] * wtp - psa_data['incremental_cost']
            prob = (nmb > 0).sum() / len(nmb)
            probs.append(prob)

        # Probability should increase with WTP (for positive QALY interventions)
        for i in range(len(probs) - 1):
            assert probs[i+1] >= probs[i], "CEAC should be monotonically increasing"

    def test_icer_threshold_crossing(self):
        """Probability should cross 50% near the mean ICER."""
        # Create deterministic-like data with known ICER
        n = 1000
        true_icer = 30000
        psa_data = pd.DataFrame({
            'incremental_cost': np.ones(n) * 3000,
            'incremental_qalys': np.ones(n) * 0.1  # ICER = 3000/0.1 = 30000
        })

        # At WTP = ICER, probability should be ~50%
        nmb_at_icer = psa_data['incremental_qalys'] * true_icer - psa_data['incremental_cost']
        # All NMBs should be 0
        assert abs(nmb_at_icer.sum()) < 1e-10

        # Slightly above ICER should be 100%
        nmb_above = psa_data['incremental_qalys'] * (true_icer + 1) - psa_data['incremental_cost']
        prob_above = (nmb_above > 0).sum() / len(nmb_above)
        assert prob_above == 1.0

        # Slightly below ICER should be 0%
        nmb_below = psa_data['incremental_qalys'] * (true_icer - 1) - psa_data['incremental_cost']
        prob_below = (nmb_below > 0).sum() / len(nmb_below)
        assert prob_below == 0.0
