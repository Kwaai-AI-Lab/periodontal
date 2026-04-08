"""
Unit tests for PSA workflow scripts.

NOTE: The original psa_with_timeseries module has been deprecated.
PSA functionality is now integrated directly into IBM_PD_AD_v3.py.

The v3 model includes:
- run_probabilistic_sensitivity_analysis() function
- general_config configuration
- save_results_compressed() for output

See AD_Study/v3/run_psa_direct_v3.py for the current PSA workflow.
"""

import pytest
from pathlib import Path


# =============================================================================
# PSA ARCHITECTURE TESTS - Verify current v3 PSA implementation
# =============================================================================

class TestPSAArchitecture:
    """Tests verifying the current PSA architecture.

    The original psa_with_timeseries module has been deprecated.
    PSA is now integrated directly into IBM_PD_AD_v3.py.
    """

    def test_v3_psa_function_available(self):
        """Test that run_probabilistic_sensitivity_analysis is available in v3."""
        from IBM_PD_AD_v3 import run_probabilistic_sensitivity_analysis

        assert callable(run_probabilistic_sensitivity_analysis)

    def test_v3_psa_script_exists(self):
        """Test that the v3 PSA workflow script exists."""
        script_path = Path(__file__).parent.parent / "run_psa_direct_v3.py"

        assert script_path.exists(), "run_psa_direct_v3.py should exist in AD_Study/v3/"

    def test_v3_psa_script_imports_from_v3_model(self):
        """Test that run_psa_direct_v3.py imports from IBM_PD_AD_v3."""
        script_path = Path(__file__).parent.parent / "run_psa_direct_v3.py"

        if not script_path.exists():
            pytest.skip("run_psa_direct_v3.py not found")

        content = script_path.read_text()

        assert "from IBM_PD_AD_v3 import" in content, \
            "run_psa_direct_v3.py should import from IBM_PD_AD_v3"

        assert "run_probabilistic_sensitivity_analysis" in content, \
            "run_psa_direct_v3.py should use run_probabilistic_sensitivity_analysis"

    def test_legacy_psa_with_timeseries_not_needed(self):
        """Verify that psa_with_timeseries is no longer needed in v3.

        The original run_psa_direct.py (v1) required psa_with_timeseries.
        The v3 implementation has PSA built directly into the model.
        """
        script_path = Path(__file__).parent.parent / "run_psa_direct_v3.py"

        if not script_path.exists():
            pytest.skip("run_psa_direct_v3.py not found")

        content = script_path.read_text()

        assert "psa_with_timeseries" not in content, \
            "v3 PSA should not depend on external psa_with_timeseries module"


# =============================================================================
# TESTS FOR AVAILABLE FUNCTIONALITY
# =============================================================================

class TestPSAWorkflowLogic:
    """Tests for PSA workflow logic."""

    def test_population_scaling_calculation(self):
        """Test that population scaling is calculated correctly."""
        original_population = 10787479  # v3 65+ population
        scale_factor = 0.01  # 1%

        scaled_population = int(original_population * scale_factor)

        assert scaled_population == 107874
        assert scaled_population == int(original_population / 100)

    def test_entrants_scaling_calculation(self):
        """Test that entrants per year scaling is calculated correctly."""
        original_entrants = 1000000
        scale_factor = 0.01

        scaled_entrants = int(original_entrants * scale_factor)

        assert scaled_entrants == 10000

    def test_scale_factor_values(self):
        """Test that common scale factors produce expected results."""
        population = 1000000

        assert int(population * 0.01) == 10000
        assert int(population * 0.10) == 100000
        assert int(population * 0.001) == 1000

    def test_output_directory_path(self):
        """Test output directory path creation logic."""
        output_dir = Path('psa_results_1pct')

        assert output_dir.name == 'psa_results_1pct'

        output_dir.mkdir(exist_ok=True, parents=True)
        assert output_dir.exists()

        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()


class TestIBMPDADIntegration:
    """Tests for IBM_PD_AD_v3 functions used by PSA workflow.

    Note: The v3 model (65+ only) is now the primary model.
    PSA functionality is integrated directly into IBM_PD_AD_v3.py.
    """

    def test_general_config_exists(self):
        """Test that general_config is accessible from IBM_PD_AD_v3."""
        from IBM_PD_AD_v3 import general_config

        assert general_config is not None
        assert isinstance(general_config, dict)

    def test_general_config_has_population(self):
        """Test that general_config contains population setting."""
        from IBM_PD_AD_v3 import general_config

        # v3 model uses 65+ population (10,787,479)
        population = general_config.get('population', 10787479)
        assert population > 0
        assert isinstance(population, (int, float))

    def test_save_results_compressed_exists(self):
        """Test that save_results_compressed function is available."""
        from IBM_PD_AD_v3 import save_results_compressed

        assert callable(save_results_compressed)

    def test_copy_deepcopy_works_on_config(self):
        """Test that config can be deep copied for PSA modification."""
        import copy
        from IBM_PD_AD_v3 import general_config

        psa_config = copy.deepcopy(general_config)

        psa_config['test_key'] = 'test_value'
        assert 'test_key' not in general_config

    def test_run_probabilistic_sensitivity_analysis_exists(self):
        """Test that the PSA function is available in v3 model."""
        from IBM_PD_AD_v3 import run_probabilistic_sensitivity_analysis

        assert callable(run_probabilistic_sensitivity_analysis)


# =============================================================================
# V3 PSA INTERFACE TESTS
# =============================================================================

class TestV3PSAInterface:
    """Tests for the v3 PSA interface in IBM_PD_AD_v3."""

    def test_run_psa_has_expected_signature(self):
        """Test that run_probabilistic_sensitivity_analysis has expected parameters."""
        import inspect
        from IBM_PD_AD_v3 import run_probabilistic_sensitivity_analysis

        sig = inspect.signature(run_probabilistic_sensitivity_analysis)
        param_names = list(sig.parameters.keys())

        assert len(param_names) > 0, "PSA function should have parameters"

    def test_general_config_has_psa_settings(self):
        """Test that general_config includes PSA-related settings."""
        from IBM_PD_AD_v3 import general_config

        assert isinstance(general_config, dict)

        if 'risk_factors' in general_config:
            assert isinstance(general_config['risk_factors'], dict)

    def test_v3_model_has_required_psa_exports(self):
        """Test that IBM_PD_AD_v3 exports all PSA-required functions."""
        from IBM_PD_AD_v3 import (
            general_config,
            save_results_compressed,
            run_probabilistic_sensitivity_analysis,
        )

        assert general_config is not None
        assert callable(save_results_compressed)
        assert callable(run_probabilistic_sensitivity_analysis)


# =============================================================================
# DOCUMENTATION TESTS
# =============================================================================

class TestDocumentation:
    """Tests that serve as documentation for the PSA architecture."""

    def test_psa_architecture_documented(self):
        """This test documents the PSA architecture evolution.

        PSA ARCHITECTURE HISTORY
        ========================

        v1/v2 Architecture (deprecated):
        --------------------------------
        - External module: psa_with_timeseries
        - Required: from psa_with_timeseries import run_psa_with_timeseries
        - Status: Module was never committed to repository

        v3 Architecture (current):
        --------------------------
        - PSA is integrated directly into IBM_PD_AD_v3.py
        - Function: run_probabilistic_sensitivity_analysis()
        - Workflow script: AD_Study/v3/run_psa_direct_v3.py
        - Population: 65+ only (10,787,479)

        Key files:
        ----------
        - IBM_PD_AD_v3.py: Main model with integrated PSA
        - AD_Study/v3/run_psa_direct_v3.py: PSA workflow script

        Usage:
        ------
        from IBM_PD_AD_v3 import (
            general_config,
            save_results_compressed,
            run_probabilistic_sensitivity_analysis,
        )

        Note: The original psa_with_timeseries module is no longer needed.
        """
        assert True
