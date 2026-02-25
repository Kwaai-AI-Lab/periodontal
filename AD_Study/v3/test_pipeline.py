"""
Quick Test Script for Analysis Pipeline

This script runs a quick test of the full analysis pipeline with 1% population
to verify everything is configured correctly before a full weekend run.

Estimated runtime: 10-20 minutes
"""

import sys
import io
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for output (Windows compatibility)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import the configuration from run_full_analysis
from run_full_analysis import main, CONFIG, Logger

# Override config for quick testing
TEST_CONFIG = CONFIG.copy()
TEST_CONFIG.update({
    'output_dir': Path('test_analysis_results'),
    'log_file': 'test_analysis_log.txt',

    # Use small population fractions for speed
    'main_model_population_fraction': 0.01,  # 1% population
    'main_model_prevalences': [0.50],  # Test only baseline

    # Reduce iterations for speed
    'run_psa': False,  # Skip PSA in quick test (takes too long)
    'psa_iterations': 10,  # If enabled, use minimal iterations

    # Reduce replicates for tornado
    'run_tornado': False,  # Skip tornado in quick test
    'tornado_n_replicates': 2,
    'tornado_prevalences': [0.50],

    # Validation should be quick
    'run_external_validation': True,
    'validation_years': [2024],  # Test only one year

    # Skip long-running steps in quick test (but they ARE enabled by default in full run)
    'run_age_validation': False,  # Disabled for test speed (takes 1-2 hours)
    'run_counterfactual': False,  # Disabled for test speed (takes 30-60 min)

    # Test figure generation with small population
    'run_generate_figures': True,
    'figures_population_fraction': 0.01,
    'figures_export_excel': False,  # Skip Excel export for speed
})

def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*80)
    print("Testing imports...")
    print("="*80)

    try:
        from IBM_PD_AD_v3 import general_config, run_model
        print("✓ IBM_PD_AD_v3 imported successfully")

        from external_validation import run_population_validation, run_prevalence_validation
        print("✓ external_validation imported successfully")

        from validate_age_distribution import run_full_age_distribution_validation
        print("✓ validate_age_distribution imported successfully")

        from ons_projection_data import ONS_AGE_BAND_MULTIPLIER_SCHEDULE
        print("✓ ons_projection_data imported successfully")

        import numpy, pandas, matplotlib
        print("✓ Required packages (numpy, pandas, matplotlib) imported successfully")

        print("\n✓ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False


def test_configuration():
    """Test that configuration is valid"""
    print("\n" + "="*80)
    print("Testing configuration...")
    print("="*80)

    try:
        from IBM_PD_AD_v3 import general_config

        # Check key config values
        assert 'population' in general_config, "Missing 'population' in config"
        assert 'base_year' in general_config, "Missing 'base_year' in config"
        assert 'risk_factors' in general_config, "Missing 'risk_factors' in config"
        assert 'open_population' in general_config, "Missing 'open_population' in config"

        # Check fixed_entry_age is set
        fixed_entry_age = general_config.get('open_population', {}).get('fixed_entry_age')
        if fixed_entry_age == 65:
            print(f"✓ fixed_entry_age correctly set to 65")
        else:
            print(f"⚠ Warning: fixed_entry_age is {fixed_entry_age}, expected 65")

        # Check periodontal disease is configured
        pd_cfg = general_config['risk_factors'].get('periodontal_disease')
        assert pd_cfg is not None, "Periodontal disease not in risk_factors"
        print("✓ Periodontal disease risk factor configured")

        print("\n✓ Configuration valid!")
        return True

    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        return False


def run_quick_model_test():
    """Run a very quick model test"""
    print("\n" + "="*80)
    print("Running quick model test (1 timestep, 0.1% population)...")
    print("="*80)

    try:
        from IBM_PD_AD_v3 import general_config, run_model
        import copy

        # Create minimal test config
        test_config = copy.deepcopy(general_config)
        test_config['population'] = int(test_config['population'] * 0.001)  # 0.1%
        test_config['number_of_timesteps'] = 1  # Just 1 year

        print(f"Test population: {test_config['population']:,}")
        print(f"Timesteps: {test_config['number_of_timesteps']}")

        start = datetime.now()
        results = run_model(test_config, seed=42, return_agents=False)
        duration = (datetime.now() - start).total_seconds()

        print(f"✓ Model ran successfully in {duration:.1f} seconds")

        # Basic validation of results
        assert 'summary_history' in results, "Missing summary_history in results"
        assert len(results['summary_history']) > 0, "Empty summary_history"

        print("✓ Results structure valid")
        print("\n✓ Quick model test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main_test():
    """Run all tests"""
    print("="*80)
    print("PIPELINE TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now()}")
    print(f"This will test the pipeline with 1% population and minimal settings")
    print("Estimated runtime: 10-20 minutes")
    print("="*80)

    test_results = {}

    # Test 1: Imports
    test_results['Imports'] = test_imports()

    # Test 2: Configuration
    test_results['Configuration'] = test_configuration()

    # Test 3: Quick model run
    test_results['Quick Model'] = run_quick_model_test()

    # Test 4: Run pipeline with test config
    if all(test_results.values()):
        print("\n" + "="*80)
        print("All pre-flight tests passed! Running pipeline test...")
        print("="*80)

        # Temporarily replace global CONFIG
        import run_full_analysis
        original_config = run_full_analysis.CONFIG
        run_full_analysis.CONFIG = TEST_CONFIG

        try:
            result = main()
            test_results['Pipeline'] = (result == 0)
        except Exception as e:
            print(f"\n✗ Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            test_results['Pipeline'] = False
        finally:
            run_full_analysis.CONFIG = original_config
    else:
        print("\n⚠ Skipping pipeline test due to earlier failures")
        test_results['Pipeline'] = None

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in test_results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {test_name:<20} {status}")

    print("="*80)

    if all(v in [True, None] for v in test_results.values()):
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now run the full analysis with:")
        print("  python run_full_analysis.py")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please fix the issues before running the full analysis.")
        return 1


if __name__ == "__main__":
    exit_code = main_test()
    sys.exit(exit_code)
