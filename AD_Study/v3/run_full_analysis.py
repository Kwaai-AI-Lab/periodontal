"""
Master Script - Full Analysis Pipeline

Runs the complete analysis suite sequentially for weekend batch processing:
1. Main model runs at 25%, 50%, 75% PD prevalence (full population)
2. PSA (Probabilistic Sensitivity Analysis)
3. One-way sensitivity analysis (tornado diagrams)
4. External validation (population and prevalence)
5. Age distribution validation (ONS projections)
6. Counterfactual analysis (non-preventable risk)
7. Generate manuscript figures

Each step logs progress and saves results. If a step fails, the script
continues to the next step and logs the error.
"""

import sys
import io
from pathlib import Path
from datetime import datetime
import traceback
import copy

# Set UTF-8 encoding for output (Windows compatibility)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION - Adjust these settings as needed
# ============================================================================

CONFIG = {
    # General settings
    'seed': 42,
    'output_dir': Path('full_analysis_results'),
    'log_file': 'full_analysis_log.txt',

    # Step 1: Main model runs
    'run_main_model': True,
    'main_model_prevalences': [0.25, 0.50, 0.75],
    'main_model_population_fraction': 1.0,  # Use full population

    # Step 2: PSA
    'run_psa': True,
    'psa_iterations': 500,
    'psa_population_fraction': 0.01,  # 1% population for PSA

    # Step 3: One-way sensitivity analysis
    'run_tornado': True,
    'tornado_n_replicates': 10,
    'tornado_population_fraction': 0.01,
    'tornado_prevalences': [0.25, 0.50, 0.75],

    # Step 4: External validation
    'run_external_validation': True,
    'validation_years': [2024, 2025],

    # Step 5: Age distribution validation
    'run_age_validation': True,

    # Step 6: Counterfactual analysis
    'run_counterfactual': True,

    # Step 7: Generate figures
    'run_generate_figures': True,
    'figures_population_fraction': 1.0,  # Full population for publication figures
    'figures_export_excel': True,
}

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class Logger:
    """Simple logger that writes to both console and file"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear log file
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"Full Analysis Log - Started {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = "INFO"):
        """Log a message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"

        print(formatted)

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")

    def section(self, title: str):
        """Log a section header"""
        separator = "=" * 80
        self.log("")
        self.log(separator)
        self.log(title)
        self.log(separator)
        self.log("")


# ============================================================================
# STEP FUNCTIONS
# ============================================================================

def step_1_main_model_runs(logger: Logger, config: dict) -> bool:
    """Run main model at different PD prevalence levels"""
    logger.section("STEP 1: Main Model Runs (25%, 50%, 75% PD)")

    try:
        from IBM_PD_AD_v3 import general_config, run_model, save_results_compressed, export_results_to_excel

        prevalences = config['main_model_prevalences']
        pop_fraction = config['main_model_population_fraction']
        seed = config['seed']
        output_dir = config['output_dir'] / "main_model_runs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for prevalence in prevalences:
            logger.log(f"Running model with {prevalence*100:.0f}% PD prevalence...")

            # Create config with modified PD prevalence
            run_config = copy.deepcopy(general_config)
            if pop_fraction < 1.0:
                run_config['population'] = int(run_config['population'] * pop_fraction)

            pd_cfg = run_config['risk_factors']['periodontal_disease']
            pd_cfg['prevalence'] = {'female': prevalence, 'male': prevalence}

            # Run model
            results = run_model(run_config, seed=seed, return_agents=False)

            # Save compressed results
            output_file = output_dir / f"results_pd_{int(prevalence*100)}_percent.pkl.gz"
            save_results_compressed(results, output_file)
            logger.log(f"  ✓ Compressed results saved: {output_file}")

            # Export to Excel with unique filename (for journal reproducibility)
            excel_file = Path("results") / f"Baseline_Model_PD_{int(prevalence*100)}.xlsx"
            excel_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                export_results_to_excel(results, path=str(excel_file))
                logger.log(f"  ✓ Excel export saved: {excel_file}")
            except Exception as e:
                logger.log(f"  ⚠ Excel export failed: {e}", level="WARNING")

        logger.log("✓ Step 1 complete!")
        return True

    except Exception as e:
        logger.log(f"✗ Step 1 FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def step_2_psa(logger: Logger, config: dict) -> bool:
    """Run probabilistic sensitivity analysis"""
    logger.section("STEP 2: Probabilistic Sensitivity Analysis (PSA)")

    try:
        logger.log("Executing run_psa_direct_v3.py...")
        logger.log(f"  PSA iterations: {config['psa_iterations']}")
        logger.log(f"  Population fraction: {config['psa_population_fraction']}")

        # Import and run PSA
        # Note: run_psa_direct_v3.py is self-contained and runs when imported as main
        import subprocess
        result = subprocess.run(
            [sys.executable, 'run_psa_direct_v3.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=Path(__file__).parent
        )

        logger.log(result.stdout)
        if result.returncode != 0:
            logger.log(result.stderr, level="ERROR")
            raise RuntimeError(f"PSA script failed with exit code {result.returncode}")

        logger.log("✓ Step 2 complete!")
        return True

    except Exception as e:
        logger.log(f"✗ Step 2 FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def step_3_tornado(logger: Logger, config: dict) -> bool:
    """Run one-way sensitivity analysis for tornado diagrams"""
    logger.section("STEP 3: One-Way Sensitivity Analysis (Tornado Diagrams)")

    try:
        logger.log("Executing run_pd_tornado.py...")
        logger.log(f"  Replicates: {config['tornado_n_replicates']}")
        logger.log(f"  Prevalence levels: {[f'{p*100:.0f}%' for p in config['tornado_prevalences']]}")

        import subprocess
        result = subprocess.run(
            [sys.executable, 'run_pd_tornado.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=Path(__file__).parent
        )

        logger.log(result.stdout)
        if result.returncode != 0:
            logger.log(result.stderr, level="ERROR")
            raise RuntimeError(f"Tornado script failed with exit code {result.returncode}")

        logger.log("✓ Step 3 complete!")
        return True

    except Exception as e:
        logger.log(f"✗ Step 3 FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def step_4_external_validation(logger: Logger, config: dict) -> bool:
    """Run external validation against NHS data"""
    logger.section("STEP 4: External Validation (Population & Prevalence)")

    try:
        from external_validation import (
            run_population_validation,
            run_prevalence_validation
        )

        output_dir = config['output_dir'] / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = config['seed']

        # Population validation
        logger.log("Running population validation (2024)...")
        pop_stats = run_population_validation(seed=seed, save_dir=str(output_dir))
        logger.log(f"  R² = {pop_stats['r2']:.4f}, β = {pop_stats['beta']:.4f}")

        # Prevalence validation for each year
        for year in config['validation_years']:
            logger.log(f"Running prevalence validation ({year})...")
            prev_stats = run_prevalence_validation(year=year, seed=seed, save_dir=str(output_dir))
            logger.log(f"  R² = {prev_stats['r2']:.4f}, β = {prev_stats['beta']:.4f}")

        logger.log("✓ Step 4 complete!")
        return True

    except Exception as e:
        logger.log(f"✗ Step 4 FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def step_5_age_validation(logger: Logger, config: dict) -> bool:
    """Run age distribution validation against ONS projections"""
    logger.section("STEP 5: Age Distribution Validation (ONS Projections)")

    try:
        logger.log("Warning: This step can take 1-2 hours with full population")
        logger.log("Running validate_age_distribution.py...")

        from validate_age_distribution import run_full_age_distribution_validation

        output_dir = config['output_dir'] / "age_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = config['seed']

        results = run_full_age_distribution_validation(seed=seed, save_dir=str(output_dir))

        # Log summary
        if results:
            for result in results:
                if result:
                    logger.log(f"  {result['year']}: R² = {result['fit_stats']['r2']:.4f}, "
                             f"β = {result['fit_stats']['beta']:.4f}")

        logger.log("✓ Step 5 complete!")
        return True

    except Exception as e:
        logger.log(f"✗ Step 5 FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def step_6_counterfactual(logger: Logger, config: dict) -> bool:
    """Run counterfactual analysis for non-preventable risk"""
    logger.section("STEP 6: Counterfactual Analysis (Non-Preventable Risk)")

    try:
        logger.log("Executing run_non_preventable_risk_analysis.py...")

        import subprocess
        result = subprocess.run(
            [sys.executable, 'run_non_preventable_risk_analysis.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=Path(__file__).parent
        )

        logger.log(result.stdout)
        if result.returncode != 0:
            logger.log(result.stderr, level="ERROR")
            raise RuntimeError(f"Counterfactual analysis failed with exit code {result.returncode}")

        logger.log("✓ Step 6 complete!")
        return True

    except Exception as e:
        logger.log(f"✗ Step 6 FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


def step_7_generate_figures(logger: Logger, config: dict) -> bool:
    """Generate manuscript figures"""
    logger.section("STEP 7: Generate Manuscript Figures")

    try:
        logger.log("Executing generate_figures_2_3_4_from_model.py...")
        logger.log(f"  Population fraction: {config['figures_population_fraction']}")
        logger.log(f"  Export Excel: {config['figures_export_excel']}")

        import subprocess

        cmd = [sys.executable, 'generate_figures_2_3_4_from_model.py']

        if config['figures_population_fraction'] < 1.0:
            cmd.extend(['--population-fraction', str(config['figures_population_fraction'])])

        if config['figures_export_excel']:
            cmd.append('--export-excel')

        cmd.extend(['--seed', str(config['seed'])])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=Path(__file__).parent
        )

        logger.log(result.stdout)
        if result.returncode != 0:
            logger.log(result.stderr, level="ERROR")
            raise RuntimeError(f"Figure generation failed with exit code {result.returncode}")

        logger.log("✓ Step 7 complete!")
        return True

    except Exception as e:
        logger.log(f"✗ Step 7 FAILED: {str(e)}", level="ERROR")
        logger.log(traceback.format_exc(), level="ERROR")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete analysis pipeline"""

    # Initialize
    start_time = datetime.now()
    output_dir = CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / CONFIG['log_file']
    logger = Logger(log_path)

    logger.section("FULL ANALYSIS PIPELINE - START")
    logger.log(f"Configuration: {CONFIG}")
    logger.log(f"Start time: {start_time}")
    logger.log(f"Output directory: {output_dir.absolute()}")
    logger.log(f"Log file: {log_path.absolute()}")

    # Track results
    results = {
        'Step 1: Main Model Runs': None,
        'Step 2: PSA': None,
        'Step 3: Tornado Analysis': None,
        'Step 4: External Validation': None,
        'Step 5: Age Validation': None,
        'Step 6: Counterfactual Analysis': None,
        'Step 7: Generate Figures': None,
    }

    # Execute steps
    if CONFIG['run_main_model']:
        results['Step 1: Main Model Runs'] = step_1_main_model_runs(logger, CONFIG)
    else:
        logger.log("Step 1 skipped (disabled in config)")

    if CONFIG['run_psa']:
        results['Step 2: PSA'] = step_2_psa(logger, CONFIG)
    else:
        logger.log("Step 2 skipped (disabled in config)")

    if CONFIG['run_tornado']:
        results['Step 3: Tornado Analysis'] = step_3_tornado(logger, CONFIG)
    else:
        logger.log("Step 3 skipped (disabled in config)")

    if CONFIG['run_external_validation']:
        results['Step 4: External Validation'] = step_4_external_validation(logger, CONFIG)
    else:
        logger.log("Step 4 skipped (disabled in config)")

    if CONFIG['run_age_validation']:
        results['Step 5: Age Validation'] = step_5_age_validation(logger, CONFIG)
    else:
        logger.log("Step 5 skipped (disabled in config)")

    if CONFIG['run_counterfactual']:
        results['Step 6: Counterfactual Analysis'] = step_6_counterfactual(logger, CONFIG)
    else:
        logger.log("Step 6 skipped (disabled in config)")

    if CONFIG['run_generate_figures']:
        results['Step 7: Generate Figures'] = step_7_generate_figures(logger, CONFIG)
    else:
        logger.log("Step 7 skipped (disabled in config)")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.section("FULL ANALYSIS PIPELINE - COMPLETE")
    logger.log(f"End time: {end_time}")
    logger.log(f"Total duration: {duration}")
    logger.log("")
    logger.log("SUMMARY:")
    logger.log("-" * 80)

    for step_name, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
        logger.log(f"  {step_name}: {status}")

    logger.log("-" * 80)
    logger.log("")
    logger.log(f"Full log saved to: {log_path.absolute()}")
    logger.log(f"Results directory: {output_dir.absolute()}")

    # Exit with error code if any step failed
    if any(result is False for result in results.values()):
        logger.log("\nWARNING: Some steps failed. Check the log for details.", level="ERROR")
        return 1

    logger.log("\n✓ ALL ENABLED STEPS COMPLETED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
