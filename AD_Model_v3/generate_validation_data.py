"""
Generate 2024 Model Results for External Validation

This script runs IBM_PD_AD to 2024 with the FULL population and saves
the results for external validation.

Run this ONCE to generate the data, then use external_validation.py
to load the results and perform validation against observed data.

Usage:
    python generate_validation_data.py

This will:
1. Run IBM_PD_AD from 2023 to 2024 (full population)
2. Save results to 'validation_results_2024.pkl.gz'
3. These results can then be loaded instantly by external_validation.py
"""

import copy
from pathlib import Path
from IBM_PD_AD import run_model, general_config, save_results_compressed


def generate_2024_validation_data(seed=42, save_path="validation_results_2024.pkl.gz"):
    """
    Run the full model to 2024 and save results for validation

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    save_path : str
        Path to save compressed results
    """
    print("="*80)
    print("GENERATING 2024 VALIDATION DATA")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Population: {general_config['population']:,}")
    print(f"  Base year: {general_config.get('base_year', 2023)}")
    print(f"  Target year: 2024")
    print(f"  Random seed: {seed}")

    # Create a copy of config and set to run to 2024
    validation_config = copy.deepcopy(general_config)

    base_year = int(validation_config.get('base_year', 2023))
    target_year = 2024
    timesteps_needed = target_year - base_year

    validation_config['number_of_timesteps'] = timesteps_needed

    print(f"\nRunning model for {timesteps_needed} timestep(s)...")
    print("This may take several minutes with full population...\n")

    # Run the model (same way IBM_PD_AD.py runs it)
    results = run_model(validation_config, seed=seed, return_agents=False)

    print(f"\nModel run complete!")
    print(f"  Final year: {base_year + timesteps_needed}")

    # Save results using IBM_PD_AD's built-in compression
    print(f"\nSaving results to: {save_path}")
    save_results_compressed(results, save_path)

    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nValidation data saved to: {save_path}")
    print(f"\nNext step:")
    print(f"  python external_validation.py")
    print(f"\nThis will load the saved results and perform validation.")

    return results


if __name__ == "__main__":
    # Generate validation data with full population
    generate_2024_validation_data(seed=42)
