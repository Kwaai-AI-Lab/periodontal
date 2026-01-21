"""
External Validation - Using Pre-computed Results

This script demonstrates loading pre-computed model results for validation.

WORKFLOW:
1. First run: python generate_validation_data.py
   (This runs IBM_PD_AD to 2024 with FULL population and saves results)

2. Then run this script to perform validation against observed data

This approach allows you to:
- Use the full 33M population (same as IBM_PD_AD.py)
- Run the simulation once, validate many times
- Quickly iterate on validation plots/analysis

Usage:
    python external_validation_fast.py
"""

from external_validation import run_external_validation
import os

if __name__ == "__main__":
    results_file = "validation_results_2024.pkl.gz"

    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        print(f"\nPlease run this first:")
        print(f"  python generate_validation_data.py")
        print(f"\nThis will generate the 2024 model results with full population.")
        exit(1)

    # Load pre-computed results and perform validation
    validation_results = run_external_validation(
        seed=42,
        results_file=results_file
    )

    print("\n[SUCCESS] External validation complete!")
    print("  All plots and tables generated from full population results")
