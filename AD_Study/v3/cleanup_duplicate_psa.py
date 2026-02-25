"""
Cleanup Script - Remove Duplicate PSA Results

This script helps identify and remove duplicate PSA result directories that may have
been created when run_psa_direct_v3.py was executed from different working directories.

SAFE MODE: This script only identifies duplicates by default.
Set ACTUALLY_DELETE = True to perform deletion.
"""

from pathlib import Path
import shutil

# Safety flag - must be explicitly enabled
ACTUALLY_DELETE = False

# Get script directory (AD_Model_v3)
SCRIPT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = SCRIPT_DIR.parent

print("="*80)
print("PSA DUPLICATE CLEANUP UTILITY")
print("="*80)
print()
print(f"Script directory: {SCRIPT_DIR}")
print(f"Parent directory: {PARENT_DIR}")
print()

# Check for PSA directories in both locations
psa_dirs = {
    'Inside AD_Model_v3': [],
    'Outside AD_Model_v3 (parent)': []
}

for prevalence in [25, 50, 75]:
    dir_name = f'psa_results_{prevalence}_v3'

    # Check inside AD_Model_v3
    inside_path = SCRIPT_DIR / dir_name
    if inside_path.exists():
        psa_dirs['Inside AD_Model_v3'].append(inside_path)

    # Check in parent directory
    outside_path = PARENT_DIR / dir_name
    if outside_path.exists():
        psa_dirs['Outside AD_Model_v3 (parent)'].append(outside_path)

# Report findings
print("FOUND PSA DIRECTORIES:")
print("-"*80)
for location, dirs in psa_dirs.items():
    print(f"\n{location}:")
    if dirs:
        for d in dirs:
            size_mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / (1024*1024)
            print(f"  ✓ {d.name} ({size_mb:.1f} MB)")
    else:
        print(f"  (none)")

print()
print("="*80)

# Check if duplicates exist
has_duplicates = all(len(dirs) > 0 for dirs in psa_dirs.values())

if has_duplicates:
    print("DUPLICATE PSA DIRECTORIES DETECTED")
    print("="*80)
    print()
    print("The PSA results exist in both locations, likely because the script was run")
    print("from different working directories at different times.")
    print()
    print("RECOMMENDED ACTION:")
    print("  - Keep: PSA directories INSIDE AD_Model_v3/ (these will be used by the pipeline)")
    print("  - Delete: PSA directories OUTSIDE AD_Model_v3/ (in parent directory)")
    print()

    if not ACTUALLY_DELETE:
        print("="*80)
        print("SAFE MODE: No files will be deleted")
        print("="*80)
        print()
        print("To actually delete the duplicate PSA directories outside AD_Model_v3/,")
        print("edit this script and set: ACTUALLY_DELETE = True")
        print()
        print("Then run: python cleanup_duplicate_psa.py")
        print()
    else:
        print("="*80)
        print("DELETION MODE ENABLED")
        print("="*80)
        print()
        print("The following directories will be DELETED:")
        for d in psa_dirs['Outside AD_Model_v3 (parent)']:
            print(f"  ✗ {d}")
        print()

        response = input("Are you sure you want to delete these directories? (yes/no): ")

        if response.lower() == 'yes':
            print()
            print("Deleting duplicate PSA directories...")
            for d in psa_dirs['Outside AD_Model_v3 (parent)']:
                try:
                    shutil.rmtree(d)
                    print(f"  ✓ Deleted: {d.name}")
                except Exception as e:
                    print(f"  ✗ Failed to delete {d.name}: {e}")
            print()
            print("✓ Cleanup complete!")
        else:
            print()
            print("Deletion cancelled.")

elif len(psa_dirs['Inside AD_Model_v3']) > 0:
    print("✓ PSA directories found only inside AD_Model_v3/ (correct location)")
    print("  No duplicates detected. No cleanup needed!")

elif len(psa_dirs['Outside AD_Model_v3 (parent)']) > 0:
    print("⚠ PSA directories found only OUTSIDE AD_Model_v3/ (parent directory)")
    print()
    print("RECOMMENDATION:")
    print("  These should be inside AD_Model_v3/ for the pipeline to find them.")
    print("  Either:")
    print("    1. Move them into AD_Model_v3/")
    print("    2. Delete them and re-run the PSA")

else:
    print("No PSA result directories found in either location.")
    print("Run the PSA to generate results: python run_psa_direct_v3.py")

print()
print("="*80)
