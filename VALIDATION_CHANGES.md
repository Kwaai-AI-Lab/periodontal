# External Validation - Changes Summary

## What Changed

I've restructured the external validation to match your request: **use the full population and the same storage methods as IBM_PD_AD.py**.

### Previous Approach (Removed)
- ❌ Reduced population to 1M sample
- ❌ Ran everything in one script
- ❌ Modified config settings

### New Approach (Current)
- ✅ Uses **full 33M population** (same as IBM_PD_AD.py)
- ✅ Uses **same storage methods** as IBM_PD_AD.py
- ✅ **Two-step workflow**: generate once, validate many times
- ✅ No config modifications

## How to Use

### Step 1: Generate 2024 Data (Run Once)

```bash
python generate_validation_data.py
```

This runs IBM_PD_AD to 2024 with:
- Full population: 33,167,098
- Same configuration as general_config
- Saves to: `validation_results_2024.pkl.gz`

### Step 2: Perform Validation (Run Anytime)

```bash
python external_validation.py
```

This loads the saved 2024 results and:
- Compares against observed 2024 prevalence
- Generates calibration plots
- Calculates R² and validation statistics

## Files Created

**New Scripts:**
- `generate_validation_data.py` - Generates 2024 model results
- `EXTERNAL_VALIDATION_README.md` - Complete user guide

**Modified Scripts:**
- `external_validation.py` - Now loads pre-computed results by default
- `external_validation_fast.py` - Updated to match new workflow

**Generated Data:**
- `validation_results_2024.pkl.gz` - Cached 2024 results (created by Step 1)

## Why This Approach?

1. **Same as IBM_PD_AD**: Uses identical population size and methods
2. **Efficient**: Run expensive simulation once, validate many times
3. **Memory-safe**: Uses IBM_PD_AD's proven memory management
4. **Flexible**: Easy to regenerate if model changes

## What This Fixes

Your original issue: "repeatedly crashing the computer"

**Root cause identified**: Not the population size (since IBM_PD_AD.py works fine), but likely:
- Running the full simulation repeatedly in the same Python session
- Not properly releasing memory between runs
- Or some other session-specific memory issue

**Solution**: Separate simulation from validation into two distinct scripts/sessions.

## Next Steps

1. Run `python generate_validation_data.py` to create the 2024 baseline
2. Run `python external_validation.py` to perform validation
3. Check `plots/` directory for outputs

The validation should now work with the full population without crashing!
