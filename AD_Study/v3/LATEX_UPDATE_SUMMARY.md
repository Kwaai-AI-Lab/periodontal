# LaTeX File Update Summary

## Date: 2026-02-18

## Source
Updated from: `AD_FullText_v3.docx` → converted to `AD_FullText_v3.md` → applied to `AD_FullText_v3.tex`

## Changes Made

### 1. **Added Package**
- Added `\usepackage{makecell}` to preamble (line 7)
- Required for multi-line table headers in Table 5

### 2. **Methods - Sensitivity Analysis Section (Updated)**

**Old text:**
> We conducted probabilistic sensitivity analysis (PSA) with 500 Monte Carlo iterations for 1% of the target population (n = 107,874), sampling uncertain parameters from probability distributions (sampling costs with the gamma distribution, utilities and probabilities with beta, and risk HRs with lognormal from CIs; relative SD = 10%) [41]. The outcomes were then scaled back up by a factor of 100. This was due to the computational strain of sampling for the full population. Results are reported as means with 95% confidence intervals.
>
> Deterministic one-way sensitivity analysis was conducted to specifically address the uncertainty surrounding PD's impact on dementia onset using the published 95% CI bounds for PD HRs (1.07-1.38 for onset). As with our PSA, the runs used 1% of the total population with 10 independent model runs per PD prevalence due to computational feasibility. Outputs were then scaled up to full-cohort equivalents.

**New text:**
> We conducted probabilistic sensitivity analysis (PSA) with 500 Monte Carlo iterations for 1% of the target population (n = 107,874), sampling uncertain parameters from probability distributions (sampling costs with the gamma distribution, utilities and probabilities with beta, and risk HRs with lognormal from CIs; relative SD = 10%) [41]. The outcomes were then scaled back up by a factor of 100. This was due to the computational strain of sampling for the full population. Results are reported as means with 95% confidence intervals. **Coefficients of variation ((standard deviation / mean)*100) are also reported.**
>
> Deterministic one-way sensitivity analysis was conducted to specifically address the uncertainty surrounding PD's impact on dementia onset using the published 95% CI bounds for PD HRs (1.07-1.38 for onset). **We used 100% of the total population, with each lower and upper bound of the HR used for each PD prevalence run.**

**Changes:**
- Added mention of Coefficients of variation (CV) reporting
- Changed one-way sensitivity from "1% population with 10 replicates" to "100% population"

### 3. **Table 4 - PSA Results (Updated Values)**

**Caption updated:**
- Added "2024£ costs." to caption

**Values updated:**
| Outcome | 25% PD | 50% PD | 75% PD |
|---------|--------|--------|--------|
| Total QALYs (mn) | **151.9** (136.1-167.2); CV: **5.3%** | **152.2** (136.4-167.3); CV: **5.3%** | **152.4** (136.6-167.4); 5.2% |
| Incident cases (mn) | **2.7** (2.2-3.2); CV: **9.1%** | **2.8** (2.3-3.3); CV: **9.3%** | **2.9** (2.4-3.5); **9.7%** |

(Total costs values remained the same)

### 4. **Results - Sensitivity Analysis Section (NEW Content Added)**

**Replaced:**
> Deterministic one-way sensitivity analysis varying the PD-dementia onset HR across its confidence interval (1.07-1.38) showed proportional changes in dementia incidence and costs, confirming the linear relationship between PD prevalence and dementia outcomes.

**With:**
> The results of our one-way sensitivity analysis are summarised in Table 5 (full results in Supplementary Material, Table XXX). Using the lower HR reduced total costs, caregiver QALYs, and incident dementia onsets by 8%-13% across scenarios. Total population QALYs were largely insensitive to HR variation, changing by only ±3% across all scenarios. All results behaved as expected, except the fall in total incident onsets under 25% PD prevalence when the upper bound HR was used. For all other outputs, using the upper HR led to total costs falling by 1%-5% and incident onsets increasing by up to 3% under the 75% PD prevalence scenario.

### 5. **Table 5 - One-Way Sensitivity Results (NEW Table Added)**

**NEW Table 5:**
- Caption: "One-way sensitivity results from varying periodontal disease-dementia onset HR. * = Not including caregiver QALYs. All costs are in 2024£."
- Label: `\label{tab:oneway}`

**Structure:**
- 6 data rows (3 PD prevalence levels × 2 HR bounds)
- Columns:
  1. PD prevalence (25%, 50%, 75%)
  2. HR Used (High 1.38, Low 1.07)
  3. Total Costs (% change from baseline)
  4. Total population QALYs* (% change from baseline)
  5. Total caregiver QALYs (% change from baseline)
  6. Total Incident Onsets (% change from baseline)

**Sample data:**
- 75% PD, High HR: £643,958,973,993.60 (-1%), 136,230,776.09 QALYs (-3%), 10,981,842.40 caregiver QALYs (1%), 3,087,083 onsets (3%)
- 75% PD, Low HR: £564,567,763,775.41 (-13%), 136,915,148.84 QALYs (-3%), 9,545,927.86 caregiver QALYs (-12%), 2,621,631 onsets (-13%)
- [Similar pattern for 50% and 25% PD prevalence]

## Verification

To compile the updated LaTeX file:
```bash
pdflatex AD_FullText_v3.tex
bibtex AD_FullText_v3
pdflatex AD_FullText_v3.tex
pdflatex AD_FullText_v3.tex
```

## Files Affected
- ✅ `AD_FullText_v3.tex` - Updated
- ✅ `AD_FullText_v3.md` - Already converted from .docx
- ✅ `AD_FullText_v3.docx` - Source document (no changes needed)

## Next Steps
1. Review the LaTeX file to ensure formatting is correct
2. Compile to PDF to verify table rendering
3. Check that all cross-references work correctly (Table \ref{tab:oneway})
4. Verify supplementary material reference (Table XXX) is updated when available
