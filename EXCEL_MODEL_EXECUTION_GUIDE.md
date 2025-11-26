# Excel Model Execution Guide
## CVD Markov Model - Step-by-Step Instructions

### Date: November 12, 2025
### Model File: `PD_CVD_markov - PSA On.xlsm`

---

## PART 1: Pre-Execution Checklist

Before opening the Excel file, ensure you have:
- [ ] Microsoft Excel installed (preferably 2016 or later)
- [ ] Macros enabled in Excel (required for PSA functionality)
- [ ] At least 2GB free RAM (for 10,000 Monte Carlo simulations)
- [ ] Backup copy of the Excel file saved

---

## PART 2: Opening and Verifying the Model

### Step 1: Open Excel File
1. Navigate to: `/Users/rezarassool/Source/periodontal/`
2. Double-click: `PD_CVD_markov - PSA On.xlsm`
3. **Enable Macros** when prompted (required for PSA simulations)

### Step 2: Verify Model Structure
Expected worksheets (check tabs at bottom):
- [ ] **Parameters**: All model inputs (baseline hazards, HRs, costs, utilities)
- [ ] **Base Case**: Cohort simulation without treatment
- [ ] **Treatment**: Cohort simulation with non-surgical periodontal therapy
- [ ] **Results**: Summary outputs (ICER, NMB, total costs/QALYs)
- [ ] **PSA**: Probabilistic sensitivity analysis setup
- [ ] **PSA Results**: Storage for 10,000 simulations
- [ ] **Figures**: Cost-effectiveness plane, CEAC, tornado diagrams

### Step 3: Verify Key Parameters
Navigate to **Parameters** worksheet and verify these critical values are populated:

#### Baseline Hazards (Age 65):
- Base state stroke hazard: _____ (record value)
- Base state MI hazard: _____ (record value)
- Background death probability: _____ (should align with ONS life tables)

#### Treatment Effect Hazard Ratios:
- Stroke HR (median of 0.40-0.78): Expected ~0.59
- MI HR (median of 0.54-0.90): Expected ~0.72

#### Annual Costs (2024 GBP):
- Periodontal treatment: £1,275/year (verified from methodology)
- Stroke Year 1: _____
- Stroke Year 2+: _____
- MI Year 1: _____
- MI Year 2+: _____

#### Utility Values:
- Base (no CVD): ~0.80-0.85 (age-adjusted)
- Post-stroke Year 1: _____
- Post-stroke Year 2+: _____
- Post-MI Year 1: _____
- Post-MI Year 2+: _____
- Post-both (multiplicative): _____

#### Discount Rate:
- [ ] Confirm 3.5% for both costs and QALYs (NICE standard)

---

## PART 3: Running Base Case Analysis

### Step 4: Execute Base Case
1. Navigate to **Results** or **Control Panel** worksheet
2. Look for button labeled "Run Base Case" or "Calculate"
3. Click to execute single deterministic run
4. Wait for completion (should take 10-30 seconds)

### Step 5: Extract Base Case Results
Record the following outputs in the table below:

| Outcome | No Treatment Arm | Treatment Arm | Incremental |
|---------|------------------|---------------|-------------|
| **Total QALYs** (discounted) | | | |
| **Total Costs** (discounted, £) | | | |
| **Life Years** (undiscounted) | | | |
| **Stroke Events** (count) | | | |
| **MI Events** (count) | | | |
| **Deaths** (count) | | | |

#### Calculate ICER:
```
ICER = (Cost_Treatment - Cost_NoTreatment) / (QALY_Treatment - QALY_NoTreatment)
ICER = £_____ per QALY gained
```

#### Net Monetary Benefit (at £20,000/QALY threshold):
```
NMB = (Incremental QALYs × £20,000) - Incremental Costs
NMB = £_____
```

#### Net Monetary Benefit (at £30,000/QALY threshold):
```
NMB = (Incremental QALYs × £30,000) - Incremental Costs
NMB = £_____
```

---

## PART 4: Running Probabilistic Sensitivity Analysis

### Step 6: Configure PSA Settings
1. Navigate to **PSA** worksheet
2. Verify simulation settings:
   - Number of iterations: 10,000
   - Random seed: (note if specified for reproducibility)
   - Parameter distributions:
     - [ ] Costs: Gamma distribution
     - [ ] Utilities: Beta distribution
     - [ ] Transition probabilities: Beta distribution

### Step 7: Execute PSA
1. Click "Run PSA" or "Monte Carlo Simulation" button
2. **IMPORTANT**: This may take 30 minutes to 2 hours depending on your computer
3. Monitor progress bar/counter
4. Do not close Excel or interrupt the process
5. Save file after completion

### Step 8: Extract PSA Summary Statistics

| Statistic | Incremental Costs (£) | Incremental QALYs |
|-----------|----------------------|-------------------|
| **Mean** | | |
| **Median** | | |
| **95% CI Lower** | | |
| **95% CI Upper** | | |
| **Standard Deviation** | | |

#### Cost-Effectiveness Probability:
- Probability cost-effective at £20,000/QALY: _____%
- Probability cost-effective at £30,000/QALY: _____%

---

## PART 5: One-Way Sensitivity Analysis

### Step 9: Run Deterministic Sensitivity Analysis
For each parameter, vary between min and max values and record ICER:

| Parameter | Base Case | Min Value | ICER at Min | Max Value | ICER at Max | Range |
|-----------|-----------|-----------|-------------|-----------|-------------|-------|
| Stroke HR | 0.59 | 0.40 | | 0.78 | | |
| MI HR | 0.72 | 0.54 | | 0.90 | | |
| Periodontal cost | £1,275 | £1,020 (-20%) | | £1,530 (+20%) | | |
| Stroke Y1 cost | | -20% | | +20% | | |
| MI Y1 cost | | -20% | | +20% | | |
| Post-stroke utility | | CI lower | | CI upper | | |
| Post-MI utility | | CI lower | | CI upper | | |
| Discount rate | 3.5% | 1.5% | | 5% | | |

**Note**: Rank parameters by "Range" to identify most influential for tornado diagram.

---

## PART 6: Extract Figures and Data for Publication

### Step 10: Cost-Effectiveness Plane
1. Navigate to **PSA Results** or **Figures** worksheet
2. Locate scatter plot of 10,000 simulations
   - X-axis: Incremental QALYs
   - Y-axis: Incremental Costs (£)
3. Verify willingness-to-pay threshold lines (£20k, £30k) are displayed
4. **Export as high-resolution image**:
   - Right-click chart → Save as Picture → PNG (600 DPI minimum)
   - Save as: `CVD_CE_Plane.png`

### Step 11: Cost-Effectiveness Acceptability Curve (CEAC)
1. Locate CEAC chart showing probability cost-effective across WTP thresholds
   - X-axis: Willingness-to-pay threshold (£0 to £50,000)
   - Y-axis: Probability cost-effective (0% to 100%)
2. Record key probabilities:
   - At £20,000/QALY: _____%
   - At £30,000/QALY: _____%
3. **Export as high-resolution image**: `CVD_CEAC.png`

### Step 12: Tornado Diagram
1. Locate one-way sensitivity tornado diagram
2. Verify parameters are ranked by impact on ICER
3. **Export as high-resolution image**: `CVD_Tornado.png`

### Step 13: Markov Trace
1. Navigate to **Base Case** or **Treatment** worksheet
2. Locate cohort distribution table showing proportion in each health state over 10 years
3. Export data table:

| Cycle | Base | Post-Stroke Y1 | Post-Stroke Y2+ | Post-MI Y1 | Post-MI Y2+ | Post-Both Y1 | Post-Both Y2+ | Dead |
|-------|------|----------------|-----------------|------------|-------------|--------------|---------------|------|
| 0 | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| ... | | | | | | | | |
| 10 | | | | | | | | |

4. Create stacked area chart showing state proportions over time
5. **Export as high-resolution image**: `CVD_Markov_Trace.png`

---

## PART 7: Export Raw Data for Analysis

### Step 14: Save Key Data Tables
Create CSV exports for further analysis:

1. **Base case results**: `CVD_base_case_results.csv`
2. **PSA raw data** (10,000 × 2 matrix): `CVD_PSA_raw_data.csv`
   - Column 1: Incremental Costs
   - Column 2: Incremental QALYs
3. **One-way sensitivity results**: `CVD_one_way_sensitivity.csv`
4. **Markov trace data**: `CVD_markov_trace_no_treatment.csv` and `CVD_markov_trace_treatment.csv`

---

## PART 8: Complete Parameter Tables for Supplementary Material

### Step 15: Extract All Parameters from Excel Model
From the **Parameters** worksheet, copy all values and populate the supplementary material tables:

#### Table 1: Model Parameters and Values
(Already has structure in Supplementary_Material_CVD.docx - fill in the "Value" column)

| Parameter | Description | Value | Source |
|-----------|-------------|-------|--------|
| Baseline stroke hazard | Annual hazard in base state | | [Reference] |
| Baseline MI hazard | Annual hazard in base state | | [Reference] |
| Background death | Age-specific mortality | | ONS life tables |
| Stroke case fatality | Proportion dying in first year | 10% | [Reference 6] |
| MI case fatality | Proportion dying in first year | 14% | [Reference 7] |
| ... | | | |

#### Table 2: Transition Probabilities
From Excel transition matrices, fill in the empty cells in supplementary document.

**Base Arm** (No Treatment):
| From State | To Base | To Post-Stroke Y1 | To Post-Stroke Y2+ | To Post-MI Y1 | To Post-MI Y2+ | To Post-Both Y1 | To Post-Both Y2+ | To Dead |
|------------|---------|-------------------|-------------------|---------------|----------------|-----------------|------------------|---------|
| Base | | | 0 | | 0 | 0 | 0 | |
| Post-Stroke Y1 | 0 | 0 | | 0 | 0 | | 0 | |
| ... | | | | | | | | |

**Treatment Arm**:
(Same table structure with treatment-adjusted probabilities)

---

## PART 9: Troubleshooting

### Common Issues:

#### "Macros Disabled" Error
**Solution**:
1. Close Excel
2. Open Excel Options/Preferences
3. Trust Center → Trust Center Settings → Macro Settings
4. Select "Enable all macros" (temporarily)
5. Reopen file

#### PSA Takes Too Long
**Options**:
1. Reduce iterations to 5,000 (less precise but faster)
2. Run overnight
3. Use more powerful computer
4. Check for infinite loops in VBA code

#### "Circular Reference" Warning
**Solution**:
- This may be intentional for iterative calculations
- Check Excel Options → Formulas → Enable iterative calculation
- Set max iterations: 100, max change: 0.001

#### Model Results Seem Implausible
**Checks**:
- [ ] Verify discount rate applied correctly
- [ ] Check hazard rates vs. probabilities (conversion formula: p = 1 - exp(-hazard))
- [ ] Confirm tunnel state logic working (Year 1 → Year 2+ transition)
- [ ] Validate against published CVD cost-effectiveness studies

---

## PART 10: Validation Checks

### Step 16: Cross-Check Results Against Literature

Compare your results to published studies:

1. **ICER Range**: Should be between £5,000 - £25,000/QALY based on periodontal-diabetes study [Reference 20]
2. **Incremental QALYs**: Expected 0.1 - 0.3 QALYs over 10 years
3. **Incremental Costs**: Expected £5,000 - £15,000 total
4. **Event Reduction**: Treatment should reduce stroke by ~20-40%, MI by ~15-35%

**Red Flags** (investigate if these occur):
- ICER < £0 (cost-saving and more effective - check signs)
- ICER > £50,000/QALY (unlikely to be cost-effective)
- Incremental QALYs < 0 (treatment worse than no treatment - error)
- All PSA iterations in single quadrant (insufficient parameter uncertainty)

---

## PART 11: Deliverables Checklist

### After completing all steps above, you should have:

#### Numerical Outputs:
- [ ] Base case ICER (£/QALY)
- [ ] Net monetary benefit at £20k and £30k thresholds
- [ ] PSA mean ICER with 95% CI
- [ ] Probability cost-effective at NICE thresholds
- [ ] One-way sensitivity results for all key parameters

#### Tables (for Results section):
- [ ] Table 1: Base case results (costs, QALYs, events by arm)
- [ ] Table 2: PSA summary statistics
- [ ] Table 3: One-way sensitivity analysis results
- [ ] Table 4: Complete parameter values (for Supplementary)
- [ ] Table 5: Transition probability matrices (for Supplementary)

#### Figures (for Results section):
- [ ] Figure 1: Cost-effectiveness plane (scatter plot)
- [ ] Figure 2: Cost-effectiveness acceptability curve
- [ ] Figure 3: Tornado diagram (one-way sensitivity)
- [ ] Figure 4: Markov trace (cohort distribution over time)

#### Raw Data Files:
- [ ] CVD_base_case_results.csv
- [ ] CVD_PSA_raw_data.csv
- [ ] CVD_one_way_sensitivity.csv
- [ ] CVD_markov_trace_no_treatment.csv
- [ ] CVD_markov_trace_treatment.csv

---

## PART 12: Next Steps After Model Execution

Once you have all the above outputs:

1. **Update Supplementary Material** (Week 2):
   - Fill all empty parameter tables
   - Complete transition probability matrices
   - Add CHEERS 2022 checklist locations
   - Insert sensitivity analysis ranges

2. **Write Results Section** (Week 3):
   - Base case findings
   - PSA interpretation
   - Sensitivity analysis discussion
   - Insert tables and figures

3. **Write Discussion** (Week 4):
   - Interpret ICER relative to NICE thresholds
   - Compare to periodontal-diabetes study (£1,474/QALY)
   - Policy implications
   - Limitations and future research

4. **Draft Abstract** (Week 4):
   - Background (2-3 sentences)
   - Methods (2-3 sentences)
   - Results (3-4 sentences with key numbers)
   - Conclusions (1-2 sentences)

---

## ESTIMATED TIME REQUIRED:

- **Pre-execution verification**: 15-30 minutes
- **Base case execution**: 1-2 minutes
- **PSA execution**: 30 minutes - 2 hours
- **One-way sensitivity**: 30-60 minutes
- **Extract outputs and figures**: 1-2 hours
- **Complete parameter tables**: 1-2 hours

**Total**: 4-6 hours of focused work

---

## Contact for Issues:

If you encounter technical problems with the Excel model that cannot be resolved using this guide, document:
1. Error message (screenshot)
2. Step where error occurred
3. Excel version and operating system
4. Parameter values being used

---

## Notes:
- Save the Excel file frequently during execution
- Keep a backup copy before running PSA
- Document any assumptions or modifications made to the model
- Record date and time of model execution for reproducibility
