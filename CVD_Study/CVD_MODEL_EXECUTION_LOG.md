# CVD Markov Model Execution Log
## Execution Date: November 26, 2025
## Model File: PD_CVD_markov - PSA On.xlsm (6.9 MB)

---

## EXECUTION STATUS TRACKER

- [ ] **PART 1**: Pre-execution checklist completed
- [ ] **PART 2**: Model structure verified
- [ ] **PART 3**: Base case executed
- [ ] **PART 4**: PSA executed (10,000 iterations)
- [ ] **PART 5**: One-way sensitivity analysis completed
- [ ] **PART 6**: Figures extracted
- [ ] **PART 7**: Raw data exported
- [ ] **PART 8**: Parameter tables completed

---

## PART 1: PRE-EXECUTION CHECKLIST

**System Requirements:**
- [x] Excel file opened (6.9 MB)
- [ ] Macros enabled (REQUIRED - verify in Excel)
- [x] 2GB+ free RAM available
- [x] Backup copy saved

**Next Action Required:**
👉 **Enable Macros in Excel when prompted**

---

## PART 2: MODEL STRUCTURE VERIFICATION

**Expected Worksheets (verify these tabs exist):**
- [ ] Parameters
- [ ] Base Case
- [ ] Treatment
- [ ] Results
- [ ] PSA
- [ ] PSA Results
- [ ] Figures

**Action Required:**
1. Look at the bottom tabs of the Excel file
2. Check off each worksheet as you verify it exists
3. Navigate to **Parameters** worksheet first

---

## PART 3: KEY PARAMETERS TO VERIFY

### Navigate to Parameters worksheet and record these values:

#### A. Baseline Hazards (Age 65):
- Base state → Stroke annual hazard: ____________
- Base state → MI annual hazard: ____________
- Background death probability: ____________ (should align with ONS)

#### B. Treatment Effect Hazard Ratios (from Supplementary_Material_CVD.md):
- [ ] Stroke HR: 0.59 (95% CI: 0.40-0.78) - verify in Excel
- [ ] MI HR: 0.72 (95% CI: 0.54-0.90) - verify in Excel

#### C. Annual Costs (2024 GBP):
**Periodontal Treatment:**
- Annual cost: £____________ (expected: £1,275)

**CVD Event Costs:**
- Stroke Year 1: £____________
- Stroke Year 2+: £____________
- MI Year 1: £____________
- MI Year 2+: £____________

#### D. Utility Values:
- Base (no CVD): ____________ (expected ~0.80-0.85)
- Post-stroke Year 1: ____________
- Post-stroke Year 2+: ____________
- Post-MI Year 1: ____________
- Post-MI Year 2+: ____________
- Post-both (multiplicative): ____________

#### E. Model Settings:
- [ ] Time horizon: ______ years (expected: 10 years)
- [ ] Discount rate (costs): ______ % (expected: 3.5%)
- [ ] Discount rate (QALYs): ______ % (expected: 3.5%)
- [ ] Starting age: ______ (expected: 65)
- [ ] Starting cohort size: ______ (expected: 1,000 or 10,000)

**Status:** Parameters verification - IN PROGRESS

---

## PART 4: BASE CASE EXECUTION

### Step 1: Run Base Case
**Instructions:**
1. Navigate to **Results** worksheet (or look for a "Control Panel" tab)
2. Find button labeled "Run Base Case" or "Calculate"
3. Click the button
4. Wait for completion (10-30 seconds)

**Status:** NOT STARTED

### Step 2: Extract Base Case Results

Once execution completes, record these values:

| Outcome | No Treatment | Treatment | Incremental |
|---------|--------------|-----------|-------------|
| **Total QALYs** (discounted) | | | |
| **Total Costs** (discounted, £) | | | |
| **Life Years** (undiscounted) | | | |
| **Stroke Events** (count) | | | |
| **MI Events** (count) | | | |
| **Deaths** (count) | | | |

### Step 3: Calculate Key Metrics

**ICER Calculation:**
```
Incremental Costs: £____________
Incremental QALYs: ____________
ICER = £____________ per QALY gained
```

**Expected Range:** £5,000 - £25,000/QALY (based on periodontal-diabetes literature)

**Net Monetary Benefit (£20,000/QALY threshold):**
```
NMB = (Incremental QALYs × £20,000) - Incremental Costs
NMB = £____________
```

**Net Monetary Benefit (£30,000/QALY threshold):**
```
NMB = (Incremental QALYs × £30,000) - Incremental Costs
NMB = £____________
```

**Interpretation:**
- [ ] NMB > 0 at £20K threshold → Cost-effective by NICE lower bound
- [ ] NMB > 0 at £30K threshold → Cost-effective by NICE upper bound

---

## PART 5: PROBABILISTIC SENSITIVITY ANALYSIS (PSA)

### Pre-PSA Checklist:
- [ ] Base case results recorded above
- [ ] Computer plugged in (if laptop)
- [ ] No other intensive programs running
- [ ] Save Excel file before starting PSA

### PSA Configuration Verification:

**Navigate to PSA worksheet and verify:**
- [ ] Number of iterations: 10,000
- [ ] Random seed: ______ (record for reproducibility)
- [ ] Parameter distributions:
  - [ ] Costs: Gamma distribution
  - [ ] Utilities: Beta distribution
  - [ ] Transition probabilities: Beta distribution

### Execute PSA:

**Instructions:**
1. Click "Run PSA" or "Monte Carlo Simulation" button
2. Note start time: ____________
3. **DO NOT CLOSE EXCEL DURING EXECUTION**
4. Monitor progress bar/counter
5. Note completion time: ____________
6. **SAVE FILE IMMEDIATELY AFTER COMPLETION**

**Expected Duration:** 30 minutes to 2 hours (depends on CPU speed)

**Status:** NOT STARTED

### PSA Results to Extract:

| Statistic | Incremental Costs (£) | Incremental QALYs |
|-----------|----------------------|-------------------|
| **Mean** | | |
| **Median** | | |
| **95% CI Lower** | | |
| **95% CI Upper** | | |
| **Std Deviation** | | |

**Cost-Effectiveness Probability:**
- Probability cost-effective at £20,000/QALY: ____________%
- Probability cost-effective at £30,000/QALY: ____________%

**Expected Results:**
- Should have >50% probability at £30K threshold
- Ideally >70% probability at £20K threshold (based on periodontal-diabetes study ICER of £1,474/QALY)

---

## PART 6: ONE-WAY SENSITIVITY ANALYSIS

### Parameters to Test:

For each parameter below, vary between min/max and record the resulting ICER:

| Parameter | Base | Min | ICER at Min | Max | ICER at Max | Range |
|-----------|------|-----|-------------|-----|-------------|-------|
| **Stroke HR** | 0.59 | 0.40 | | 0.78 | | |
| **MI HR** | 0.72 | 0.54 | | 0.90 | | |
| **Periodontal cost** | £1,275 | £1,020 | | £1,530 | | |
| **Stroke Y1 cost** | | -20% | | +20% | | |
| **MI Y1 cost** | | -20% | | +20% | | |
| **Post-stroke utility** | | CI lower | | CI upper | | |
| **Post-MI utility** | | CI lower | | CI upper | | |
| **Discount rate** | 3.5% | 1.5% | | 5% | | |

**Status:** NOT STARTED

**Instructions:**
1. Most models have a sensitivity analysis section/button
2. Or manually change each parameter and re-run base case
3. Record ICER for each parameter variation
4. Calculate "Range" = |ICER at Max - ICER at Min|
5. Rank parameters by Range (highest = most influential)

---

## PART 7: FIGURES TO EXTRACT

### Figure 1: Cost-Effectiveness Plane
- [ ] Located scatter plot showing 10,000 PSA iterations
- [ ] X-axis: Incremental QALYs
- [ ] Y-axis: Incremental Costs (£)
- [ ] WTP threshold lines visible (£20K, £30K)
- [ ] **Exported as:** CVD_CE_Plane.png (600+ DPI)
- [ ] **Saved to:** `/Users/rezarassool/Source/periodontal/figures/`

**Visual Check:**
- Most points should be in NE quadrant (more costly, more effective)
- Points should cross £20K and £30K threshold lines

### Figure 2: Cost-Effectiveness Acceptability Curve (CEAC)
- [ ] Located CEAC showing probability curve
- [ ] X-axis: WTP threshold (£0 to £50,000)
- [ ] Y-axis: Probability cost-effective (0% to 100%)
- [ ] **Exported as:** CVD_CEAC.png (600+ DPI)
- [ ] **Saved to:** `/Users/rezarassool/Source/periodontal/figures/`

**Key Values from CEAC:**
- Probability at £20K: ____________%
- Probability at £30K: ____________%

### Figure 3: Tornado Diagram
- [ ] Located tornado diagram for one-way sensitivity
- [ ] Parameters ranked by influence on ICER
- [ ] **Exported as:** CVD_Tornado.png (600+ DPI)
- [ ] **Saved to:** `/Users/rezarassool/Source/periodontal/figures/`

**Most Influential Parameters (record top 5):**
1. ____________ (Range: £______)
2. ____________ (Range: £______)
3. ____________ (Range: £______)
4. ____________ (Range: £______)
5. ____________ (Range: £______)

### Figure 4: Markov Trace
- [ ] Located cohort distribution over time table
- [ ] **Exported as:** CVD_Markov_Trace.png (600+ DPI)
- [ ] **Saved to:** `/Users/rezarassool/Source/periodontal/figures/`

**Status:** NOT STARTED

---

## PART 8: RAW DATA EXPORTS

### CSV Files to Create:

1. **Base case results:**
   - [ ] File: `CVD_base_case_results.csv`
   - [ ] Location: `/Users/rezarassool/Source/periodontal/data/`

2. **PSA raw data:**
   - [ ] File: `CVD_PSA_raw_data.csv`
   - [ ] Columns: Incremental_Costs, Incremental_QALYs
   - [ ] Rows: 10,000 iterations

3. **One-way sensitivity:**
   - [ ] File: `CVD_one_way_sensitivity.csv`

4. **Markov trace:**
   - [ ] File: `CVD_markov_trace_no_treatment.csv`
   - [ ] File: `CVD_markov_trace_treatment.csv`

**Status:** NOT STARTED

---

## PART 9: VALIDATION CHECKS

### Cross-Check Against Literature:

**Expected Ranges (from periodontal-CVD and periodontal-diabetes literature):**

| Metric | Expected Range | Your Result | ✓/✗ |
|--------|---------------|-------------|-----|
| **ICER** | £5,000 - £25,000/QALY | £______ | |
| **Incremental QALYs** | 0.1 - 0.3 over 10 years | ______ | |
| **Incremental Costs** | £5,000 - £15,000 | £______ | |
| **Stroke reduction** | 20-40% | ______% | |
| **MI reduction** | 15-35% | ______% | |

**Red Flags to Investigate:**
- [ ] ICER < £0 (dominated - check calculation signs)
- [ ] ICER > £50,000/QALY (unlikely to be cost-effective)
- [ ] Incremental QALYs < 0 (treatment worse - ERROR)
- [ ] All PSA iterations in single quadrant (insufficient uncertainty)

**Status:** Awaiting results

---

## PART 10: NEXT STEPS AFTER EXECUTION

Once all data is extracted:

### Week 2 Tasks:
- [ ] Update `Supplementary_Material_CVD.md` with all parameter values
- [ ] Complete transition probability matrices
- [ ] Add sensitivity analysis ranges

### Week 3 Tasks:
- [ ] Write Results section (`Results_CVD.md`)
- [ ] Insert base case table
- [ ] Insert PSA summary table
- [ ] Insert all 4 figures
- [ ] Describe sensitivity analysis findings

### Week 4 Tasks:
- [ ] Write Discussion section (`Discussion_CVD.md`)
- [ ] Interpret ICER relative to NICE thresholds (£20K-£30K)
- [ ] Compare to periodontal-diabetes study (£1,474/QALY)
- [ ] Discuss policy implications
- [ ] Address limitations

### Week 5 Tasks:
- [ ] Draft Abstract
- [ ] Co-author review
- [ ] Final revisions
- [ ] Submit for publication

---

## NOTES AND OBSERVATIONS

### Execution Notes:
(Record any issues, warnings, or unusual observations during execution)

-
-
-

### Technical Issues:
(Document any errors and how they were resolved)

-
-
-

### Questions for Discussion:
(Note any unexpected results or interpretation questions)

-
-
-

---

## ESTIMATED TIME TRACKING

| Task | Estimated Time | Actual Time | Status |
|------|---------------|-------------|--------|
| Pre-execution verification | 15-30 min | | |
| Base case execution | 1-2 min | | |
| PSA execution | 30 min - 2 hrs | | |
| One-way sensitivity | 30-60 min | | |
| Extract outputs/figures | 1-2 hrs | | |
| Complete parameter tables | 1-2 hrs | | |
| **TOTAL** | **4-6 hours** | | |

---

## CONTACT INFORMATION

**For technical Excel issues:**
- Document error message (screenshot)
- Note step where error occurred
- Record Excel version and macOS version
- Save current parameter values

**Last Updated:** November 26, 2025
**Executed By:** [Your Name]
**Model Version:** PD_CVD_markov - PSA On.xlsm (6.9 MB)
