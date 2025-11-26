# Results Section Template
## Cost-Effectiveness of Periodontal Therapy to Prevent Severe Cardiovascular Events

### Instructions:
Fill in all [PLACEHOLDER] fields with actual values from Excel model execution.
Target word count: 2,000-3,000 words

---

## RESULTS

### Base Case Analysis

#### Cohort Characteristics and Model Assumptions
We modeled a cohort of [PLACEHOLDER: cohort size, e.g., 100,000] 65-year-old males with severe periodontal disease over a 10-year time horizon. The cohort entered the model in the base state (no prior CVD) and could transition to post-stroke, post-myocardial infarction (MI), post-both events, or death annually. [PLACEHOLDER: Add any relevant population characteristics or initial distribution details from Excel model].

#### Health Outcomes

Over the 10-year time horizon, individuals receiving non-surgical periodontal therapy accumulated [PLACEHOLDER: X.XX] quality-adjusted life years (QALYs) compared to [PLACEHOLDER: X.XX] QALYs in the untreated group, yielding an incremental gain of [PLACEHOLDER: 0.XX] QALYs (95% CI: [PLACEHOLDER: 0.XX - 0.XX]). Life years gained were [PLACEHOLDER: X.XX] in the treatment arm versus [PLACEHOLDER: X.XX] in the no-treatment arm (incremental: [PLACEHOLDER: 0.XX] years).

The treatment arm experienced [PLACEHOLDER: XXX] stroke events compared to [PLACEHOLDER: XXX] events in the untreated arm, representing a [PLACEHOLDER: XX%] relative reduction. Similarly, MI events were reduced from [PLACEHOLDER: XXX] in the untreated group to [PLACEHOLDER: XXX] in the treatment group ([PLACEHOLDER: XX%] relative reduction). By the end of the 10-year horizon, [PLACEHOLDER: XX%] of the untreated cohort had died compared to [PLACEHOLDER: XX%] in the treatment arm.

**[INSERT Table 1: Base Case Results - Costs, QALYs, and Clinical Outcomes by Treatment Arm]**

| Outcome | No Treatment | Treatment | Incremental | 95% CI |
|---------|--------------|-----------|-------------|--------|
| **Total QALYs** (discounted) | [X.XX] | [X.XX] | [0.XX] | [0.XX - 0.XX] |
| **Total Costs** (£, discounted) | [£XX,XXX] | [£XX,XXX] | [£X,XXX] | [£X,XXX - £XX,XXX] |
| **Life Years** (undiscounted) | [X.XX] | [X.XX] | [0.XX] | [0.XX - 0.XX] |
| **Stroke Events** (n) | [XXX] | [XXX] | [-XX] | [-XX - -XX] |
| **MI Events** (n) | [XXX] | [XXX] | [-XX] | [-XX - -XX] |
| **Deaths** (n) | [XXX] | [XXX] | [-XX] | [-XX - -XX] |

#### Cost Outcomes

Total discounted costs over 10 years were £[PLACEHOLDER: XX,XXX] in the treatment arm compared to £[PLACEHOLDER: XX,XXX] in the no-treatment arm, resulting in incremental costs of £[PLACEHOLDER: X,XXX] (95% CI: £[PLACEHOLDER: X,XXX - XX,XXX]).

The annual cost of non-surgical periodontal therapy was £1,275 per patient, comprising initial treatment (£530.75, amortized), maintenance therapy every six months (£35.90 per visit), re-treatment (£265.37), extractions (£71.79), resin-bonded bridges (£382.04), and removable partial dentures (£484.60). Over 10 years, the cumulative periodontal treatment costs totaled £[PLACEHOLDER: XX,XXX] per patient.

CVD event costs were substantially higher in the untreated arm. First-year stroke costs averaged £[PLACEHOLDER: XX,XXX] per event, declining to £[PLACEHOLDER: X,XXX] annually in subsequent years. First-year MI costs were £[PLACEHOLDER: XX,XXX] per event, with subsequent annual costs of £[PLACEHOLDER: X,XXX]. For individuals experiencing both stroke and MI, we applied the higher post-stroke costs to avoid double-counting, totaling £[PLACEHOLDER: XX,XXX] in the first year and £[PLACEHOLDER: X,XXX] annually thereafter.

**[INSERT Table 2: Breakdown of Costs by Category and Treatment Arm]**

| Cost Category | No Treatment (£) | Treatment (£) | Incremental (£) |
|---------------|------------------|---------------|-----------------|
| **Periodontal therapy** | 0 | [X,XXX] | [+X,XXX] |
| **Stroke events (acute)** | [XX,XXX] | [X,XXX] | [-X,XXX] |
| **Stroke events (chronic)** | [XX,XXX] | [XX,XXX] | [-X,XXX] |
| **MI events (acute)** | [XX,XXX] | [X,XXX] | [-X,XXX] |
| **MI events (chronic)** | [XX,XXX] | [XX,XXX] | [-X,XXX] |
| **Total** | [XX,XXX] | [XX,XXX] | [X,XXX] |

#### Incremental Cost-Effectiveness Ratio

The incremental cost-effectiveness ratio (ICER) for non-surgical periodontal therapy was £[PLACEHOLDER: XX,XXX] per QALY gained (95% CI: £[PLACEHOLDER: XX,XXX - XX,XXX]). This falls [PLACEHOLDER: below/within/above] the UK National Institute for Health and Care Excellence (NICE) willingness-to-pay threshold range of £20,000-£30,000 per QALY.

Net monetary benefit (NMB) at the £20,000 per QALY threshold was £[PLACEHOLDER: X,XXX] (95% CI: £[PLACEHOLDER: -X,XXX to XX,XXX]), indicating [PLACEHOLDER: the intervention is/is not] cost-effective at this threshold. At the £30,000 per QALY threshold, NMB was £[PLACEHOLDER: XX,XXX] (95% CI: £[PLACEHOLDER: X,XXX to XX,XXX]), [PLACEHOLDER: further supporting/questioning] cost-effectiveness.

---

### Cohort Distribution Over Time

Figure [X] illustrates the Markov trace showing the proportion of the cohort in each health state over the 10-year time horizon. In the untreated arm, the proportion in the base state (no prior CVD) declined from 100% at baseline to [PLACEHOLDER: XX%] at year 10. The post-stroke states (combined Year 1 and Year 2+) comprised [PLACEHOLDER: XX%] of the cohort by year 10, while post-MI states comprised [PLACEHOLDER: XX%], and post-both events [PLACEHOLDER: X%]. Cumulative mortality reached [PLACEHOLDER: XX%] by year 10.

In the treatment arm, [PLACEHOLDER: XX%] remained in the base state at year 10, with [PLACEHOLDER: XX%] in post-stroke states, [PLACEHOLDER: XX%] in post-MI states, [PLACEHOLDER: X%] in post-both states, and [PLACEHOLDER: XX%] deceased. The treatment group maintained a higher proportion in the base state throughout the time horizon, reflecting the protective effect of periodontal therapy.

**[INSERT Figure 1: Markov Trace - Cohort Distribution Over 10 Years]**
- Stacked area chart showing proportion in each health state (0-100%) on Y-axis
- Time (years 0-10) on X-axis
- Separate panels for No Treatment vs. Treatment arms
- Color-coded areas: Base (green), Post-Stroke Y1 (red), Post-Stroke Y2+ (pink), Post-MI Y1 (orange), Post-MI Y2+ (yellow), Post-Both Y1 (purple), Post-Both Y2+ (lavender), Dead (black)

---

### Probabilistic Sensitivity Analysis

To account for parameter uncertainty, we conducted a probabilistic sensitivity analysis (PSA) with 10,000 Monte Carlo simulations. Each simulation randomly sampled parameter values from their respective distributions: gamma distributions for costs, beta distributions for utilities and transition probabilities, and log-normal distributions for hazard ratios.

#### PSA Results Summary

The PSA yielded a mean ICER of £[PLACEHOLDER: XX,XXX] per QALY (95% credible interval: £[PLACEHOLDER: XX,XXX - XX,XXX]), closely aligned with the deterministic base case result (£[PLACEHOLDER: XX,XXX] per QALY). Mean incremental costs were £[PLACEHOLDER: X,XXX] (95% CI: £[PLACEHOLDER: X,XXX - XX,XXX]) and mean incremental QALYs were [PLACEHOLDER: 0.XX] (95% CI: [PLACEHOLDER: 0.XX - 0.XX]).

**[INSERT Table 3: Probabilistic Sensitivity Analysis Summary Statistics]**

| Parameter | Mean | Median | 95% Credible Interval | Standard Deviation |
|-----------|------|--------|----------------------|-------------------|
| **Incremental Costs (£)** | [X,XXX] | [X,XXX] | [X,XXX - XX,XXX] | [X,XXX] |
| **Incremental QALYs** | [0.XX] | [0.XX] | [0.XX - 0.XX] | [0.XX] |
| **ICER (£/QALY)** | [XX,XXX] | [XX,XXX] | [XX,XXX - XX,XXX] | [X,XXX] |

#### Cost-Effectiveness Plane

Figure [X] presents the cost-effectiveness plane plotting the 10,000 PSA iterations. The majority of simulations ([PLACEHOLDER: XX%]) fell in the northeast quadrant, indicating higher costs and greater effectiveness for periodontal therapy. [PLACEHOLDER: XX%] of iterations were in the southeast quadrant (dominant: lower costs and greater effectiveness), [PLACEHOLDER: X%] in the northwest quadrant (dominated: higher costs and lower effectiveness), and [PLACEHOLDER: X%] in the southwest quadrant (lower costs and lower effectiveness).

At the £20,000 per QALY threshold (represented by the solid diagonal line), [PLACEHOLDER: XX%] of iterations fell below the line, indicating cost-effectiveness at this threshold. At the £30,000 per QALY threshold (dashed line), [PLACEHOLDER: XX%] of iterations were cost-effective.

**[INSERT Figure 2: Cost-Effectiveness Plane]**
- Scatter plot with Incremental QALYs on X-axis (-0.X to 0.X)
- Incremental Costs (£) on Y-axis (-£XX,XXX to £XX,XXX)
- 10,000 simulation points (semi-transparent dots)
- Willingness-to-pay threshold lines: £20,000/QALY (solid), £30,000/QALY (dashed)
- Origin (0,0) marked with crosshairs
- Quadrant labels: NE (more effective, more costly), SE (more effective, less costly), NW (less effective, more costly), SW (less effective, less costly)

#### Cost-Effectiveness Acceptability Curve

The cost-effectiveness acceptability curve (CEAC) in Figure [X] illustrates the probability that non-surgical periodontal therapy is cost-effective across a range of willingness-to-pay thresholds from £0 to £50,000 per QALY.

At a threshold of £0/QALY (intervention must be cost-saving), the probability of cost-effectiveness was [PLACEHOLDER: X%]. This probability increased to [PLACEHOLDER: XX%] at £10,000/QALY, [PLACEHOLDER: XX%] at £20,000/QALY (NICE lower threshold), and [PLACEHOLDER: XX%] at £30,000/QALY (NICE upper threshold). At £40,000/QALY, the probability reached [PLACEHOLDER: XX%], and at £50,000/QALY it was [PLACEHOLDER: XX%].

The CEAC demonstrates [PLACEHOLDER: strong/moderate/weak] evidence for cost-effectiveness at the NICE threshold range. The curve [PLACEHOLDER: crosses/does not cross] the 50% probability threshold at £[PLACEHOLDER: XX,XXX]/QALY, indicating that this is the threshold at which the intervention has an equal chance of being cost-effective or not.

**[INSERT Figure 3: Cost-Effectiveness Acceptability Curve]**
- Line graph with Willingness-to-Pay Threshold (£/QALY) on X-axis (£0 - £50,000)
- Probability Cost-Effective (%) on Y-axis (0% - 100%)
- Smooth curve showing increasing probability with higher WTP
- Vertical reference lines at £20,000 and £30,000 (NICE thresholds)
- Horizontal reference line at 50% probability
- Annotations showing probability at key thresholds

---

### One-Way Sensitivity Analysis

We conducted deterministic one-way sensitivity analyses by varying each key parameter between its minimum and maximum plausible values (typically ±20% for costs or 95% confidence interval bounds for clinical parameters) while holding all other parameters at base case values.

#### Most Influential Parameters

The tornado diagram in Figure [X] ranks parameters by their impact on the ICER. The five most influential parameters were:

1. **Stroke hazard ratio (treatment effect)**: Varying from 0.40 (optimistic) to 0.78 (conservative), the ICER ranged from £[PLACEHOLDER: XX,XXX] to £[PLACEHOLDER: XX,XXX] per QALY. [PLACEHOLDER: The intervention remained cost-effective across the entire range / became not cost-effective at the conservative estimate].

2. **MI hazard ratio (treatment effect)**: Ranging from 0.54 to 0.90, the ICER varied between £[PLACEHOLDER: XX,XXX] and £[PLACEHOLDER: XX,XXX] per QALY. [PLACEHOLDER: This parameter had the largest/second-largest impact on cost-effectiveness].

3. **Post-stroke Year 1 utility**: Using 95% CI bounds, the ICER ranged from £[PLACEHOLDER: XX,XXX] to £[PLACEHOLDER: XX,XXX] per QALY, [PLACEHOLDER: description of impact].

4. **Periodontal treatment annual cost**: Varying ±20% (£1,020 - £1,530), the ICER ranged from £[PLACEHOLDER: XX,XXX] to £[PLACEHOLDER: XX,XXX] per QALY.

5. **Stroke Year 1 acute cost**: At -20% (£[PLACEHOLDER: XX,XXX]), the ICER was £[PLACEHOLDER: XX,XXX]/QALY; at +20% (£[PLACEHOLDER: XX,XXX]), it was £[PLACEHOLDER: XX,XXX]/QALY.

**[INSERT Table 4: One-Way Sensitivity Analysis Results]**

| Parameter | Base Case | Lower Value | ICER at Lower | Upper Value | ICER at Upper | Range |
|-----------|-----------|-------------|---------------|-------------|---------------|-------|
| Stroke HR | 0.59 | 0.40 | £[XX,XXX] | 0.78 | £[XX,XXX] | £[XX,XXX] |
| MI HR | 0.72 | 0.54 | £[XX,XXX] | 0.90 | £[XX,XXX] | £[XX,XXX] |
| Post-stroke Y1 utility | [0.XX] | [0.XX] | £[XX,XXX] | [0.XX] | £[XX,XXX] | £[XX,XXX] |
| Periodontal cost | £1,275 | £1,020 | £[XX,XXX] | £1,530 | £[XX,XXX] | £[X,XXX] |
| Stroke Y1 cost | £[XX,XXX] | £[XX,XXX] | £[XX,XXX] | £[XX,XXX] | £[XX,XXX] | £[X,XXX] |
| MI Y1 cost | £[XX,XXX] | £[XX,XXX] | £[XX,XXX] | £[XX,XXX] | £[XX,XXX] | £[X,XXX] |
| Post-MI Y1 utility | [0.XX] | [0.XX] | £[XX,XXX] | [0.XX] | £[XX,XXX] | £[X,XXX] |
| Discount rate | 3.5% | 1.5% | £[XX,XXX] | 5.0% | £[XX,XXX] | £[X,XXX] |

**[INSERT Figure 4: Tornado Diagram - One-Way Sensitivity Analysis]**
- Horizontal bar chart with ICER (£/QALY) on X-axis
- Parameters ranked by range on Y-axis (largest impact at top)
- Each parameter has two bars extending from base case value:
  - Left bar (blue): Impact of lower parameter value
  - Right bar (red): Impact of upper parameter value
- Vertical reference line at base case ICER
- Vertical reference lines at £20,000 and £30,000 (NICE thresholds)

#### Threshold Analysis

[PLACEHOLDER: Optional - add if conducted]
Threshold analysis identified the critical value at which the intervention is no longer cost-effective at the £30,000/QALY threshold:
- The stroke hazard ratio would need to exceed [PLACEHOLDER: 0.XX] for the ICER to surpass £30,000/QALY
- The MI hazard ratio would need to exceed [PLACEHOLDER: 0.XX]
- Periodontal treatment annual costs would need to exceed £[PLACEHOLDER: X,XXX]

---

### Scenario Analyses

[PLACEHOLDER: Add if you ran alternative scenarios]

#### Scenario 1: [Description, e.g., "Extended Time Horizon (20 years)"]
[Results summary]

#### Scenario 2: [Description, e.g., "Female Cohort"]
[Results summary]

#### Scenario 3: [Description, e.g., "Younger Cohort (Age 55)"]
[Results summary]

---

### Summary of Findings

Non-surgical periodontal therapy for 65-year-old males with severe periodontal disease yielded an ICER of £[PLACEHOLDER: XX,XXX] per QALY gained over a 10-year time horizon from the NHS perspective. The intervention generated [PLACEHOLDER: 0.XX] additional QALYs at an incremental cost of £[PLACEHOLDER: X,XXX], primarily driven by reduced stroke and MI events. Probabilistic sensitivity analysis demonstrated [PLACEHOLDER: XX%] probability of cost-effectiveness at the £20,000/QALY threshold and [PLACEHOLDER: XX%] at £30,000/QALY. One-way sensitivity analysis indicated that treatment effect hazard ratios for stroke and MI were the most influential parameters, but the intervention remained cost-effective across plausible parameter ranges. These findings suggest that non-surgical periodontal therapy [PLACEHOLDER: is/may be/is not] a cost-effective strategy for CVD prevention in this population.

---

## FIGURES TO CREATE:

1. **Figure 1: Markov Trace (Cohort Distribution Over Time)**
   - Stacked area chart
   - 8 health states over 10 years
   - Side-by-side panels: No Treatment vs Treatment

2. **Figure 2: Cost-Effectiveness Plane**
   - Scatter plot of 10,000 PSA iterations
   - Incremental costs vs incremental QALYs
   - WTP threshold lines at £20k and £30k

3. **Figure 3: Cost-Effectiveness Acceptability Curve**
   - Probability cost-effective (0-100%) vs WTP threshold (£0-£50k)
   - Annotations at NICE thresholds

4. **Figure 4: Tornado Diagram**
   - Horizontal bars showing ICER range for each parameter
   - Ranked by magnitude of impact
   - NICE threshold reference lines

---

## TABLES TO CREATE:

1. **Table 1: Base Case Results** (Costs, QALYs, Events by Arm)
2. **Table 2: Cost Breakdown by Category**
3. **Table 3: PSA Summary Statistics**
4. **Table 4: One-Way Sensitivity Analysis Results**

---

## WRITING TIPS:

1. **Be precise with numbers**: Always include 95% CIs where applicable
2. **Compare to benchmarks**: Reference NICE thresholds (£20k-£30k/QALY) and the periodontal-diabetes study (£1,474/QALY) throughout
3. **Interpret, don't just report**: Explain what each finding means for decision-makers
4. **Use active voice**: "The intervention reduced stroke events by X%" rather than "Stroke events were reduced by X%"
5. **Signpost**: Use clear headings and transitions between subsections
6. **Cross-reference**: "As shown in Table X" or "Figure X illustrates"
7. **Avoid repetition**: If a number is in a table, describe the pattern/trend in text rather than repeating all values

---

## WORD COUNT TARGET:
- Base Case Analysis: 600-800 words
- Cohort Distribution: 200-300 words
- Probabilistic Sensitivity Analysis: 600-800 words
- One-Way Sensitivity Analysis: 400-600 words
- Summary: 200-300 words
- **Total: 2,000-3,000 words**
