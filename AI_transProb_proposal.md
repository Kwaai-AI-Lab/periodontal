**STUDY PROTOCOL**

AI-Derived Transition Probabilities for Periodontal Disease Markov Models: A Cross-Country Validation Study Using Longitudinal Cohort Data from the UK and US

# SECTION 1. BACKGROUND AND RATIONALE

**1.1 The Transition Probability Problem in Periodontal Economic Modelling**

Markov and microsimulation models are the standard analytical framework for evaluating the cost-effectiveness of periodontal interventions. These models simulate patient health through defined health states, typically through gingivitis, mild, moderate, and severe periodontitis, to edentulism, accumulating costs and quality-adjusted life years (QALYs) at each cycle.

The validity of these models depends on the accuracy of transition probabilities between disease states within a defined cycle length. These probabilities have been derived from one of three approaches, each with important limitations:

1.  Cross-sectional prevalence data back-calculated into transition rates. This is methodologically inconsistent, as cross-sectional studies were not designed to estimate state transitions, and the conversion introduces systematic bias when disease states are not in steady state.

2.  Small clinical trials with short follow-up periods. These provide high internal validity but limited external generalisability, particularly for the elderly populations most relevant to health economic decisions about prevention and prosthetic intervention.

3.  Expert elicitation. Used when longitudinal data are unavailable, but introduces subjective uncertainty that is poorly captured by standard probabilistic sensitivity analysis.

A consequence of these limitations is that published ICERs for periodontal interventions show substantial variability that may reflect heterogeneity in input parameters rather than true differences in intervention effectiveness. The Needleman et al. (2018) systematic review of periodontal disease progression documented mean annual attachment loss of 0.1 mm (95% CI: 0.07, 0.13), with a 99% I-squared statistic. This indicates that population-average figures mask enormous individual-level variability that current models fail to capture. Existing models treat transition probabilities as fixed population-level parameters, ignoring covariate-specific heterogeneity. A patient with severe periodontitis who smokes, has type 2 diabetes, and shows a high systemic inflammatory burden has a materially different risk of tooth loss than a patient with the same probing depths who is normoglycemic and a non-smoker. This heterogeneity is clinically well-established but rarely incorporated into economic models in a principled, data-driven way.

**1.2 The Data Integration Problem**

A further limitation specific to this disease area is that the relevant longitudinal evidence base is fragmented across datasets that were not designed for integration. The most informative sources --- multi-state Markov analyses of the Goodson dataset (Mdala et al., 2014), the Sri Lanka natural history cohort (Ramseier et al., 2017), and the Needleman systematic review (2018) --- differ in their measurement protocols, follow-up durations, outcome definitions, and study populations in ways that preclude direct pooling.

Three categories of incompatibility must be resolved before any unified model can be constructed:

1.  Measurement protocol heterogeneity: Studies vary between manual CAL measurement, CPITN-based grading, and electronic probing. Reported transition intensities are not directly comparable without standardisation or calibration offsets derived from published crosswalk literature.

2.  Outcome definition heterogeneity: Electronic health record sources such as MIMIC-IV use ICD-10 diagnostic codes (K05.x) rather than clinical periodontal measurements. These sources cannot be used to assign patients to clinical disease states directly and must instead be limited to estimating comorbidity-specific modifiers of progression, cross walked via the published ICD-K05.x to periodontal staging literature.

3.  Temporal scale heterogeneity: Follow-up durations range from one-year clinical trials to the forty-year Sri Lanka natural history cohort. Direct pooling of transition rates across these time scales is not valid. Reconciliation requires conversion of all reported parameters to annual transition intensities using continuous-time multi-state Markov methods (implemented via the pymsm Python package), ensuring comparability regardless of the original study duration.

Rather than pooling these sources directly this study implements a structured three-layer data integration framework that assigns each source a defined role based on its methodological strengths, resolves incompatibilities explicitly, and reconciles the outputs into a coherent synthetic modelling dataset.

**1.3 Machine Learning Survival Models as a Solution**

Recent methodological advances in machine learning for time-to-event data offer a principled solution to this problem. Rather than estimating single population-average transition rates from cross-sectional data, survival ML models can learn covariate-specific transition probabilities from longitudinal cohort data, capturing non-linear interactions between risk factors and disease progression that traditional parametric survival models cannot. Three approaches are particularly relevant to this application:

1.  Random Survival Forests (RSF) and their extension to longitudinal endogenous predictors via the DynForest framework (Devaux et al., 2023) can estimate individual-level event probabilities by internally transforming time-dependent predictors using mixed models at each tree node, with final predictions computed as Aalen-Johansen estimators averaged across trees.

2.  DeepSurv and Dynamic-DeepHit extend neural network architectures to competing risks and longitudinal data, learning the joint distribution of event times without parametric assumptions. Dynamic-DeepHit has demonstrated superior discriminative performance over Cox-based benchmarks in cystic fibrosis and other chronic disease datasets.

3.  Multi-state Markov models with ML-derived covariates, using the continuous-time pymsm framework in Python, allow direct estimation of transition intensities as functions of observed covariates while preserving the Markov structure required for downstream economic modelling.

The key methodological contribution of applying these methods to periodontal disease is the translation of ML survival outputs into structured transition probability matrices with covariate stratification. Rather than a single transition probability from moderate to severe periodontitis, the model produces a function over patient characteristics --- enabling individualised risk profiles that improve both clinical decision support and economic model validity.

**1.4 The Cross-Country Opportunity**

A dual-country design --- UK and US --- offers distinct and complementary advantages beyond simply increasing sample size. The two settings differ systematically in healthcare system structure, dietary and lifestyle exposures, cost frameworks, and willingness-to-pay thresholds. Demonstrating that AI-derived transition probabilities produce materially different ICERs across countries, even controlling for cost framework differences, would be a significant finding for the international health economics literature. AI models trained on one country\'s longitudinal data and validated on the other provide a stringent external validity test that purely domestic studies cannot offer.

# SECTION 2. LONGITUDINAL DATA SOURCES

**2.1 Overview and Data Integration Framework**

The commercial affiliation of this study restricts access to several major longitudinal datasets, including NHANES, ELSA, UK Biobank, and All of Us, which prohibit commercial use under their data access agreements. The study therefore draws on three complementary data source categories that are either published in the open literature with sufficient granularity for parameter extraction, available under permissive licences compatible with commercial research, or accessible via academic collaboration with data custodians.

Critically, these sources are not pooled directly. Each source presents methodological characteristics that are incompatible with the others in terms of measurement protocols, follow-up durations, outcome definitions, and study populations. Direct pooling would introduce systematic bias. Instead, the study implements a structured three-layer data integration framework.

Layer 1 --- Core disease progression: Published longitudinal cohort evidence (Mdala et al. 2014; Ramseier et al. 2017; Needleman et al. 2018; DMHDS) defines the transition intensities and their covariate relationships across the five-state disease continuum. All parameters are converted to annual transition intensities using continuous-time multi-state methods to ensure temporal comparability.

Layer 2 --- Covariate-specific modifiers: Electronic health record data (MIMIC-IV) provides comorbidity-specific modifiers of progression --- particularly the diabetes-periodontitis interaction and the periodontal-CVD pathway --- that are unavailable at the required precision from published longitudinal cohorts. MIMIC-IV data are used exclusively for modifier estimation, not for direct disease state assignment, and are crosswalked to clinical staging via published ICD-K05.x literature.

Layer 3 --- Synthetic reconciliation: Published parameters from Layers 1 and 2 are used to generate a synthetic longitudinal population using the SDV (Synthetic Data Vault) library in Python. This preserves joint distributions, longitudinal correlation structures, and covariate effects reported in the source literature while resolving the temporal and measurement incompatibilities between sources. Synthetic data generation is validated by comparing marginal distributions and key bivariate associations against published summary statistics.

This framework enables a controlled comparison between conventional literature-derived and AI-derived transition probabilities within an identical economic model structure, while providing a transparent audit trail for each parameter\'s origin and the rationale for how incompatibilities were resolved.

**2.2 Primary Source: Published Longitudinal Cohort Data**

Mdala et al. (2014) --- Goodson/Massachusetts Dataset

The primary methodological precedent for this study is the multi-state Markov analysis by Mdala et al. (2014, Journal of Clinical Periodontology, doi: 10.1111/jcpe.12278), which applied continuous-time Markov models to CAL and pocket depth data from the Goodson dataset. This dataset involved patients with chronic periodontitis, with detailed CAL, probing depth, and bleeding-on-probing measurements at multiple visits. Mdala et al. define a three-state model (State 1: health, State 2: gingivitis, State 3: chronic periodontitis) and report predicted transition probabilities with 95% confidence intervals for both CAL + BOP and PD + BOP classified models (Table 2). These are annual transition probabilities, not intensities; conversion to annual transition intensities for the present study uses q = - ln(1 minus p). The three-state Mdala framework maps onto the five-state model used here as follows: Mdala State 1 corresponds to S1 (periodontal health), Mdala State 2 to S2 (gingivitis/Stage I), and Mdala State 3 spans both S3 (Stage II) and S4 (Stage III/IV). The Mdala 1 to 3 transition (direct progression from health to chronic periodontitis, bypassing gingivitis) is not modelled as a direct transition in the present five-state structure; this probability mass is instead distributed across q12 and q23 sequentially. These parameters serve as the primary source for transition probability parameters TP-03 (q12) and the combined q23/q34 pathway in the US arm of the model. Separation of q23 and q34 requires supplementary sources or distributional assumptions, discussed below.

The Sri Lanka Natural History Study --- Untreated Periodontitis

Ramseier et al. (2017, Journal of Clinical Periodontology, doi: 10.1111/jcpe.12782) published 40-year follow-up data from the original natural history cohort of Sri Lankan tea labourers. This remains the definitive dataset for untreated periodontal disease progression, with tooth loss over 40 years ranging from 0 to 28 teeth (mean 13.1). Annual transition intensity for the Stage III/IV to edentulism transition (TP-06, q45) is back-calculated from cumulative tooth loss data using q = minus ln(1 minus p) divided by t. Logistic regression parameters for attachment loss as a predictor of tooth loss (TP-12) are published with sufficient detail to parameterise the disease-to-tooth-loss transition. This dataset is used exclusively for later-stage transitions and tooth loss parameters; the Mdala et al. dataset is preferred for earlier-stage transitions given its clinical treatment context.

Dunedin Multidisciplinary Health and Development Study (DMHDS)

The DMHDS is the only birth cohort in the world to have clinically investigated dental health from birth to midlife, with periodontal assessments from age 26 onwards across multiple waves. As a New Zealand study generalisable to the UK and US context, it anchors life-course periodontal trajectories in the synthetic population. Data access requires a concept paper and institutional ethics approval. Given the ethics approval timeline of typically 3 to 6 months, the concept paper process should be initiated during Phase 1 in parallel with literature extraction.

Needleman et al. (2018) Systematic Review Parameters

The definitive systematic review of periodontal disease progression (Journal of Periodontology, doi: 10.1002/JPER.17-0062) provides mean annual attachment loss (0.10 mm/year, 95% CI 0.07 to 0.13) and tooth loss (0.20 teeth/year, 95% CI 0.10 to 0.33) with subgroup analyses by geographic region, smoking status, and baseline disease severity. The I-squared value of 99% quantifies the heterogeneity that individual-level models must capture and constitutes the primary statistical justification for the AI-derived stratification approach. These parameters provide prior distributions for Bayesian sensitivity analyses (TP-01, TP-02) and covariate modifier estimates (TP-09, TP-11).

**2.3 Secondary Source: MIMIC-IV (PhysioNet)**

MIMIC-IV is an electronic health record dataset from Beth Israel Deaconess Medical Center, available under PhysioNet credentialing which explicitly permits commercial use. The oral health signal in MIMIC-IV is indirect --- dental diagnoses are coded via ICD-10 (K05.x) rather than clinical periodontal measurements --- and this source is therefore used exclusively within Layer 2 of the data integration framework. It provides longitudinal data on comorbid conditions relevant to the periodontal-systemic disease pathway: diabetes status and glycaemic control (HbA1c), cardiovascular diagnoses, inflammatory markers (CRP), BMI trajectories, and mortality. MIMIC-IV will be used to estimate the comorbidity-specific modifier of periodontal disease transition probabilities (TP-10) via logistic regression linking ICD-10 K05.x codes to diabetes diagnosis codes (E11.x) and HbA1c values. The current placeholder value for this parameter (OR 2.10, 95% CI 1.70 to 2.58) is drawn from the Preshaw et al. (2012, Lancet) meta-analysis and will be replaced by the MIMIC-IV derived estimate in Phase 2.

**2.4 Tertiary Source: Open Zenodo/Figshare Deposits**

Individual study datasets deposited under Creative Commons Attribution (CC-BY) licences will be identified through systematic search of Zenodo and Figshare using periodontal, oral health, and longitudinal search terms. These contribute supplementary training data volume for the AI models in Phase 2 rather than specific named parameters. Automated licence verification will be conducted before any dataset is used.

**2.5 Synthetic Population Generation Framework**

Published parameters from the above sources will be used to generate a synthetic longitudinal population using the SDV library in Python, preserving joint distributions, longitudinal correlation structures, and covariate effects reported in the source literature. The incompatibilities between sources described in Section 1.2 are resolved at this stage: all transition parameters are expressed as annual intensities, measurement protocol differences are handled via published calibration offsets, and ICD-10 derived modifiers are applied multiplicatively to clinically-derived base rates rather than pooled with them. Synthetic data generation will be validated by comparing marginal distributions and key bivariate associations against published summary statistics.

# SECTION 3. METHODS

**3.1 Study Design**

This is a methodological proof-of-concept study comparing AI-derived periodontal disease transition probabilities against literature-derived parameters currently used in health economic models, and evaluating the impact of this methodological choice on model outputs (ICERs). The study follows a three-stage design: (1) survival ML model training and transition probability extraction; (2) integration of AI-derived parameters into a validated Markov model structure; (3) cross-country economic analysis and comparison with literature-derived parameters.

**3.2 Periodontal Disease State Definition**

The model defines five mutually exclusive health states, consistent with the 2018 EFP/AAP classification.

State 1: Periodontal health. No sites with CAL of 2 mm or more; no BOP at more than 25% of sites.

State 2: Gingivitis / Stage I Periodontitis. CAL 1 to 2 mm; pocket depth 4 mm or less.

State 3: Stage II Periodontitis. CAL 3 to 4 mm; maximum pocket depth 5 to 6 mm.

State 4: Stage III/IV Periodontitis. CAL 5 mm or more; pocket depth 7 mm or more, or tooth loss due to periodontitis.

State 5: Edentulism. Complete tooth loss. This is an absorbing state.

Death is included as a competing absorbing state using published UK and US life tables stratified by age and sex. Reversibility (States 3 to 2 and 4 to 3) following successful periodontal treatment is modelled as a covariate-dependent parameter estimated from the treatment response literature, drawing on Preshaw et al. (2017) for the finding that 42 to 77% of progressing sites show subsequent reversal under treatment.

**3.3 Parameter Extraction and Distribution Specification**

All literature-derived parameters are assigned full probability distribution specifications for use in probabilistic sensitivity analysis, using the following conventions.

LogNormal distributions are assigned to transition intensities, hazard ratios, and odds ratios, as these are strictly positive and right-skewed. Distribution parameters (mu, sigma) are derived from the point estimate and 95% CI using moment-matching: sigma equals (ln(upper) minus ln(lower)) divided by 3.92; mu equals ln(estimate) minus sigma-squared divided by 2. Beta distributions are assigned to utilities and proportions (bounded 0 to 1). Parameters (alpha, beta) are derived as: variance equals ((upper minus lower) divided by 3.92) squared; alpha equals mean-squared times (1 minus mean) divided by variance, minus mean; beta equals alpha times (1 divided by mean, minus 1). Gamma distributions are assigned to costs and counts (strictly positive). Parameters are derived as: alpha equals mean-squared divided by variance; beta equals mean divided by variance. Normal distributions are assigned to mean differences and continuous outcomes with symmetric uncertainty. The factor 3.92 equals 2 times 1.96, converting a 95% CI to a standard deviation in all formulae above.

**3.4 AI Model Architecture**

Stage 1: Random Survival Forest with Longitudinal Predictors

The primary AI approach uses the DynForest framework (Devaux et al., 2023, Statistical Methods in Medical Research), implemented in Python using scikit-survival with custom longitudinal feature engineering. DynForest extends competing-risk Random Survival Forests to handle endogenous longitudinal predictors by internally transforming time-dependent covariates using mixed models at each tree node. Final individual event probabilities are computed as Aalen-Johansen estimators averaged across trees. Model performance is assessed using Harrell's C-statistic (target 0.70 or above), time-dependent AUC, and integrated Brier score (target 0.25 or below) with bootstrap confidence intervals.

Stage 2: Dynamic-DeepHit for Competing Risks

As a secondary AI approach, Dynamic-DeepHit (Lee et al., 2019) is implemented to handle the competing risk structure of the model (progression versus tooth loss versus death). This deep learning architecture jointly models longitudinal trajectories and cause-specific cumulative incidence functions using an LSTM encoder, making no parametric assumptions about the underlying hazard.

Stage 3: msm Benchmark

A continuous-time multi-state Markov model implemented in Python using the pymsm package serves as the methodological benchmark, estimating transition intensities as log-linear functions of observed covariates. This preserves the Markov structure required for downstream economic modelling and provides a direct comparator against which the ML models' discriminative performance is assessed.

**3.5 Transition Probability Extraction and Stratification**

Individual-level predicted probabilities from both ML models are aggregated into covariate-stratified transition probability matrices. Strata are defined by: (1) diabetes status (yes/no); (2) smoking status (current/former/never); (3) age group (18 to 44, 45 to 64, 65 and over); (4) country (UK/US). This produces 36 stratified transition matrices (2 times 3 times 3 times 2), each with associated uncertainty intervals from bootstrap resampling (1,000 iterations). Bootstrap-derived distributions replace the standard beta-distribution assumptions used for literature-derived parameters in PSA, providing a more empirically grounded representation of parameter uncertainty.

**3.6 Markov Model Integration**

AI-derived transition probability matrices are integrated into a cohort Markov model implemented in Python. The model structure, state definitions, and utility weights are held constant across comparisons; only the source of transition probabilities (AI-derived vs literature-derived) varies. The model runs over a 10-year and lifetime horizon with annual cycles, using a 3.5% discount rate (UK NICE) and 3.0% discount rate (US Panel on Cost-Effectiveness in Health and Medicine). Costs are assigned using NHS dental banding charges (UK) and ADA CDT procedure codes with Medicare/Medicaid fee schedules (US).

**3.7 Comparative Analysis**

The primary comparison estimates the ICER for non-surgical periodontal therapy (NSPT) versus no treatment using: (a) literature-derived transition probabilities; and (b) AI-derived covariate-stratified transition probabilities. The primary outcome measure is *whether AI-derived transition probabilities produce materially different ICERs, defined as more than 20% change in the ICER point estimate or a change in the cost-effectiveness decision at the NICE threshold*. The economic value of improved precision is quantified using Expected Value of Perfect Parameter Information (EVPPI) analysis, with q34, q45, and the diabetes modifier identified a priori as the highest-priority parameters based on their uncertainty range and influence on model outcomes.

**3.8 Uncertainty Analysis**

Probabilistic sensitivity analysis (PSA) propagates uncertainty from all model parameters simultaneously using Monte Carlo simulation (10,000 samples). Cost-effectiveness acceptability curves (CEACs) are generated for both UK (0 to 50,000 pounds) and US (0 to 150,000 dollars) WTP threshold ranges.

# SECTION 4. PRECEDENT STUDIES

**4.1 Periodontal Disease: Existing Multi-State and Longitudinal Models**

Mdala et al. (2014) --- Comparing CAL and pocket depth for predicting periodontal disease progression in healthy sites of patients with chronic periodontitis using multi-state Markov models. Journal of Clinical Periodontology, doi: 10.1111/jcpe.12278. This is the closest methodological precedent, applying continuous-time Markov models with covariate adjustment to the Goodson dataset. Mdala et al. use a three-state model (health, gingivitis, chronic periodontitis) and report annual transition probabilities rather than intensities. The present study extends this by: (a) replacing parametric estimation with ML survival models; (b) expanding from the three-state to a five-state disease continuum, requiring mapping assumptions for the Mdala State 3 (chronic periodontitis) to the Stage II and Stage III/IV distinction; (c) adding systemic covariates including diabetes and BMI; and (d) linking directly to economic outcomes.

Preshaw et al. (2017) --- Patterns of periodontal disease progression based on linear mixed models of CAL. Journal of Clinical Periodontology, doi: 10.1111/jcpe.12772. Demonstrated that 42 to 77% of progressing sites show subsequent reversal, motivating non-zero treatment reversal probabilities (q32, q43) in the model structure and the use of ML methods that capture this non-stationary behaviour.

Mailoa et al. (2015) --- Long-term effect of four surgical periodontal therapies and one non-surgical therapy: a systematic review and meta-analysis \[14\]. Reported percentage CAL gain by baseline pocket depth category, providing the primary data for parameterising treatment reversal transitions q32 (NSPT) and q43 (surgical therapy) in the present model. Mean CAL gain of 8.4% in 4 to 6 mm pockets and 9.8% in 7 mm or greater pockets over approximately 5 years provides the basis for estimating annual reversal intensities, though conversion from CAL gain to state transition probabilities requires threshold assumptions regarding the clinical attachment level boundaries between disease states.

Tonetti and Claffey (2005) --- Advances in the progression of periodontitis and proposal of definitions of a periodontitis case. Journal of Clinical Periodontology \[15\]. This consensus report synthesised available longitudinal evidence on progression rates and proposed standardised case definitions that influenced the 2018 EFP/AAP classification used in the present model. The reported progression thresholds inform the boundary definitions between disease states S2 through S4.

Axelsson et al. (2004) --- The long-term effect of a plaque control program on tooth mortality, caries and periodontal disease in adults: results after 30 years of maintenance. Journal of Clinical Periodontology \[16\]. This 30-year Swedish longitudinal study reported tooth loss rates of 0.4 to 0.7 teeth per subject over 30 years under supportive periodontal therapy, providing a treated-population benchmark for the q45 transition under maintenance care and informing the treatment effect estimates for the economic model.

Chambrone et al. (2010) --- Tooth loss in treated periodontitis patients: a systematic review with meta-analysis \[17\]. Reported pooled annual tooth loss rates in treated populations, providing cross-validation data for q45 under treatment and complementing the untreated natural history data from the Sri Lanka cohort.

**4.2 Other Chronic Diseases: AI-Derived Transition Probabilities**

Haug et al. (2024, JAMIA) --- Markov modelling for cost-effectiveness using federated health data networks. Demonstrated that patient-level state trajectories from longitudinal EHR data could replicate UK heart failure Markov analyses across five international databases with no statistically significant differences by log-rank test. This provides the strongest methodological precedent for the international validation approach proposed here.

Keane et al. (npj Digital Medicine, 2026) --- Health economic simulation modelling of an AI-enabled coronary revascularisation decision support system. Using real-world data from 25,942 patients, demonstrated mean cost savings of approximately 22,960 dollars and QALY gains equivalent to approximately 22,439 dollars per patient at a 50,000 dollar per QALY threshold. This is the most direct precedent for AI-informed health economic simulation in a chronic disease context.

Langenberger et al. (2023, PLOS ONE) --- Machine learning for predicting high-cost patients using healthcare claims data. Applied RF, GBM, ANN, and logistic regression to German sickness fund data, with RF achieving AUC 0.883. Demonstrates the feasibility of using ML-derived individual risk scores to inform resource allocation decisions.

# SECTION 5. EXPECTED OUTPUTS AND CONTRIBUTION

**5.1 Primary Publication**

The primary output is a methods paper targeting Value in Health, Medical Decision Making, or npj Digital Medicine demonstrating that AI-derived covariate-stratified transition probabilities for periodontal disease: (1) are feasible to derive from published longitudinal data without direct data access, using a structured three-layer data integration framework; (2) differ materially from literature-derived population-average parameters; and (3) produce different cost-effectiveness conclusions for periodontal interventions in identifiable patient subgroups.

**5.2 Secondary Outputs**

An open-source Python package implementing the AI-Markov pipeline, enabling other researchers to apply the method to their periodontal or other disease longitudinal datasets.

A publicly available synthetic longitudinal periodontal dataset generated from the published parameter synthesis, enabling future model validation studies.

An EVPPI analysis identifying which transition probability parameters have the highest value for future primary data collection, informing the design of any future prospective study.

A structured parameter extraction log and project workbook providing a reproducible audit trail for all model inputs.

**5.3 Positioning in the Broader Research Programme**

This paper is conceived as Stage 1 of a three-paper programme. Stage 2 would apply the validated framework to the specific economic question relevant to the company\'s prosthetic context: the cost-effectiveness of periodontal disease prevention in reducing prosthetic failure and edentulism incidence. Stage 3 would use the Value of Information analysis from Stage 1 to design and cost a primary data collection study that would provide the longitudinal data currently absent from the literature.

# SECTION 7. KEY REFERENCES

\[1\] Carletti M, Pandit J, Gadaleta M, et al. Multimodal AI correlates of glucose spikes in people with normal glucose regulation, pre-diabetes and type 2 diabetes. Nature Medicine. 2025. doi:10.1038/s41591-025-03849-7

\[2\] Needleman I, Garcia R, Gkranias N, et al. Mean annual attachment, bone level, and tooth loss: A systematic review. J Periodontol. 2018;89(Suppl 1):S120-S139. doi:10.1002/JPER.17-0062

\[3\] Mdala I, Olsen I, Haffajee AD, Socransky SS, Thoresen M, de Blasio BF. Comparing clinical attachment level and pocket depth for predicting periodontal disease progression in healthy sites of patients with chronic periodontitis using multi-state Markov models. J Clin Periodontol. 2014;41:837-845. doi:10.1111/jcpe.12278

\[4\] Preshaw PM, Bissett SM, de Jager M, et al. Patterns of periodontal disease progression based on linear mixed models of clinical attachment loss. J Clin Periodontol. 2017;44(12):1253-1261. doi:10.1111/jcpe.12772

\[5\] Ramseier CA, Anerud A, Dulac M, et al. Natural history of periodontitis: Disease progression and tooth loss over 40 years. J Clin Periodontol. 2017;44(12):1182-1191. doi:10.1111/jcpe.12782

\[6\] Haug M, Oja M, Pajusalu M, et al. Markov modeling for cost-effectiveness using federated health data network. JAMIA. 2024;31(5):1093-1101. doi:10.1093/jamia/ocae044

\[7\] Devaux A, Helmer C, Genuer R, Proust-Lima C. Random survival forests with multivariate longitudinal endogenous covariates. Stat Methods Med Res. 2023. doi:10.1177/09622802231206709

\[8\] Lee C, Yoon J, van der Schaar M. Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis with Competing Risks. IEEE Trans Biomed Eng. 2019. doi:10.1109/TBME.2019.2929474

\[9\] Suresh K, Taylor J, Spratt D, et al. Random survival forests for dynamic predictions of a time-to-event outcome using a longitudinal biomarker. BMC Med Res Methodol. 2021;21(1):216. doi:10.1186/s12874-021-01375-x

\[10\] Keane P et al. Health economic simulation modeling of an AI-enabled clinical decision support system for coronary revascularization. npj Digital Medicine. 2026. doi:10.1038/s41746-026-02430-x

\[11\] Broadbent JM, Thomson WM, Boyens JV, Poulton R. The Dunedin Multidisciplinary Health and Development Study: Oral health findings and their implications. NZ Dent J. 2020;116(1):20-28.

\[12\] Langenberger B, Schulte T, Groene O. The application of machine learning to predict high-cost patients. PLOS ONE. 2023;18(1):e0279540. doi:10.1371/journal.pone.0279540

\[13\] Preshaw PM, Alba AL, Herrera D, et al. Periodontitis and diabetes: a two-way relationship. Lancet. 2012;379(9785):2244-2256. doi:10.1016/S0140-6736(11)61931-1

\[14\] Mailoa J, Lin GH, Khoshkam V, et al. Long-term effect of four surgical periodontal therapies and one non-surgical therapy: a systematic review and meta-analysis. J Periodontol. 2015;86(10):1138-1151. doi:10.1902/jop.2015.150159

\[15\] Tonetti MS, Claffey N; European Workshop in Periodontology Group C. Advances in the progression of periodontitis and proposal of definitions of a periodontitis case and disease progression for use in risk factor research. J Clin Periodontol. 2005;32(Suppl 6):210-213. doi:10.1111/j.1600-051X.2005.00822.x

\[16\] Axelsson P, Nystrom B, Lindhe J. The long-term effect of a plaque control program on tooth mortality, caries and periodontal disease in adults: results after 30 years of maintenance. J Clin Periodontol. 2004;31(9):749-757. doi:10.1111/j.1600-051X.2004.00563.x

\[17\] Chambrone L, Chambrone D, Lima LA, Chambrone LA. Predictors of tooth loss during long-term periodontal maintenance: a systematic review of observational studies. J Clin Periodontol. 2010;37(7):675-684. doi:10.1111/j.1600-051X.2010.01576.x
