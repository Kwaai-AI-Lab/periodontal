Cost-effectiveness analysis of non-surgical periodontal treatment for preventing strokes and myocardial infarctions in the United Kingdom

Edward James Brastock Coote, MSc (21D Clinical Limited) Edward.coote@21d.co.uk (Corresponding author)

Declarations:

Ethics approval: Not applicable

Consent for publication: Not applicable

Availability of data and materials: The datasets used and/or analysed during the current study are available from the corresponding author on reasonable request (email). All data used was from previously published articles or national sources.

Competing interests: EJBC, [Author 2] and [Author 3] work for 21D Clinical Limited. This is a for-profit company that offer long-term dental prosthetics. The purpose of this study was the provide evidence on the impact of treating oral health conditions non-surgically through the National Health Service, in order to increase holistic understanding about the benefits of oral health and the methods used to investigate. [Author 4] works for a non-profit research lab Kwaai, which collaborated with 21D in research projects. Funding: The funders had no role in the study design, collection of data, presentation of results or writing of the article. The author declares no conflicts of interest with this study. The author does not have any direct or indirect financial interests in the information listed or data used in the study.

Authors contribution: EJBC conceived the study, collected and curated input data, built the Markov model, ran analyses, and drafted the initial manuscript. [Author 4 initials] contributed to model design/validation, sensitivity analysis specification, and interpretation of modelling results. [Author 2 initials] and [Author 3 initials] contributed to manuscript structuring, framing, and substantive revisions to the introduction, discussion, and conclusions. All authors critically revised the manuscript, approved the final version, and agree to be accountable for all aspects of the work.



Dear Editors,

Cardiovascular disease (CVD) remains a major driver of NHS costs and morbidity, and periodontal disease (PD) is increasingly implicated as a modifiable risk factor. However, the policy relevance of this association is limited without an assessment of whether periodontal treatment could represent good value for money as a preventive strategy.

In this manuscript, we present a Markov cohort cost-effectiveness model (NHS payer perspective) evaluating non-surgical periodontal therapy (NSPT) in 65-year-olds with severe PD and no prior CVD. The study makes three contributions relevant to health-economic decision-making:

it quantifies the expected downstream reduction in myocardial infarction and stroke events associated with NSPT under published effect sizes

it demonstrates how time horizon drives conclusions for a chronic preventive intervention

it identifies the treatment-effect magnitude required for NSPT to meet NICE cost-effectiveness thresholds, thereby clarifying the evidence gap for UK-specific studies

In the base case (10 years), NSPT is unlikely to be cost-effective for CVD prevention alone, while longer horizons improve value substantially. Importantly, our estimates are structurally conservative because the model excludes direct oral-health quality-of-life gains, focusing solely on downstream CVD outcomes.

We believe this manuscript fits the aims of Cost Effectiveness and Resource Allocation by providing a transparent evaluation, highlighting key uncertainties, and defining the conditions under which NSPT might represent good value for money.

This work is original, has not been published or submitted elsewhere, and all authors approve the submission. Competing interests and funding statements are provided in the manuscript.



Yours sincerely,

Edward Coote (on behalf of all authors)



Abstract

Background: Periodontal disease (PD) is common in the UK and is associated with higher risks of stroke and myocardial infarction (MI). Non-surgical periodontal therapy (NSPT) may therefore generate downstream cardiovascular benefits, but its value for money as a CVD-prevention strategy for the NHS is unclear.

Methods: A Markov cohort model simulated 65-year-olds with severe PD and no prior CVD over 10 years (with a 25-year scenario). Eight health states captured acute and chronic phases of stroke, MI, combined events, and death. Baseline CVD risks were derived from published sources; NSPT effects (stroke HR 0.55, MI HR 0.70) were taken from large international cohort studies. Costs (2024 £, NHS payer perspective) and QALYs (EQ-5D) were discounted at 3.5%. Deterministic and probabilistic sensitivity analyses assessed parameter and structural uncertainty and identified effect sizes required to meet NICE thresholds.

Results: Over 10 years, NSPT increased costs by £6,487 and QALYs by 0.15 (ICER £44,858/QALY), exceeding NICE thresholds. NSPT reduced non-fatal stroke and MI by 41% and 24%, respectively, increasing event-free survival at 10 years (56% vs 44%). Probabilistic analysis produced a mean ICER of £34,723/QALY and a 25% (at £20,000) to 52% (at £30,000) probability of cost-effectiveness. Extending the horizon to 25 years increased incremental QALYs to 0.82 and reduced the ICER to £16,121/QALY, illustrating strong time-horizon dependence for chronic preventive effects.

Conclusions: NSPT is unlikely to be cost-effective for CVD prevention alone over a 10-year horizon under current evidence, but longer horizons materially improve value. The model also likely underestimates benefits by excluding direct oral-health quality-of-life gains. Improved UK evidence on NSPT's cardiovascular effect size is the key driver of decision uncertainty.




Key words: Cost-effectiveness, periodontal disease, cardiovascular disease, NHS, Markov model




Background

The economic burden of cardiovascular disease (CVD) in the UK is significant, despite falling incidence and mortality rates in recent years [1]. The total costs of coronary heart disease (CHD) and stroke in England are forecast to reach £19.6 billion and £15.9 billion, respectively [2]. CVDs cause approximately 25% of all deaths in the UK and significantly raise an individual's likelihood of developing vascular dementia [3]. Despite a 30% fall in CVD incidence between 2000 and 2019, the economic burden remains high [4]. As CVD affects a large proportion of the population, preventive measures must be cost-effective and deliver health gains beyond CVD, leading to downstream cost reductions.

Periodontal disease (PD) affects at least 50% of people in the UK, with severe periodontal pocketing expected to increase by 56.7% by 2050 [5,6]. A growing number of longitudinal cohort studies and randomised controlled trials supports evidence that treating PD can lower risk of CVD. Non-surgical periodontal therapy (NSPT) has been found to reduce cardiovascular biomarkers in patients with stable coronary artery disease, and to improve endothelial function in patients with a recent myocardial infarction and severe PD [7,8,9]. A meta-analysis found significant reductions in interleukin-6 and systolic blood pressure levels in the treated groups [10]. In patients that have suffered a recent stroke, both standard and intensive PD treatment reduced later stroke and myocardial infarction (MI) mortality [11]. A limitation of trial studies is short follow-up time and small patient pool, therefore longitudinal evidence has to be used to establish stronger causal links. A meta-analysis of longitudinal studies calculated that the risk of all incident CVD was 22% higher in individuals with PD, and the risk of incident CVD increased ascendingly with PD severity [12]. A further review showed consistent findings when stratified by sex and by specific CVD event (CHD; OR = 1.19, 95% CI: 1.09-1.3 and stroke; OR = 1.29, 95% CI: 1.09-1.53) [13]. A US-based study with 21-years of follow up found PD and caries associated with an increased risk of ischemic stroke (HR 1.86; 95% CI 1.32-2.61) [14]. None of these studies explicitly assess the cost-effectiveness of treating PD to prevent CVD events, and since healthcare budgets are expected to become more constrained as the population ages cost-effectiveness will continue to be important to health policy decision makers [15]. The consistent causal link between CVD and PD highlights the potential benefit that population-level PD interventions can have on reducing CVD risk and mortality.

A prior UK study found NSPT cost-effective for improving glycated haemoglobin in Type 2 diabetes [16], and a US analysis suggested PD therapy is cost-effective in reducing diabetes-related microvascular disease [17]. Whether periodontal therapy is cost-effective as a CVD prevention strategy remains unknown.

A Markov model approach was used to estimate to what extent NSPT for adults with severe PD will have a downstream effect on reducing the costs of and number of later CVD events. Long-term costs and effects, using quality-adjusted life years (QALYs), will provide a valuation of NSPT's role in CVD burden management. The base arm will not be treated with NSPT. Ischaemic stroke and MI will be modelled as the CVD outcomes arising from PD risk. The primary objective of this study was to evaluate, from the NHS payer perspective, the 10-year cost-effectiveness of NSPT for adults with severe PD in preventing MI and stroke.



Methods

Model structure

A Markov model with 1 year cycles over 10 years was constructed to estimate the incremental costs and health outcomes associated with severe PD treated with NSPT to prevent CVD events. This was stimulated at an aggregate population level rather than at an individual level as a population-level policy question was evaluated which does not require individual-level heterogeneity.

The model has eight mutually exclusive health states: No prior CVD (base), Post-Stroke (year 1 tunnel), Post-Stroke (year 2+), Post-MI (year 1 tunnel), Post-MI (year 2+), Post-Both (year 1 tunnel), Post-Both (year 2+), and Death (absorbing). Individuals enter the model with no prior history of CVD and can experience stroke, MI, both events or death during each cycle. A tunnel system was chosen to represent temporary health states to allow us to capture acute-phase costs and utilities. CVD event costs and mortality are much higher in the initial onset year compared to subsequent years, therefore, tunnel states provide a more accurate estimation of costs [18,19]. No recurrent events in Post-both year 1 was allowed. Tunnel states relax the memoryless property while retaining time-since-event effects; tunnel duration was one cycle with death as the only competing exit. Half-cycle correction was not modelled.



The base utilities for a 65-year old male were used. No sex-specific differences in NSPT effects have been found [20]. The decision to model at aged 65 and for a horizon of 10 years was justified by the expected onset, survival rates and recurrent events for an individual after a stroke or MI [21,22]. The rate of fatal stroke and MI events was given as 10% and 14%, respectively, from recent studies [23,24]. The background death probability was calculated from Office of National Statistics life tables [25].

Baseline stroke and MI hazards in the base state were calculated from literature that provided the risk of onset events for individuals over 60 years old with severe PD [26, 27]. This was cross checked with a retrospective study that assessed by how much severe PD increased the incidence of the respective CVD event [28]. State-specific hazard multipliers captured elevated risks of subsequent CVD events after a prior event, reflecting disease progression.

NSPT treatment effects were applied to the treatment arm using hazard ratios (HR) obtained from existing literature from international data. There have been no randomised controlled trials or long-term cohort studies testing the effect of NSPT on preventing CVD events within a UK population [29]. Given the absence of longitudinal data for the UK and the use of sensitivity analysis they were deemed appropriate to use despite the parameter uncertainty. There are varied estimates for the size of the impact that NSPT has on CVD events. Retrospective and prospective cohort studies of adults with PD who have had regular NSPT treatment have found lower hazards of stroke of 0.40-0.78 (95% CI 0.29-0.81) relative to untreated PD [30,31]. NSPT has been found to lower the hazard of MI by 0.54-0.90 (95% CI 0.44-0.95) relative to untreated PD [32,33]. The included studies primarily involved older adults with later stage PD who received regular NSPT. Given the heterogeneity in study populations and follow-up durations, the median of reported HRs was used as a summary estimate for base-case analysis. Therefore, stroke treatment HR was 0.55 and MI treatment HR was 0.7. Treatment effects only had an effect on the chance of stroke and MI, background death hazard remained constant throughout. Expected annual periodontal resource use per patient was estimated by combining procedure-specific unit costs with expected frequencies (probability of each procedure per year). This is in line with NHS and dental periodontal treatment protocol [34,35]. We have assumed fixed periodontal treatment intensity and procedures as severe PD is a chronic illness that will require consistent care over the horizon.

Final transition probabilities are in the Supplementary Material Table 3.



Health utilities and costs

The NHS payer perspective was used and reported costs and QALYs in accordance with National Institute for Health and Care Excellence (NICE) guidance [36]. This does not capture the true costs to clinicians and society, as it omits opportunity costs and non-medical costs. The NHS payer perspective is most relevant for reimbursement decisions and has been the most common perspective taken in dental health economics [37].

Base costs for non-surgical treatment were taken from a previous cost-effectiveness study, inflated to 2024 costs, and cross referenced with current NHS Dental band costs and procedure to ensure accuracy [16,38,39]. The per-cycle cost of a patient with severe PD was calculated as £1,274.82 . The use of pre-calculated costs from previous cost-effectiveness analysis ensured replicability and comparability of our results.



Stroke and MI costs for each health state were obtained from contemporary literature and applied to the relevant model states [18,40,41,42]. All individuals experiencing a stroke or MI were assumed to receive treatment within the NHS. Costs were inflated to 2024 prices, with long-term costs assigned to chronic post-event states. These included NHS social care costs. A one-off acute cost was applied during the cycle in which the event occurred. These estimates were cross-checked for consistency against the National Stroke Audit Programme [43]. Where costs were stratified by age and sex chosen the costs most applicable to a 65-year old male. For individuals who experienced both a stroke and a MI, the higher of the two event costs was assigned (post-stroke) to ensure full capture of healthcare resource use. This approach avoids underestimation of costs when overlapping conditions occur, given the lack of a clear consensus on combining costs for multimorbidity in health economic evaluations [44]. Full calculations are found in the Supplementary Material Table 5.

All QALY values were derived from EQ-5D sources. The decision to use 1 year as a cycle length was based on the available data for costs and QALYs associated with CVDs [19,45]. There was found to be minimal difference between the health state of an individual 3 months after a stroke or MI, compared to 9 to 12 months after, which supports the decision to use a year for the Markov states. The health utility values for post-stroke and post-MI for the immediate post-event state were obtained from a single study reporting quality of life impacts for CVD on individuals in the UK [19]. Values for stroke events were cross referenced with values estimated in a meta-analysis [45]. MI utility values were verified with those used in a lipid-lowering cost-effectiveness analysis which we also used costs for [42]. The multiplicative approach was taken to calculate the utility values for post-both events for first year state and subsequent years state. This approach has been found to perform best for combined health states of two conditions simultaneously [46]. A one off disutility effect for each acute event was attached to the event states [45, 47]. The model purposefully excludes any direct oral-health utility gains from improved periodontal status; therefore, estimated QALY gains reflect downstream CVD effects only and are likely conservative.

All cost and QALY outcomes were discounted at the suggested rate of 3.5% for NHS evaluations [36]. Full costing, QALY utility values and parameter breakdown can be found in Supplementary Material Table 2.



Outcomes

Model results included total discounted QALYs, costs and the incremental cost-effectiveness ratio (ICER). The intervention was considered cost-effective if the ICER was below the NICE threshold of £20,000-£30,000 per QALY gained. Net monetary benefit was also calculated.

The Consolidated Health Economic Evaluation Reporting Standards (CHEERS) 2022 checklist for this study has been completed and can be found in Supplementary Material Table 1. The model was built in Microsoft Excel.



Sensitivity Analysis

One-way sensitivity analysis was used to evaluate the impact of changing single parameter values. The maximum and minimum values within a range were used to test uncertainties associated with costs, utilities and transitional probabilities. Periodontal and CVD treatment costs were varied by ±10% of the base case value. Utility and disutility values were tested using the provided confidence intervals.

A lifetime horizon of 25 years was tested as a method of structural sensitivity analysis.

PSA was used to evaluate the robustness of the results, especially given treatment effect data is not from the UK. The β-distribution was used for utility and transition probabilities. The γ-distribution was used for costs. 10,000 Monte Carlo simulations were ran with the value of each model input being randomly drawn from the assigned parametric distribution. A cost-effectiveness plane and cost-effectiveness acceptability curve were drawn using these results. Full distribution parameters for PSA are in Supplementary Table 6.



Results

Base case analysis

Table 1 provides the 10-year costs, accumulated QALYs, incremental costs, incremental QALYs and ICER of NSPT on CVD events. Treatment was associated with a £6,487 higher cost and 0.15 additional QALY per patient treated. This resulted in an ICER of £44,858 per QALY gained, above the NICE threshold of £20,000-£30,000. Given this, NSPT to reduce MI and stroke would not be deemed cost-effective under base-case assumptions. This results in a negative net monetary benefit of -£2,149 at the upper limit of the £30,000 threshold.

Table 1 Results of base-case cost-effectiveness analysis

Treatment resulted in a 41% fall in non-fatal stroke incidence and a 24% fall in non-fatal MI events over the simulated 10-years. By year 10, 56% of individuals in the treatment arm had experienced no CVD events versus 44% in the untreated arm. Cumulative all-cause mortality was lower in the treatment group by 13% at 10 years. Lower mortality and more event-free survivors has translated into health gains, but mostly in avoiding morbidity rather than extending lifespan.



Sensitivity analysis

Figure 1 shows the results of our one-way sensitivity analysis. None of the hazard multipliers by state had an effect of over ±1% on the final ICER. Changing the treatment HR for both MI and stroke had the largest effect on the model outcomes. A stroke treatment effect ranging from 0.29-0.81 gave an ICER range of £14,429-£76,222, and a MI treatment effect range of 0.44-0.95 gave £21,475-£85,902. Only under the most optimistic treatment effects did the ICER fall below £20,000/QALY. The model was moderately sensitive to changing utility values. Most utility values showed less than 5% variation in final ICER calculations. Applying a range of 0.79-0.87 of the base state utility changed the ICER by -17% to 26%. Altering the disutility of an acute stroke event had a much larger effect of -13% to +17% than changing MI disutility, which provided a ±1% difference to the final ICER value. Changing the cost of the post-stroke (Y2) state showed the largest variation out of all the costs, most of which were unresponsive to sensitivity. Changing MI and stroke acute costs, by ±10% each, had a minimal effect on the final ICER. Altering the discount rate to 2% and 5% did not have an effect on the final results.

Holding all else equal, stroke treatment effects would have to have a value of 0.305 to be cost effective at the £20,000 threshold (ICER = £19,899.63), and 0.41 to be cost effective at £30,000 (ICER = £29,485.69). MI treatment effects would need to have a value of 0.32 to be cost effective at the £20,000 threshold (ICER = £19,753), and 0.52 to be cost effective at £30,000 (ICER = £29,490.27). Figure 2 shows the full variation of one-way sensitivity for the treatment effects and their net monetary benefit.




Figure 1 Tornado Plot and Table Of One-Way Sensitivity Analysis For Selected Variable



Figure 2 One-way sensitivity analysis of treatment effects

Figure 3 and Figure 4 reports the results of our PSA. Given a cost-effectiveness threshold range of £20,000 NSPT was cost-effective in 25% of the run. Given a threshold of £30,000, NSPT was cost-effective in 52% of the runs. In 99% of runs NSPT was more effective but more costly. Table 2 provides the mean results of the PSA.

Table 2 PSA Results



Figure 3 Cost-effectiveness plane

Figure 4 Cost-effectiveness acceptability curve



Although the base case analysis used a 65-year-old cohort, preliminary exploration suggest that treating at a younger age would yield more QALY gains as more life-years are at risk. However, modelling at this age is not relevant to CVD policy due to the disease risk at an earlier age, say 55, relative to 65. Instead, a lifetime horizon of 25 years was tested. This led to an incremental cost of £13,189 with 0.82 additional QALYs gained, resulting in an ICER of £16,121. By year 25, 22% of individuals in the treated arm were CVD free, compared to only 11% in the base arm. The average rate of MI and stroke events was the same in the 25 year horizon and 10 year horizon.



Discussion

This study provides a structured economic assessment of the downstream cardiovascular implications of NSPT in a UK-relevant setting. The analysis quantifies potential reductions in MI and stroke events under published effect sizes, demonstrates that conclusions are highly time-horizon dependent for chronic preventive interventions, and identifies the treatment-effect magnitude required for NSPT to meet NICE thresholds. The 41% and 24% fall in non-fatal stroke and MI incidence over a 10-year period demonstrates the impact that treating periodontal disease can have on CVD. It aligns with epidemiological consensus evidence that treating PD could significantly help to prevent CVD [29]. However, these relative reductions translated to only 0.15 QALYs gained per person, indicating that the absolute health benefit was modest relative to the incremental cost. This stems from the fact that despite many non-fatal events being averted, those events carry limited per-person QALY gains. At an incremental cost of £6,487 per patient for 0.15 QALYs, the cost-effectiveness of NSPT compares unfavourably with well-established CVD prevention methods [47].

Changing to a lifetime horizon resulted in a cost-effective ICER of £16,121. This scenario suggests potential long-term value but should be interpreted as exploratory rather than as evidence for CVD-targeted commissioning.

The ICER range resulting from the sensitivity analysis of the two treatment effect parameters has highlighted the uncertainty around the effectiveness of NSPT. Wide testing intervals were used for the treatment effects, but this was to reflect the uncertainty surrounding the magnitude of treatment effects. Using optimistic published hazard reductions (stroke HR ~0.29, MI HR ~0.44) drove the ICER around or below £20,000, whereas using the higher end (HR ~0.8–0.9) made the intervention even more not cost-effective. The estimated level of clinical benefit at an incremental cost of £6,487 emphasizes the need for greater clinical trials and cohort studies investigating the effects of periodontal treatment beyond oral health benefits. Current findings highlight that evidence for CVD benefits of periodontal interventions remains low-certainty [49], making high-quality RCTs or longitudinal studies essential to address this gap. That being said, a major UK randomised trial recently published has shown that intensive periodontal treatment can slow carotid artery atherosclerosis progression and improve vascular function [50].

Using the base case analysis assumptions, NSPT would need to decrease to a cost of £1,000 to result in an ICER of £26,433. This cost reduction could be achieved through shorter dental appointments or less frequent checkups, however, this would impact NSPT effectiveness. In the context of the NHS, any form of periodontal treatment often involves patient co-payments, lowering the true cost to the NHS. While co-payments may improve affordability, they do not change the underlying value to the health system, and therefore do not alter the ICER from the NHS perspective. NSPT as a method of preventing future CVD events could be covered with co-payments and therefore proved cost-effective if further health benefits beyond oral health and CVD events are also accounted for. A strict NHS perspective was used, so taking a broader societal perspective which would include other elements such as productivity gains could effectively bring the ICER into an acceptable range. At an incremental cost of £6,487 per patient for 0.15 QALYs the better value of established CVD prevention methods has to be considered. Well researched cost-effectiveness of methods such as statins and smoking cessation will likely buy more QALYs, especially in a crowded priority-setting space such as the NHS.



Previous economic studies have focused on diabetes-related outcomes which limits comparisons. Choi et al found that expanding periodontal treatment in a US population was cost-saving in averting tooth loss and microvascular disease, with the majority of healthcare cost saving coming from averting tooth loss [17]. This approach simulated by individual rather than an aggregate population due to their testing of correlations between demographic characteristics and chronic disease risks obtained from the National Health and Nutrition Examination Survey (NHANES) (2009-2014). A UK model by Solowiej-Wedderburn found periodontal treatment in a 58-year old man with type 2 diabetes (HbA1c 7%-7.9%) to have an ICER of £28,000, with health gains larger in patients with higher HbA1c [16]. These attributed savings helped to improve glycaemic control, suggesting that periodontal therapy's value is greater when considering the benefits beyond CVD alone. The CVD model in this study did not find cost savings, highlighting that the value of NSPT depends heavily on the range of outcomes included and that CVD-only analyses provide a narrower perspective.



Limitations

The most significant limitation is the uncertainty surrounding NSPT's effect on CVD events. To address this uncertainty, a wide range of NSPT effect sizes was examined. Due to this uncertainty, it was also decided that it was appropriate to assume time-invariant hazard ratios which implies constant relative risk reduction.

The scope of this study was to evaluate the cost-effectiveness of NSPT on fatal and non-fatal CVD events. Therefore, these results do not include the patient-level benefits of NSPT on severe PD. This makes the QALY estimations to be underestimates of total health benefits, however, this was out of the study scope. Direct utility gains from reduced tooth pain, tooth retention, and better chewing function and nutrition are not captured in this analysis. Including these gains would improve the cost-effectiveness of periodontal treatment, but widen the scope beyond CVD outcomes.

For individuals with severe PD and co-existing comorbidities, NSPT may offer broader systemic health improvements, such as reductions in hypertension and enhancements in endothelial function [51]. These potential benefits were not modelled and should not be inferred as contributing to the results.

A further limitation of the base-case analysis is the 10-year time horizon, which may not capture the full benefits of NSPT as a chronic preventive intervention. Scenario analysis extending the horizon to 25 years showed that the ICER decreased markedly to £16,121 per QALY gained, highlighting that the choice of time horizon has a major impact on cost-effectiveness conclusions. Modelling at an aggregate Markov level does not account for individual patient-level factors that could affect treatment effectiveness, despite the evaluation targeting a population-level policy. However, a UK-based longitudinal study on NSPT's effect size controlling for individual characteristics does not exist which is why we chose not to use a microsimulation evidence-based risk model. A longitudinal study such as that would also help to reduce the uncertainty surrounding NSPT's effect on cardiovascular risk.



Conclusions

This analysis defines the conditions under which NSPT could represent good value for money as part of CVD prevention policy. ICER results are highly time-horizon dependent and are primarily driven by uncertainty in the NSPT cardiovascular effect size. NSPT was not found to be cost-effective for CVD prevention alone over a 10-year horizon. This primary result indicates that, at current costs and with existing evidence, NSPT should not be considered an NHS-funded CVD prevention strategy in its own right. A 25-year analysis indicated an ICER of £16,121 per QALY, suggesting that NSPT is likely to represent good value for money when longer-term benefits are considered. Despite the consistent link between PD and systemic health, our results do not justify NSPT as an NHS-funded CVD prevention strategy on its own. If NSPT is to be considered, it should include its direct oral-health benefits and any concurrently modelled systemic benefits in a multi-outcome framework. The uncertainty surrounding treatment-effect estimates highlights the need for high-quality trials and longitudinal studies.




List of abbreviations

CVD – Cardiovascular disease

CHD – Coronary heart disease

PD – Periodontal disease

NSPT – Non-surgical periodontal treatment

MI - Myocardial infarction

QALY – Quality-adjusted life year

ICER – Incremental cost-effective ratio

PSA – probabilistic sensitivity analysis




References

1.  Shih K, Herz N, Sheikh A, O'Neill C, Carter P, Anderson M. Economic burden of cardiovascular disease in the United Kingdom. Eur Heart J Qual Care Clin Outcomes. 2025;qcaf011. doi:10.1093/ehjqcco/qcaf011

2.  Landeiro F, Harris C, Groves D, et al. The economic burden of cancer, coronary heart disease, dementia, and stroke in England in 2018, with projection to 2050: an evaluation of two cohort studies. Lancet Healthy Longev. 2024;5(8):e514-e523.

3.  British Heart Foundation. UK Cardiovascular Disease Factsheet. London, England: British Heart Foundation; September 2025. Accessed October 27, 2025. Available from: https://www.bhf.org.uk/-/media/files/for-professionals/research/heart-statistics/bhf-cv.. d-statistics-uk-factsheet.pdf?rev=0759fdcb1d3248f9b9331c4039e6075c&hash=B0C8BEA1A48B306E4D2FC73C4265FBFA

4.  Conrad N, Molenberghs G, Verbeke G, et al. Trends in cardiovascular disease incidence among 22 million people in the UK over 20 years: population based study. BMJ. 2024;385:e075210.

5.  Department of Health and Social Care, NHS England. Delivering Better Oral Health: An Evidence-Based Toolkit for Prevention. Chapter 5: Periodontal Diseases. London, England: Department of Health and Social Care; updated September 10, 2025. Available from: https://www.gov.uk/government/publications/delivering-better-oral-health

6.  Elamin A, Ansah JP. Projecting the burden of dental caries and periodontal diseases among the adult population in the United Kingdom using a multi-state population model. Front Public Health. 2023;11:1190197. doi:10.3389/fpubh.2023.1190197

7.  Montenegro MM, Ribeiro IW, Kampits C, et al. Randomized controlled trial of the effect of periodontal treatment on cardiovascular risk biomarkers in patients with stable coronary artery disease: preliminary findings of 3 months. J Clin Periodontol. 2019;46(3):321-331.

8.  Bokhari SAH, Khan AA, Butt AK, et al. Non-surgical periodontal therapy reduces coronary heart disease risk markers: a randomised controlled trial. J Clin Periodontol. 2012;39(11):1065-1074.

9.  Lobo MG, Schmidt MM, Lopes RD, et al. Treating periodontal disease in patients with myocardial infarction: a randomised clinical trial. Eur J Intern Med. 2020;71:76-80.

10.  Meng R, Xu J, Fan C, Liao H, Wu Z, Zeng Q. Effect of non-surgical periodontal therapy on risk markers of cardiovascular disease: a systematic review and meta-analysis. BMC Oral Health. 2024;24(1):692.

11.  Sen S, Curtis J, Hicklin D, et al. Periodontal disease treatment after stroke or transient ischemic attack: the PREMIERS Study, a randomised clinical trial. Stroke. 2023;54(9):2214-2222.

12.  Larvin H, Kang J, Aggarwal VR, Pavitt S, Wu J. Risk of incident cardiovascular disease in people with periodontal disease: a systematic review and meta-analysis. Clin Exp Dent Res. 2021;7(1):109-122.

13.  Leng Y, Hu Q, Ling Q, et al. Periodontal disease is associated with the risk of cardiovascular disease independent of sex: a meta-analysis. Front Cardiovasc Med. 2023;10:1114927.

14.  Wood S, Logue L, Meyer J, et al. Combined influence of dental caries and periodontal disease on ischemic stroke risk. Neurology Open Access. 2025;1(4):e000036.

15.  Rachet-Jacquet L, Rocks S, Charlesworth A. Long-term projections of health care funding, bed capacity and workforce needs in England. Health Policy. 2023;132:104815.

16.  Solowiej-Wedderburn J, Ide M, Pennington M. Cost-effectiveness of non-surgical periodontal therapy for patients with type 2 diabetes in the UK. J Clin Periodontol. 2017;44(7):700-707.

17.  Choi SE, Sima C, Pandya A. Impact of treating oral disease on preventing vascular diseases: a model-based cost-effectiveness analysis of periodontal treatment among patients with type 2 diabetes. Diabetes Care. 2020;43(3):563-571.

18.  Patel A, Berdunov V, Quayyum Z, King D, Knapp M, Wittenberg R. Estimated societal costs of stroke in the UK based on a discrete event simulation. Age Ageing. 2020;49(2):270-276.

19.  Lui JN, Williams C, Keng MJ, et al. Impact of new cardiovascular events on quality of life and hospital costs in people with cardiovascular disease in the United Kingdom and United States. J Am Heart Assoc. 2023;12(19):e030766.

20.  Angelov N, Soldatos N, Ioannidou E, et al. A retrospective analysis of the role of age and sex in outcomes of non-surgical periodontal therapy at a single academic dental center. Sci Rep. 2024;14(1):9504.

21.  Hall M, Smith L, Wu J, et al. Health outcomes after myocardial infarction: a population study of 56 million people in England. PLoS Med. 2024;21(2):e1004343.

22.  Shavelle RM, Brooks JC, Strauss DJ, Turner-Stokes L. Life expectancy after stroke based on age, sex, and Rankin grade of disability: a synthesis. J Stroke Cerebrovasc Dis. 2019;28(12):104450.

23.  Morgan A, Sinnott SJ, Smeeth L, Minassian C, Quint J. Concordance in the recording of stroke across UK primary and secondary care datasets: a population-based cohort study. BJGP Open. 2021;5(2):BJGPO.2021.0011.

24.  Allara E, Shi W, Bolton T, et al. Burden of cardiovascular diseases in England (2020–24): a national cohort using electronic health records data. Lancet Public Health. 2025;10(11):e943-e954.

25.  Office for National Statistics. National Life Tables: UK [Internet]. London, England: Office for National Statistics; March 18, 2025. Accessed October 23, 2025. Available from: https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/nationallifetablesunitedkingdomreferencetables

26.  Zemedikun DT, Chandan JS, Raindi D, et al. Burden of chronic diseases associated with periodontal diseases: a retrospective cohort study using UK primary care data. BMJ Open. 2021;11(12):e048296.

27.  Wagner AK, D'Souza M, Bang CN, et al. Treated periodontitis and recurrent events after first-time myocardial infarction: a Danish nationwide cohort study. J Clin Periodontol. 2023;50(10):1305-1314.

28.  Cho HJ, Shin MS, Song Y, et al. Severe periodontal disease increases acute myocardial infarction and stroke: a 10-year retrospective follow-up study. J Dent Res. 2021;100(7):706-713. doi:10.1177/0022034520986097

29.  Sanz M, Marco del Castillo A, Jepsen S, et al. Periodontitis and cardiovascular diseases: consensus report. J Clin Periodontol. 2020;47(3):268-288.

30.  Lee YL, Hu HY, Huang N, Hwang DK, Chou P, Chu D. Dental prophylaxis and periodontal treatment are protective factors to ischemic stroke. Stroke. 2013;44(4):1026-1030. doi:10.1161/STROKEAHA.111.000076

31.  Kim YR, Son M, Kim SR. Association between regular dental scaling and stroke risk in patients with periodontal diseases: evidence from a Korean nationwide database. Epidemiol Health. 2025;47:e2025020.

32.  Kao YW, Shia BC, Chiang HC, Chen M, Wu SY. Association of tooth scaling with acute myocardial infarction and analysis of the corresponding medical expenditure: a nationwide population-based study. Int J Environ Res Public Health. 2021;18(14):7613. doi:10.3390/ijerph18147613

33.  Lee YL, Hu HY, Chou P, Chu D. Dental prophylaxis decreases the risk of acute myocardial infarction: a nationwide population-based study in Taiwan. Clin Interv Aging. 2015;10:175-182. doi:10.2147/CIA.S67854

34.  Department of Orthodontics and Restorative Dentistry, Glenfield Hospital. Periodontal Treatment Protocol. Leicester, England: Glenfield Hospital; [year unknown].

35.  NHS England. Avoidance of Doubt: Provision of Phased Treatments. London, England: NHS England; July 8, 2021. Accessed October 27, 2025. Available from: https://www.bsperio.org.uk/assets/downloads/B0615-Update-to-NHS_avoidance-of-doubt-provision-of-phased-treatments-July-2021.pdf

36.  National Institute for Health and Care Excellence (NICE). Developing NICE Guidelines: The Manual. NICE Process and Methods [PMG20]. London, England: NICE; October 31, 2014.

37.  Schwendicke F, Rossi JG, Göstemeyer G, et al. Cost-effectiveness of artificial intelligence for proximal caries detection. J Dent Res. 2021;100(4):369-376.

38.  HM Treasury. GDP Deflators at Market Prices, and Money GDP: March 2025 (Spring Statement & Quarterly National Accounts) [Internet]. London, England: HM Treasury; March 2025. Accessed October 24, 2025. Available from: https://www.gov.uk/government/statistics/gdp-deflators-at-market-prices-and-money-gdp-march-2025-spring-statement-quarterly-national-accounts

39.  National Health Service (England). How Much NHS Dental Treatment Costs. London, England: NHS; [no date]. Accessed October 27, 2025. Available from: https://www.nhs.uk/nhs-services/dentists/how-much-nhs-dental-treatment-costs/

40.  Danese MD, Gleeson M, Kutikova L, et al. Estimating the economic burden of cardiovascular events in patients receiving lipid-modifying therapy in the UK. BMJ Open. 2016;6(8):e011805.

41.  National Health Service (NHS). 2020/21 National Cost Collection for the NHS. London, England: NHS England; [no date]. Accessed October 27, 2025. Available from: https://www.england.nhs.uk/costing-in-the-nhs/national-cost-collection/

42.  Morton JI, Marquina C, Lloyd M, et al. Lipid-lowering strategies for primary prevention of coronary heart disease in the UK: a cost-effectiveness analysis. Pharmacoeconomics. 2024;42(1):91-107.

43.  Xu XM, Vestesson E, Paley L, et al. The economic burden of stroke care in England, Wales and Northern Ireland: using a national stroke register to estimate and report patient-level health economic outcomes in stroke. Eur Stroke J. 2018;3(1):82-91. doi:10.1177/2396987317746516

44.  Lomas J, Asaria M, Bojke L, Gale CP, Richardson G, Walker S. Which costs matter? Costs included in economic evaluation and their impact on decision uncertainty for stable coronary artery disease. Pharmacoeconomics Open. 2018;2(4):403-413.

45.  Joundi RA, Adekanye J, Leung AA, et al. Health state utility values in people with stroke: a systematic review and meta-analysis. J Am Heart Assoc. 2022;11(13):e024296. doi:10.1161/JAHA.121.024296

46.  Thompson AJ, Sutton M, Payne K. Estimating joint health condition utility values. Value Health. 2019;22(4):482-490.

47. Mihaylova B, Wu R, Zhou J, et al. Lifetime effects and cost-effectiveness of standard and higher-intensity statin therapy across population categories in the UK: a microsimulation modelling study. Lancet Reg Health Eur. 2024;40:100887.

48.  Thom HH, Hollingworth W, Sofat R, et al. Directly acting oral anticoagulants for the prevention of stroke in atrial fibrillation in England and Wales: cost-effectiveness model and value of information analysis. MDM Policy Pract. 2019;4(2):2381468319866828.

49. Ye Z, Cao Y, Miao C, et al. Periodontal therapy for primary or secondary prevention of cardiovascular disease in people with periodontitis. Cochrane Database Syst Rev. 2022;10:CD009197. doi:10.1002/14651858.CD009197.pub5.

50. Orlandi M, Masi S, Lucenteforte E, et al. Periodontitis treatment and progression of carotid intima-media thickness: a randomised trial. Eur Heart J. 2025;ehaf555. doi:10.1093/eurheartj/ehaf555.

51.  Rodrigues JV, Deroide MB, Sant'ana AP, de Molon RS, Theodoro LH. The role of non-surgical periodontal treatment in enhancing quality of life for hypertensive patients with periodontitis. Rev Odontol UNESP. 2024;53:e20240030.




Online Supplementary Material to: Cost-effectiveness analysis of non-surgical periodontal treatment for preventing strokes and myocardial infarction in the UK



Consolidated Health Economic Evaluation Reporting Standards 2022 Checklist

Table 1 CHEERS 2022 Checklist

Model Parameters And Values

Table 2 All model parameters and values

Transition Matrices

Table 3 Base Markov model transition matrices

Cost Associated With Periodontal Disease

Table 4 Full breakdown of periodontal costs

Costs Associated With Markov States

Costs sourced from Patel et al. (2020) were multiplied using a 2014/2015 GPD inflator value of 1.3396 to get our 2024 deterministic value. A value of 1.1706 was used for costs sourced from Morton et al. (2024).

Table 5 Breakdown of costs per Markov state

Sensitivity Analysis

Table 6 Parameters and related values used in sensitivity analysis



References

[1] Zemedikun DT, Chandan JS, Raindi D, et al. Burden of chronic diseases associated with periodontal diseases: a retrospective cohort study using UK primary care data. BMJ Open. 2021;11(12):e048296. doi:10.1136/bmjopen-2020-048296.

[2] Seoane T, Bullon B, Fernandez-Riejos P, et al. Periodontitis and other risk factors related to myocardial infarction and its follow-up. J Clin Med. 2022;11(9):2618. doi:10.3390/jcm11092618.

[3] Office for National Statistics. National Life Tables: UK [Internet]. London, England: Office for National Statistics; March 18, 2025. Accessed October 23, 2025. Available from: https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/nationallifetablesunitedkingdomreferencetables

[4] Morgan A, Sinnott SJ, Smeeth L, Minassian C, Quint J. Concordance in the recording of stroke across UK primary and secondary care datasets: a population-based cohort study. BJGP Open. 2021;5(2):BJGPO.2020.0117. doi:10.3399/BJGPO.2020.0117

[5] Allara E, Shi W, Bolton T, et al. Burden of cardiovascular diseases in England (2020–24): a national cohort using electronic health records data. Lancet Public Health. 2025;10(11):e943–e954. doi:10.1016/S2468-2667(25)00163-X.

[6] Danese MD, Pemberton-Ross P, Catterick D, Villa G. Estimation of the increased risk associated with recurrent events or polyvascular atherosclerotic cardiovascular disease in the United Kingdom. Eur J Prev Cardiol. 2021;28(3):335–343. doi:10.1177/2047487319899212.

[7] Lui JN, Williams C, Keng MJ, et al; REVEAL Collaborative Group. Impact of new cardiovascular events on quality of life and hospital costs in people with cardiovascular disease in the United Kingdom and United States. J Am Heart Assoc. 2023;12(19):e030766. doi:10.1161/JAHA.123.030766.

[8] Thom HH, Hollingworth W, Sofat R, et al. Directly acting oral anticoagulants for the prevention of stroke in atrial fibrillation in England and Wales: cost-effectiveness model and value of information analysis. MDM Policy Pract. 2019;4(2):2381468319866828. doi:10.1177/2381468319866828.

[9] Morton JI, Marquina C, Lloyd M, Watts GF, Zoungas S, Liew D, Ademi Z. Lipid-lowering strategies for primary prevention of coronary heart disease in the UK: a cost-effectiveness analysis. Pharmacoeconomics. 2024;42(1):91–107. doi:10.1007/s40273-023-01306-2.

[10] Patel A, Berdunov V, Quayyum Z, King D, Knapp M, Wittenberg R. Estimated societal costs of stroke in the UK based on a discrete event simulation. Age Ageing. 2020;49(2):270–276. doi:10.1093/ageing/afz162.

[11] Youman P, Wilson K, Harraf F, Kalra L. The economic burden of stroke in the United Kingdom. Pharmacoeconomics. 2003;21:43–50. doi:10.2165/00019053-200321001-00005.

[12] National Health Service. 2020/21 National Cost Collection for the NHS. Available from: https://www.england.nhs.uk/costing-in-the-nhs/national-cost-collection/. Accessed December 1, 2025.
