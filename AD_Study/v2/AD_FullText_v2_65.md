**The impact of periodontal prevalence scenarios on the costs and prevalence of dementia in England: a modelling study**

**Introduction**
The burden of periodontal diseases (PD) in England is substantial. NHS Dental estimates that at least 50% of the UK have some level of irreversible (severe) periodontal disease (PD), although this is likely to be an underestimate [1]. Globally, there was an estimated 50% increase in oral disease cases between 1990 and 2019, which was higher than the 45% population growth over the period [2]. This trend is expected to increase in high-income countries as the ageing population results in individuals living longer with chronic diseases [3]. Estimations using multi-state population models for periodontal pocketing in the UK project an 8.7% increase from 2020 to 2050, with severe pocketing increasing by 56.7% [4].
*There is substantial evidence that PD is a risk factor for dementia [5,6]. Periodontal infection leads to systemic inflammation, allowing high inflammatory immune responses to cross the blood-brain barrier which activates the microglial cells in the cerebral cortex, which may contribute to the pathogenesis of dementia [7,8]. Dementia patients have shown higher fusobacterium loads, as well as Campylobacter rectus and Porphyromonas gingivalis, both species of bacteria that are biomarkers for severe PD [9]. One review found that PD was associated with a 22% higher risk of incident dementia during a mean follow up of 11 years, with another using cohort studies to suggest a relative risk (RR) of 1.18 (1.06-1.31) for incident dementia in relation to PD [6,10]. There is bidirectional causality, as an individual’s ability to maintain their oral health deteriorates with increased dementia symptoms [11]. However, trial emulation studies testing periodontal treatments have shown their ability to reduce dementia-related brain atrophy in the treatment arms (-0.41; 95% CI; -0.70 to -0.12; P= 0.0051) [12]. A retrospective cohort study also identified that periodontal treatment reduced the risk of dementia mortality [13]. Longitudinal cohort studies have revealed that PD was significantly associated with increased dementia incidence, even after adjusting for major chronic diseases [14,15,16]. These longitudinal studies were restricted to dementia-free participants at baseline, minimising reverse causality.*
Dementia costs in England are expected to rise significantly, with one estimate forecasting a 173% rise from 2019 to 2040, totalling £80.4 billion by 2040 [17]. Costs for England account for 85% of the total 2040 costs for the UK. The same projection gives 1.35 million total older people living with dementia in England in 2040, also accounting for 85% of the UK’s total burden. Approximately 45% of all dementia cases are attributable to preventable risk factors [18]. Modelling the effects of population-based interventions on dementia risk factors can provide estimates of potential cost-saving and disease reduction. If hypertension prevalence fell in England and Wales by 50% between 2027 and 2060, an estimated 57 fewer deaths per 100,000 population but an increase of 9 more dementia cases per 100,000 is expected [19]. This highlights the interplay between reduced risk factor prevalence and extended life expectancy, which may raise overall dementia prevalence. A 50% reduction in the current global prevalence of PD could prevent 850,000 individuals from developing dementia [6]. Given the strong evidence linking periodontal disease to dementia and the projected rise in dementia-related costs, estimating the effects of reducing periodontal disease prevalence is essential to inform prevention strategies.
We use an individual-level microsimulation model to estimate the costs and benefits associated with counterfactual scenarios in severe PD prevalence on future dementia outcomes. We start with the current estimated PD prevalence of 50%, then test a prevalence of 25% and 75%. This will account for the effects of other dementia risk factors on both dementia and mortality risk.

**Methods**
**Model overview**
We simulated the England population (aged 36 years and older) from a 2023 baseline, used for initialisation only, to 2040 in annual time steps, with results reported from 2024 onward. New individuals entered the model yearly through a fixed-size entrant cohort (n = 700,000), reweighted and sized to match age- and sex-specific population projections from the Office for National Statistics [20,21]. The baseline dementia prevalence, age distribution and severity stage mix was set using the Primary Care Dementia Data register. Dementia was defined as diagnosed all-cause dementia, referring to any type of dementia regardless of its cause [22]. We developed a stochastic time-to-event microsimulation model with a dementia stage structure consistent with previous dementia progression models, using longitudinal evidence and mortality projections to inform transition hazards [19, 23, 24, 25].
Dementia in our model is split in five stages; cognitively normal, mild, moderate, severe, and death. Dementia-related death was permitted only in the severe stage, while background mortality could occur from any state. Disease state changes are simulated each year. Individuals with dementia could transition between home and institution (or remain in current) care each year based on setting transition probabilities [26]. The baseline onset probability of dementia was calculated using current dementia incidence from NHS England data. We externally validated our model to observed prevalence data from 2023 to 2024, with goodness of fit assessed with  and slope coefficients.

**Equation 1 Dementia onset formula**
The mean time in each dementia stage has been collected from [27]. This provides a constant hazard (exponential assumptions) for the effect that time in a dementia stage has on disease progression. Initial dementia stage was set from a previously published report [26]. We used a parametric proportional hazards age effect to scale the increasing effects that age has on dementia onset. This provides a continuous increase in risk with age rather than discrete banding, as seen in longitudinal studies of dementia incidence [28]. We calculated the relevant β’s for the per-year log-hazard slope increase of stage transitions [29,30,31]. Risk factor hazard ratios were applied to dementia onset. Onset from cognitively normal is driven by a baseline annual onset probability (converted to a hazard), then scaled by a parametric age effect and individual risk-factor hazard ratios (Equation 1). Full details of our model mechanisms are provided in the supplementary information (pp.x-x).

**Risk factors**
In this model, risk factors interact with dementia onset. Progression uses exponential stage-duration hazards, scaled by age. We used HRs from literature to build risk factor interaction terms, split by sex where available. These selected risk factors were chosen due to the available evidence linking them to increased dementia risk, while together representing the major preventable factors of dementia incidence. Prevalence and HRs were held constant, with only PD prevalence changing.
The prevalence of PD in England was set as 50% in our baseline model, based on NHS Dental estimates, which estimates this as the level of severe periodontitis [1]. The NHS expects this to be an underestimate, therefore we tested at 75% to provide an upper bound estimate, and used 25% as an optimistic lower bound target. Severe PD-related HRs were applied to dementia onset and dementia-related mortality from large national studies calculating risk of incident dementia and death [33,34].
The model included smoking status, diabetes, cerebrovascular disease and cardiovascular disease as risk factors. Whether an individual smokes, or has a history of smoking, is an established risk factor for dementia. Sex differences in related onset have been explored, which is reflected in our model parameters [35, 36]. Longitudinal studies have demonstrated Type 2 diabetes as a risk factor for dementia, linking chronic hyperglycaemia to damaged blood vessels in the brain and systemic inflammation and oxidative stress [37,38]. We used diabetes as a proxy for obesity related-risk as the metabolic syndrome mechanisms are significantly associated with increased dementia risk, whereas obesity individually has a minor independent effect on dementia risk or death [39]. Strokes, representing cerebrovascular disease, cause brain tissue damage due to interrupted blood flow, with strong evidence suggesting those with prior stroke are at a heightened risk of dementia onset [40,41]. Heart failure was chosen as a representative of the cardiovascular disease risk factor as it captures the contribution of cardiac disfunction to dementia onset and death [35]. Prevalence data for each was obtained through national survey sources [42,43,44].
This risk set captures the vascular, lifestyle and inflammatory pathways that influence cognitive decline. Risk factors remain constant from an individual’s entry into the model. Full HR values can be found in the supplementary information (pp.xx-xx).

**Estimation of costs and QALYs**
We estimated the economic costs of dementia for direct formal healthcare and social care and informal caregivers. Table X shows the yearly costs used and Figure XXX shows the cost allocation. Costing data was sourced from a 2024 report on dementia and converted to 2023 values using GDP deflators [45, 46]. Direct formal healthcare and social care costs included healthcare and social care. Informal costs include unpaid care and quality of life costs, which we separated into costs for dementia patients in home or institution care (Supplementary Material pp.xx-xx).
QALYs are commonly used to value the benefit of interventions, with one QALY equating a year of life in perfect health [47]. This means years of life can take into account the quality and quantity of life. Dementia stage specific QALY utility values were obtained from a previous study modelling the benefits of population-level interventions for dementia risk factors and applied to those with dementia [48]. Age specific utility values for cognitively normal were taken from the UK Population Norms for EQ-5D [49]. An informal caregiver for someone with dementia will also see their quality of life affected due to the strain of care. We apply a caregiver disutility impact for dementia-at-home individuals and treat it as an incremental disutility. These utility values were attained from literature assessing the impact of caring for people with dementia [50]. An annual discount rate of 3.5% was applied to both costs and QALYs as suggested by the National Institute of  Care and Excellence (NICE) and the UK Treasury [51].Within each calendar year, differences in total QALYs between scenarios reflect changes attributable to PD-driven differences in dementia onset, survival, and progression (and associated caregiver impacts), holding all other model components constant. Full methodology for the measurement of costs and QALYs can be found in the supplementary information (pp.x-x).

|  | Home Care | Home Care | Home Care | Institutional Care | Institutional Care | Institutional Care |
|---|---|---|---|---|---|---|
| Severity | Formal | Informal | Total | Formal | Informal | Total |
| Mild | £7,466.70 | £10,189.55 | £17,656.25 | £23,144.27 | £874.93 | £24,019.19 |
| Moderate | £7,180.18 | £33,726.09 | £40,906.28 | £15,552.58 | £1,643.14 | £17,195.71 |
| Severe | £7,668.60 | £31,523.39 | £39,191.99 | £53,084.13 | £501.88 | £53,586.01 |

**Table X Yearly costs by dementia stage by living setting (£)**


**Figure XXX Flowchart of care setting allocation**

**Outcomes**
Differences in dementia incidence and prevalence over time were estimated under each PD prevalence scenario and compared. The model estimated the costs and QALYs attached with dementia, reported year-by-year and cumulatively. Costs were categorised by formal care giving and informal care costs. QALYs were aggregated by dementia patient and informal home caregivers. This allowed for the comparison of dementia person-years avoided, total and annual cost savings, QALYs XXX, and cost savings per dementia case avoided, between the differences in PD prevalence runs.

**Sensitivity analysis**
We conducted probabilistic sensitivity analysis (PSA) with 500 Monte Carlo iterations for 1% of the target population (n = 107,874), sampling uncertain parameters from probability distributions (sampling costs with the gamma distribution, utilities and probabilities with beta, and risk HRs with lognormal from CIs; relative SD = 10%) [52]. The outcomes were then scaled back up by a factor of 100. This was due to the computational strain of sampling for the full population. Results are reported as means with 95% confidence intervals.
Deterministic one-way sensitivity analysis was conducted to specifically address the uncertainty surrounding PD’s impact on dementia onset and severe‑stage mortality, using the published 95% CI bounds for PD hazard ratios (1.32–1.65 for onset; 1.10–1.69 for severe‑stage mortality). As with our PSA, the runs used 1% of the total population with 10 independent model runs per PD prevalence due to computational feasibility. Outputs were then scaled up to full-cohort equivalents.

**Results**
**Model summary results**

|  | 25% PD | 50% PD | 75% PD |
|---|---|---|---|
| Epidemiology Outcomes | Epidemiology Outcomes | Epidemiology Outcomes | Epidemiology Outcomes |
| Dementia prevalence in 2040 (n) | 946,903 | 1,033,451 | 1,120,015 |
| % change 2024-2040 | 16 | 24 | 34 |
| Incident dementia cases in 2040 (n) | 146,962 | 159,591 | 172,404 |
| % change 2024-2040 | 63 | 60 | 59 |
| Costs 2024-2040 (£bn) | Costs 2024-2040 (£bn) | Costs 2024-2040 (£bn) | Costs 2024-2040 (£bn) |
| Total costs | 383.39 | 406.67 | 429.23 |
| Formal care | 213.87 | 226.77 | 239.25 |
| Informal care | 169.52 | 179.90 | 189.98 |
| Health Outcomes 2024-2040 (mn) | Health Outcomes 2024-2040 (mn) | Health Outcomes 2024-2040 (mn) | Health Outcomes 2024-2040 (mn) |
| Total individual QALYs | 113.82 | 113.63 | 113.44 |
| Caregiver QALYs | 6.2 | 6.65 | 7.07 |

**Table X Outcomes by periodontal disease (PD) prevalence scenarios. Cost and QALY outcomes are discounted at 3.5%**

**Figure XXX Cases per 1,000 population by periodontal disease (PD) prevalence**

Table XXX shows the summary results for each PD prevalence run. Incident dementia cases increased over the forecast period (59%-63%) with formal and informal costs reflecting a proportionate increase across PD prevalence levels. Total cohort QALYs decreased slightly with increasing PD prevalence, while caregiver QALYs showed small increases between PD prevalences. Formal care (social care and healthcare) made up a slightly larger proportion of total costs than informal (unpaid care and quality of life costs) in all PD prevalence runs. Figure XXX shows that total dementia prevalence increased approximately at the same rate year on year across all three PD prevalences, starting with a prevalence of 8-9 dementia cases per 1,000 population, rising to 11-12 per 1,000 by 2040.

**Incremental impacts**

| Outcome | 75% - 50% | 50% - 25% |
|---|---|---|
| Dementia onsets avoided (2024-2040) | 191,055 | 191,778 |
| Dementia prevalence reduction in 2040 | 86,564 | 86,548 |
| Total cost savings (£bn) | 22.56 | 23.28 |
| Formal  (%) | 12.49 (55) | 12.89 (55) |
| Informal (%) | 10.08 (45) | 10.39 (45) |
| Cost saved per incident onset avoided (2024-2040) (£) | 118,088.64 | 121,397.76 |
| Total individual QALYs (2024-2040) | -196,795 | -190,670 |
| Caregiver QALYs | 414,920 | 423,256 |
| QALYs per incident onset avoided (2024-2040) | -1.03 | -0.99 |

**Table X Summary of incremental outcomes**

The size of prevalence differences (Figure XXX) showed symmetry around the baseline 50% run, with the 25% and 75% PD prevalence scenarios demonstrating approximately equal opposite deviations across the age bands. The preventable dementia burden (per 100,000 person-years) grew 9-fold with age. Across both sexes at ages 65-69, differences ranged from -82 to +86 cases per 100,000 person-years, with ranges of -791 to +770 cases per 100,000 person-years for the age 90+ band. The preventable dementia burden (per 100,000 population) grew with age as the age 90 years and above age group showed the greatest dementia prevalence differences across PD prevalence runs. Similar sex-specific patterns were observed in both alternative runs.
The incremental differences between runs (75%-50% and 50%-25%) show similar results across all outputs. Between 2024-2040, we estimate that approximately 190,000 incident onsets of dementia could be avoided between the 50% drop in PD prevalence. Using the final 2040 prevalence number, we estimate that a 25% drop in PD prevalence results in a 8-9% fall in dementia prevalence. We estimate this will bring a total cost saving of £22bn-£23bn, with 55% of savings coming from formal care. As a result, an average of between £118,000-£121,000 can be saved per incident onset avoided over the forecast period. The cost per dementia case avoided rises from approximately £18,000 in 2024, to £140,000 in 2040. This reflects the increasing dementia severity and accumulation of costs as individuals live through the forecast period with dementia.
**Figure XXX Age-specific differences in dementia prevalence by PD scenario and sex aggregated across 2024-2040 (50% periodontal prevalence used as baseline)**

Annual total cost savings showed consistent patterns across both PD prevalence reduction scenarios. Total cost savings between prevalence runs was estimated to be £22.5bn-£23.3bn over the forecast period, which is £1.43bn per year. Annual total cost savings did not increase linearly (Figure XXX, Panel C, D and E) in either prevalence reduction scenario. Annual cost saving peaked between 2037-2039 at approximately £1.8bn, which is more than a 10-fold increase from 2024 (£156mn-179mn). Total cost savings did increase as incident dementia onsets and prevalence increased. Total informal care (unpaid care and quality of life costs) composed the majority of costs in all three PD prevalence runs (Supplementary Material Figure XXX) until 2030, at which point formal care (social care and healthcare) costs total more. This is reflected in the annual cost savings by care type in both prevalence reductions, as there are greater formal care savings from 2029 onwards in both the 75%-50% and 50%-25% scenario.  We saw this reflected in the increase in severity and institutionalisation of dementia cases as the cohort aged (Supplementary Material, Figure XXX).

**Figure XXX Incremental impacts of reducing periodontal disease on dementia patient and caregiver quality-adjusted life year outcome**

Reducing PD prevalence from 75% to 50% resulted in a total QALY gain of 218,125 across 2024-2040. This was composed of a reduction of 196,794 dementia patient QALYs as there was fewer people living with dementia in the lower PD prevalence scenario. This was offset by a gain of 414,920 caregiver QALYs as fewer caregivers were required with the lower total dementia prevalence. The 50% to 25% showed consistent QALY patterns, with a net QALY gain of 232,586.

**Figure XXX Incremental impacts of reducing periodontal disease on dementia outcomes**

**Sensitivity analysis results**
Our PSA results are summarised in Table XXX. Low coefficient of variations demonstrate the stability of our base case results across all three prevalence scenarios, displaying robustness to parameter uncertainty. Full PSA results can be found in the supplementary material (pp.xx-xx).

| Outcome (cumulative 2024-2040) | 25% PD Prevalence | 50% PD Prevalence | 75% PD Prevalence |
|---|---|---|---|
| Incident dementia cases, (n) (95% CI) | 10,009,585 
(9,620,585- 10,400,602) | 10,202,161
(9,767,848-10,641,410) | 10,394,192
(9,918,390-10,909,080) |
| Incident dementia cases coefficient of variation (%) | 2 | 2 | 2 |
| Total costs, £bn (95% CI) | 384.56
(327.05-443.22) | 407.85
(343.71-473.26) | 430.93
(359.67-504.05) |
| Formal, £bn (95% CI) | 214.52
(179.85-253.35) | 227.40
(188.08-272.03) | 240.26
(199.21-287.26) |
| Informal, £bn (95% CI) | 170.04
(141.60-195.80) | 180.46
(149.27-210.03) | 190.67
(157.08-223.10) |
| Total cost coefficient of variation (%) | 8 | 8 | 8 |
| Total QALYs, m (95% CI) | 114.11
(101.39-126.63) | 113.92
(101.27-126.26) | 113.73
(101.16-125.97) |
| Total QALYs coefficient of variation (%) | 6 | 6 | 6 |

**Table XXX Probabilistic sensitivity analysis summary across periodontal disease (PD) scenarios**

Varying the PD-dementia onset hazard ratio using lower and upper bound (1.32-1.65) enabled us to calculate the parameter swing using one-way sensitivity analysis. Total incident onsets were the most sensitive (16.67% swing) to varying the onset parameter, and total QALYs the least sensitive (0.32%). Dementia cases vary by approximately 183,500 cases and costs by approximately £20.85bn depending on the HR used. QALYs displayed robustness to parameter with only a 0.3% difference between the lower and upper bound HR.

**Figure XXX One-way sensitivity analysis of periodontal disease onset hazard ratio on total costs, quality-adjusted life years, and incident dementia cases**



**References**
*[1] Department of Health and Social Care, NHS England. Delivering better oral health: an evidence-based toolkit for prevention. Chapter 5: periodontal diseases. London: Department of Health and Social Care; updated 10 September 2025. Available from: https://www.gov.uk/government/publications/delivering-better-oral-health-an-evidence-based-toolkit-for-prevention/chapter-5-periodontal-diseases.*
*[2] World Health Organization. Global oral health status report: towards universal health coverage for oral health by 2030. Geneva: World Health Organization; 2022 [Accessed 30 October 2025]. Available from: https://www.who.int/publications/i/item/9789240061484.*
*[3] United Nations Department of Economic and Social Affairs, Population Division. World population prospects. 2022 [Accessed October 21 2025]. Available from: https://population.un.org/wpp/Download/Standard/Population/.*
*[4] Elamin A, Ansah JP. Projecting the burden of dental caries and periodontal diseases among the adult population in the United Kingdom using a multi-state population model. Front Public Health. 2023;11:1190197. doi:10.3389/fpubh.2023.1190197.*
*[5] Pazos P, Leira Y, Domínguez C, Pías-Peleteiro JM, Blanco J, Aldrey JM. Association between periodontal disease and dementia: a literature review. Neurologia (Engl Ed). 2018;33(9):602–613. doi:10.1016/j.nrleng.2016.09.017.*
*[6] Nadim R, Tang J, Dilmohamed A, Yuan S, Wu C, Bakre AT, Partridge M, Ni J, Copeland JR, Anstey KJ, Chen R. Influence of periodontal disease on risk of dementia. European journal of epidemiology. 2020;35(9):821-33.*
*[7] Watts A, Crimmins EM, Gatz M. Inflammation as a potential mediator for the association between periodontal disease and Alzheimer’s disease. Neuropsychiatr Dis Treat. 2008;4(5):865–876. doi:10.2147/ndt.s3610.*
*[8] Carter CJ, France J, Crean S, Singhrao SK. The Porphyromonas gingivalis/host interactome shows enrichment in GWASdb genes related to Alzheimer’s disease, diabetes and cardiovascular diseases. Front Aging Neurosci. 2017;9:408. doi:10.3389/fnagi.2017.00408.*
*[9] Borsa L, Dubois M, Sacco G, Lupi L. Analysis of the link between periodontal diseases and Alzheimer’s disease: a systematic review. Int J Environ Res Public Health. 2021;18(17):9312. doi:10.3390/ijerph18179312.*
*[10] Dibello V, Custodero C, Cavalcanti R, Lafornara D, Dibello A, Lozupone M, Daniele A, Pilotto A, Panza F, Solfrizzi V. Impact of periodontal disease on cognitive disorders, dementia, and depression: a systematic review and meta-analysis. Geroscience. 2024;46(5):5133–5169. doi:10.1007/s11357-024-01237-1.*
[11] Gao C, Kang J. Oral diseases are associated with cognitive decline and dementia. In: Oral microbiome: symbiosis, dysbiosis and microbiome interventions for maintaining oral and systemic health. 2025 Mar 21. p.171–183. doi:10.1007/978-3-031-55574-7_11.
*[12] Schwahn C, Frenzel S, Holtfreter B, et al. Effect of periodontal treatment on preclinical Alzheimer’s disease—results of a trial emulation approach. Alzheimers Dement. 2022;18:127–141. doi:10.1002/alz.12378.*
*[13] Ho HA, Kim BR, Shin H. Association of periodontal disease treatment with mortality in patients with dementia: a population-based retrospective cohort study (2002–2018). Sci Rep. 2024;14:5243. doi:10.1038/s41598-024-55272-6.*
*[14] Chen CK, Wu YT, Chang YC. Association between chronic periodontitis and the risk of Alzheimer’s disease: a retrospective, population-based, matched-cohort study. Alzheimers Res Ther. 2017;9(1):56. doi:10.1186/s13195-017-0282-6.*
*[15] Lee YT, Lee HC, Hu CJ, Huang LK, Chao SP, Lin CP, Su EC, Lee YC, Chen CC. Periodontitis as a modifiable risk factor for dementia: a nationwide population-based cohort study. J Am Geriatr Soc. 2017;65(2):301–305. doi:10.1111/jgs.14517.*
*[16] Choi S, Kim K, Chang J, Kim SM, Kim SJ, Cho HJ, Park SM. Association of chronic periodontitis with Alzheimer’s disease or vascular dementia. J Am Geriatr Soc. 2019;67(6):1234–1239. doi:10.1111/jgs.15802.*
*[17] Wittenberg R, Hu B, Barraza-Araiza L, Rehill A. Projections of older people living with dementia and costs of dementia care in the United Kingdom, 2019–2040. London: Care Policy and Evaluation Centre, London School of Economics and Political Science; 2019 Nov. Report No.: 79.*
*[18] Livingston G, Huntley J, Liu KY, Costafreda SG, Selbæk G, Alladi S, Ames D, Banerjee S, Burns A, Brayne C, Fox NC. Dementia prevention, intervention, and care: 2024 report of the Lancet Standing Commission. Lancet. 2024;404(10452):572–628. doi:10.1016/S0140-6736(24)01302-2.*
*[19] Chen Y, Araghi M, Bandosz P, Shipley MJ, Ahmadi-Abhari S, Lobanov-Rostovsky S, Venkatraman T, Kivimaki M, O’Flaherty M, Brunner EJ. Impact of hypertension prevalence trend on mortality and burdens of dementia and disability in England and Wales to 2060: a simulation modelling study. Lancet Healthy Longev. 2023;4(9):e470–e477. doi:10.1016/S2666-7568(23)00163-5.*
[20] Office for National Statistics. Estimates of the population for England and Wales [Internet]. 2023 [Accessed October 20 2025]. Available from: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/estimatesofthepopulationforenglandandwales.
[21] Office for National Statistics. Population projections for regions by five-year age groups and sex, England. 2023 [Accessed October 20 2025]. Available from: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/regionsinenglandtable1.
[22] NHS Digital. Primary Care Dementia Data, December 2023. England: NHS Digital; 2024 [Accessed October 20 2025]. Available from: https://digital.nhs.uk/data-and-information/publications/statistical/primary-care-dementia-data/december-2023
*[23] Brück CC, Wolters FJ, Ikram MA, de Kok IM. Projected prevalence and incidence of dementia accounting for secular trends and birth cohort effects: a population-based microsimulation study. Eur J Epidemiol. 2022;37(8):807–814. doi:10.1007/s10654-022-00894-8.*
*[24] Ermini Leaf D, Tysinger B, Goldman DP, Lakdawalla DN. Predicting quantity and quality of life with the Future Elderly Model. Health Econ. 2021;30(Suppl 1):52–79. doi:10.1002/hec.4445.*
*[25] Olfson M, Stroup TS, Huang C, Wall MM, Gerhard T. Age and incidence of dementia diagnosis. J Gen Intern Med. 2021;36(7):2167–2169. doi:10.1007/s11606-021-06670-y.*
*[26] Besley S, Kourouklis D, O’Neill P, Garau M. Dementia in the UK: estimating the potential future impact and return on research investment. London: Office of Health Economics; 2023. Available from: https://www.ohe.org/publications. (Accessed 1 Dec 2025).*
*[27] Tariot PN, Boada M, Lanctôt KL, et al. Relationships of change in Clinical Dementia Rating (CDR) on patient outcomes and probability of progression: observational analysis. Alzheimers Res Ther. 2024;16:36. doi:10.1186/s13195-024-01399-7.*
*[28] Matthews FE, Stephan BCM, Robinson L, Jagger C, Barnes LE, Arthur A, Brayne C. A two decade dementia incidence comparison from the Cognitive Function and Ageing Studies I and II. Nat Commun. 2016;7:11398. doi:10.1038/ncomms11398.*
*[29] Licher S, et al. Lifetime risk of common neurological diseases in the elderly population. J Neurol Neurosurg Psychiatry. 2019;90(2):148–156. doi:10.1136/jnnp-2018-318650.*
*[30] Crowell V, Reyes A, Zhou SQ, et al. Disease severity and mortality in Alzheimer’s disease: an analysis using the U.S. National Alzheimer’s Coordinating Center Uniform Data Set. BMC Neurol. 2023;23:302. doi:10.1186/s12883-023-03353-w.*
[31] Öksüz N, Ghouri R, Taşdelen B, Uludüz D, Özge A. Mild cognitive impairment progression and Alzheimer’s disease risk: a comprehensive analysis of 3553 cases over 203 months. J Clin Med. 2024;13(2):518. doi:10.3390/jcm13020518.
*[32] Chen Y, Bandosz P, Stoye G, Liu Y, Wu Y, Lobanov-Rostovsky S, French E, Kivimaki M, Livingston G, Liao J, Brunner EJ. Dementia incidence trend in England and Wales, 2002–19, and projection for dementia burden to 2040: analysis of data from the English Longitudinal Study of Ageing. Lancet Public Health. 2023 Nov 1;8(11):e859–67.*
*[33] Zhang RQ, Ou YN, Huang SY, Li YZ, Huang YY, Zhang YR, Chen SD, Dong Q, Feng JF, Cheng W, Yu JT. Poor oral health and risk of incident dementia: a prospective cohort study of 425,183 participants. J Alzheimers Dis. 2023;93(3):977–990. doi:10.3233/JAD-230109.*
*[34] Beydoun MA, Beydoun HA, Hossain S, El-Hajj ZW, Weiss J, Zonderman AB. Clinical and bacterial markers of periodontitis and their association with incident all-cause and Alzheimer’s disease dementia in a large national survey. J Alzheimers Dis. 2020;75(1):157–172. doi:10.3233/JAD-200064.*
*[35] Gong J, Harris K, Peters SAE, Woodward M. Sex differences in the association between major cardiovascular risk factors in midlife and dementia: a cohort study using data from the UK Biobank. BMC Med. 2021;19:110. doi:10.1186/s12916-021-01980-z.*
*[36] Zhong G, Wang Y, Zhang Y, Guo JJ, Zhao Y. Smoking is associated with an increased risk of dementia: a meta-analysis of prospective cohort studies with investigation of potential effect modifiers. PLoS One. 2015;10(3):e0118333. doi:10.1371/journal.pone.0118333.*
*[37] Batty GD, Russ TC, Starr JM, et al. Modifiable cardiovascular disease risk factors as predictors of dementia death: pooling of ten general population-based cohort studies. J Negat Results Biomed. 2014;13:8. doi:10.1186/1477-5751-13-8.*
*[38] Cao F, Yang F, Li J, et al. The relationship between diabetes and the dementia risk: a meta-analysis. Diabetol Metab Syndr. 2024;16:101. doi:10.1186/s13098-024-01346-4.*
*[39] Zhang J, Huang X, Ling Y, et al. Associations of cardiometabolic multimorbidity with all-cause dementia, Alzheimer’s disease, and vascular dementia: a cohort study in the UK Biobank. BMC Public Health. 2025;25:2397. doi:10.1186/s12889-025-23352-5.*
*[40] Morys F, Dadar M, Dagher A. Association between midlife obesity and its metabolic consequences, cerebrovascular disease, and cognitive decline. J Clin Endocrinol Metab. 2021;106(10):e4260–74. doi:10.1210/clinem/dgab421.*
*[41] Kim JH, Lee Y. Dementia and death after stroke in older adults during a 10-year follow-up: results from a competing risk model. J Nutr Health Aging. 2018;22(2):297–301. doi:10.1007/s12603-017-0914-3.*
*[42] Office for National Statistics. Adult smoking habits in the UK: 2023. Cigarette smoking habits among adults in the UK, including how many people smoke, differences between population groups, changes over time and use of e-cigarettes. Published 1 October 2024. Available from: https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/healthandlifeexpectancies/bulletins/adultsmokinghabitsingreatbritain/2023 (Accessed 27 October 2025).*
*[43] British Heart Foundation. BHF cardiovascular disease statistics compendium 2023. London: British Heart Foundation; 2023. Available from: https://www.bhf.org.uk/-/media/files/for-professionals/research/heart-statistics/bhf-statistics-compendium-2023 (Accessed 27 October 2025).*
*[44] Department of Health and Social Care. Diabetes profile. Fingertips data. Available from: https://fingertips.phe.org.uk/profile/diabetes-ft/data#page/1/gid/1938133438/pat/159/par/K02000001/ati/15/are/E92000001/yrr/1/cid/4/tbm/1 (Accessed 27 October 2025).*
*[45] Alzheimer’s Society. The economic impact of dementia – Module 1: Annual costs of dementia. London: Alzheimer’s Society; 2024 May [Accessed 2025 Oct 24]. Available from: https://www.alzheimers.org.uk/sites/default/files/2024-05/the-annual-costs-of-dementia.pdf.*
*[46] HM Treasury. GDP deflators at market prices, and money GDP: March 2025 (Spring Statement & Quarterly National Accounts) [Internet]. London: HM Treasury; 2025 Mar [cited 2025 Oct 24]. Available from: https://www.gov.uk/government/statistics/gdp-deflators-at-market-prices-and-money-gdp-march-2025-spring-statement-quarterly-national-accounts.*
*[47] Višnjić A, Veličković V, Milosavljević NŠ. QALY: measure of cost–benefit analysis of health interventions. Acta Fac Med Naiss. 2011;28(4):195–199.*
*[48] Mukadam N, Anderson R, Walsh S, Wittenberg R, Knapp M, Brayne C, Livingston G. Benefits of population-level interventions for dementia risk factors: an economic modelling study for England. Lancet Healthy Longev. 2024;5(9):e567–e577. doi:10.1016/S2666-7568(24)00156-3.*
*[49] Kind P, Hardman G, Macran S. UK population norms for EQ-5D. York: Centre for Health Economics, University of York; 1999 Nov. Report No.: 172.*
*[50] Reed C, Barrett A, Lebrec J, et al. How useful is the EQ-5D in assessing the impact of caring for people with Alzheimer’s disease? Health Qual Life Outcomes. 2017;15:16. doi:10.1186/s12955-017-0591-2.*
*[51] National Institute for Health and Care Excellence (NICE). Guide to the methods of technology appraisal. Process and methods [PMG9]. London: National Institute for Health and Care Excellence; 2013.*
*[52] O'Hagan A, Stevenson M, Madan J. Monte Carlo probabilistic sensitivity analysis for patient level simulation models: efficient estimation of mean and variance using ANOVA. Health Economics. 2007;16(10):1009-1023. doi:10.1002/hec.1199.*

