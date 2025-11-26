Online Supplementary Material to: The impact of periodontal prevalence trends on the costs and prevalence of dementia in England: a modelling study 

	•	Dementia Stage Mix and Severity
Dementia in our model was characterised by four stages (1) cognitively normal, (2) mild, (3) moderate and (4) severe, before the final absorbing death state. We used this model inspired by Brück et al. (2023) as the stage breakdown provides a more accurate evaluation of costs and QALYs [1]. 

Figure 1.A Dementia model disease stage structure
We calculated the proportion of total individuals in each stage based of the Economic Impact Of Dementia CF Report, which gave percentages by each disease stage [2]. NHS Primary Care Dementia Data gave us a total dementia prevalence of 2% in the English population, which was then multiplied by the CF Report stage proportions to give estimates of prevalence by stage in the whole population [3]. Dementia prevalence by age band was then calculated through a calibration process to reflect that of 2023 and 2024. 
Stage
Cognitively Normal
Mild
Moderate
Severe
Reported Proportion (%)
-
0.5
0.37270
0.13035
Final Proportion (%)
98.893
0.00994
0.00745
0.00261
Table 1.A Dementia prevalence by disease stage
Due to limitations in reporting data, disease severity proportions for males and females had to be assumed the same. The CF Report provides dementia prevalence by dementia severity for individuals over 65 years in the whole UK, so it was assumed this will be reflective for all individuals over 35 in England, given over 65 year olds compose 97% of dementia cases for 2023. Our model uses Office of National Statistics (ONS) population proportions to distribute individuals by age and sex accurate to our baseline English population [4]. Combined with the parametric age effects and sex-split of risk factor hazard ratios (HRs) (where possible), we believe this mitigates the potential downsides of our prevalence by stage assumptions. 

The baseline onset probability of dementia was estimated using incidence rates from previous UK-based retrospective studies [5,6]. An incidence rate of 1.4% across men and women over age 50 was calculated using the English Longitudinal Study of Aging (ELSA), with another study using UK primary care electronic health records to calculate a higher incidence rate of 3%. We calibrated the model with empirical dementia incidence data for 2024 from NHS England to get 0.0025 as baseline probability, allowing the role of the age and risk factor effects to perform on incident dementia [3].  

For average time spent in each stage, findings from Tariot et al. (2024) were used to decide due to the strength of the data [7]. The mean age of participants in that study was 72, which may mean these values underestimate the stage time of individuals younger individuals. Considering this, we used 4 years as the time in state for the severe stage, supported by empirical evidence that median survival time after a dementia diagnosis in the UK is approximately 10 years [8]. The applied dementia mortality multipliers was estimated through the results of a review of the current literature on survival times by disease severity. Retrospective studies covering a large time span gave a large range of multipliers for severe dementia (HR = 4.98-9.52), however, they focused on older individuals [9,10,11]. Kuryba et al. (2023) gave a more conservative severe dementia HR of 2.97, however, this involved censoring at two years [12]. We decided on a severe mortality multiplier of 1 for each stage due to the combination of existing high severe mortality, risk factor and age effects. State changes only happen at the end of a cycle. 
Stage
Onset
Mild
Moderate
Severe
Time in state (years)
-
2.2
2.0
4.0
Mortality multiplier
1.0
1.0
1.0
1.0
 Table 1.B Time in each dementia stage and associated mortality multipliers 

	•	Background Mortality Configuration 
Annual absolute hazards by age had to be calculated to provide individuals in the model with a sex-and age-specific hazards that represent the general population risk of death. Using ONS life tables data, the annual probability of death for 2023 for each age and sex were converted to a hazard [13]:

The calculated base hazard (see Appendix Table 1) is later scaled by the dementia and risk multipliers so individuals in more advanced disease stages have a higher all-cause mortality that the population baseline. 

	•	Parametric Proportional Hazards Age Effects On Mortality Values
For the age effects, the per year log-hazard β’s had to be chosen based on the available literature. Similar non-linear methods are used to estimate dementia incidence and prevalence in an English population [14,15]. 
Stage Transition 
Onset
Mild to Moderate
Moderate to Severe
Severe to Death
β
0.06
0.03
0.015
0.010
Table 3.A Beta values for parametric age effects 
Alzheimer’s UK estimates that after age 65, the risk of developing dementia doubles every 5 years, which implies a HR = 2.0 for a 5-year age difference. This would make the corresponding β: 

However, our estimates of onset include much younger proportions of the population which this value would overestimate for. Therefore, to align with our model population we chose a disease onset β of 0.06, which means that at age 75 the hazard would be:

This is a 1.34 fold increase over the 5 year difference, which is reasonable given the natural disease progression at age 75 compared to age 70, and to prevent excessive onset compounding from too young in the model. Other βs were tested for calibration with empirical incidence data. Age is a key influence in an individual’s chance of progressing to mild dementia, with increased age posing greater risk [16]. This is likely due to the accumulation of underlying pathology as age increases, with one memory-clinic based study finding a 3% increase in hazard of dementia per year of age [17]. The mild to moderate stage HR corresponds to a β of 0.03, which is a modest increase reflecting ages contribution but dementia’s intrinsic progression playing a dominant role. By the moderate stage, biological factors like tau tangles, and general health conditions drive disease progression more than age does. We chose a moderate β of 0.015, reflecting a 1.5% increase per year. The relative impact of dementia on death is greater in younger individuals than in very old individuals [9]. The extra hazard from severe AD does not escalate rapidly with age. Therefore, we decided a β of 0.010 was appropriate. Crowell et al. [9] estimated an adjusted 11% increase in overall mortality hazard per year of age, not by severe stage, but this will capture a too strong effect when combined with background mortality and relative risk mortality. 

	•	Hazard Ratio Effects For Risk Factors
For each risk factor, we aimed to use large UK or England population-based cohort studies that reported HRs, or meta-analyses that calculated gathered HRs. The majority were derived from the UK Biobank. We used multivariable adjusted HRs, preferring HRs that were adjusted for variable sets most comparable across risk factors in order to minimise the risk of double-counting. Where HRs from prospective studies of other countries were used, we ensured that the population was representative of our English-based cohort by age and disease pathway. All HRs were at least age and sex adjusted. Each individual starts with a Bernoulli draw for each risk factor, using the age and sex-specific prevalence. One limitation of our model is that it treats exposures as fixed from baseline. To match this, we attempted to pick HRs estimated from baseline exposures rather than time-updated covariates. Where multiple studies were found for single HRs, the study with the strongest evidence, such as largest sample size, longest follow-up length and analysis examining sex differences, was chosen with all values having to be statistically significant to be included. HRs for all-cause dementia were chosen over Alzheimer’s disease and vascular dementia due to our model definition. Studies across whole populations, not just examining risk factors for over a certain age, were also prioritised to better fit our model population. The chosen HR for the relative risk of periodontal disease on the severe to death transition was taken from a model fitted for over 65-year olds, which was deemed appropriate due to the age of those with severe dementia in our model. 
Female
Severe to Death
1.36 (1.10-1.69) [27]
1.75 (1.33-2.29) [22]
1.18 (1.12-1.25) [23]
1.34 (1.05-1.71) [25]
1.79 (1.56-2.06) [26]

Onset
1.47 (1.32-1.65) [21]
1.49 (1.29-1.72) [20]
2.40 (1.91-3.02) [19]
2.10 (1.73-2.54) [19]
1.36 (1.02-1.83) [18]
Male
Severe to Death
1.36 (1.10-1.69) [27]
1.75 (1.33-2.29) [22]
1.18 (1.12-1.25) [23]
1.34 (1.05-1.71) [25]
1.79 (1.56-2.06) [26]

Onset
1.47 (1.32-1.65) [21]
1.34 (1.19-1.51) [20]
2.24 (1.86-2.71) [19]
2.14 (1.85-2.47) [18] [24]
1.48 (1.02-2.15) [18]
Risk Factor
Periodontal Disease
Smoking
Cerebrovascular Disease (Stroke)
Cardiovascular Disease (heart failure)
Diabetes 
Table 4.A Hazard ratios for associated risk factors by sex 

	•	Costs Associated With Dementia Care
The Economic Impact Of Dementia CF Report 2024 provided us with all the relevant cost data associated with the different stages of dementia [2]. We deflated the costs to 2023 using the UK government GDP deflator tool [28]. Costs were split by dementia stage, then by location of care. A further split of NHS or informal was made to assign who those costs were accrued by. The NHS costs were composed of social care and healthcare costs, while informal where composed of unpaid care and quality of life costs. As we are estimating the costs of dementia, we held costs for cognitively normal individuals at zero. 
Cognitively Normal

NHS
Informal 

Home
0
0

Institution
0
0
Mild




Home 
7,466.70
10,189.55

Institution 
23,144.27
874.93
Moderate




Home
7,180.18
33,726.09

Institution
15,552.58
1,643.14
Severe




Home
7,668.60
31,523.39

Institution
53,084.13
501.88
Table 5.A Costs of dementia care per individual (£) [2]

Healthcare
Social care
Unpaid care
Quality of life costs
Primary care activity
Residential care
Unpaid carers
Transport costs
Secondary care activity
Nursing care

Legal and financial 
Community care
Domiciliary care

Energy costs
Mental health services
Caregiver

Criminal justice (police callouts for missing dementia patients)
Prescriptions


Scams
Table 5.B Breakdown of costs by categories [2] 

	•	QALYs Associated With Dementia Care
QALY utility values (from EQ-5D sources) for each dementia stage were taken directly from a previous modelling study [29]. These were reported to be the same for males and females. Age specific and sex specific QALY utility values for the cognitively normal population were taken from University of York EQ-5D value set, commonly used in health economic evaluations [30]. 
Stage 
Mild
Moderate
Severe
QALY utility values
0.71
0.64
0.38
Table 6.A QALY utility values for dementia patients by severity stage
Due to the impact of dementia on informal caregivers, we also used specific values for these informal caregivers by dementia stage. These were obtained from a single study due to an absence of research on caregivers, and will be ineffective at capturing the true effects of caregiving on caregivers due to their primary role of capturing physical health [31].
Stage 
Mild
Moderate
Severe
QALY utility values
0.86
0.85
0.82
Table 6.B QALY utility values for informal caregivers by severity stage

	•	Calibration 
We simulated our model from its base year 2023 to 2024 ensure accuracy. Direct incidence data is not available so the model was calibrated using observed prevalence, population and mortality data for England. We used regression-based calibration to compare our model predictions by regressing the observed values on model-predicted values across age and sex strata. 

Equation 7.A Simple linear regression for calibration 
Perfect calibration would result in α ≈ 0, β ≈ 1, and a  close to 1. If our parametric age effects are correctly specified and calibrated, they should mirror the true age pattern of onset and mortality in the population. The parametric age effects define the shape of risk with age, and the real data define the truth of that shape. So if the parametric specification is properly fitted, it will capture the real-world trend. 
 









Figure 7.A Linear regression fit by sex for prevalence 
 
 

Figure 7.B Linear regression fit by sex for mortality 


Figure 7.C Linear regression by sex for population 










References
[1] Brück CC, Wolters FJ, Ikram MA, de Kok IMCM. Projections of costs and quality-adjusted life years lost due to dementia from 2020 to 2050: a population-based microsimulation study. Alzheimers Dement. 2023;19(10):4532–4541. doi:10.1002/alz.13019.
[2] Alzheimer’s Society. The economic impact of dementia – Module 1: Annual costs of dementia. London: Alzheimer’s Society; 2024 May [cited 2025 Oct 24]. Available from: https://www.alzheimers.org.uk/sites/default/files/2024-05/the-annual-costs-of-dementia.pdf
[3] NHS Digital. Primary Care Dementia Data, December 2023. England: NHS Digital; 2024. Available from: https://digital.nhs.uk/data-and-information/publications/statistical/primary-care-dementia-data/december-2023
[4] Office for National Statistics. Estimates of the population for England and Wales [Internet]. 2023. Available from: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/estimatesofthepopulationforenglandandwales 
[5] Ahmadi-Abhari S, Guzman-Castillo M, Bandosz P, Shipley MJ, Muniz-Terrera G, Singh-Manoux A, Kivimäki M, Steptoe A, Capewell S, O’flaherty M, Brunner EJ. Temporal trend in dementia incidence since 2002 and projections for prevalence in England and Wales to 2040: modelling study. BMJ. 2017 Jul 5;358.
[6] Pham TM, Petersen I, Walters K, Raine R, Manthorpe J, Mukadam N, Cooper C. Trends in dementia diagnosis rates in UK ethnic groups: analysis of UK primary care data. Clinical Epidemiology. 2018 Aug 8:949-60.
[7] Tariot PN, Boada M, Lanctôt KL, et al. Relationships of change in Clinical Dementia Rating (CDR) on patient outcomes and probability of progression: observational analysis. Alzheimers Res Ther. 2024;16:36. doi:10.1186/s13195-024-01399-7
[8] Luo H, Koponen M, Roethlein C, Becker C, Bell JS, Beyene K, Chai Y, Chan AH, Chui CS, Haenisch B, Hartikainen S. A multinational cohort study of trends in survival following dementia diagnosis. Commun Med. 2025 May 28;5(1):203.
[9] Crowell V, Reyes A, Zhou SQ, et al. Disease severity and mortality in Alzheimer’s disease: an analysis using the U.S. National Alzheimer’s Coordinating Center Uniform Data Set. BMC Neurol. 2023;23:302. doi:10.1186/s12883-023-03353-w
[10] Villarejo A, Benito-León J, Trincado R, Posada IJ, Puertas-Martín V, Boix R, Medrano MJ, Bermejo-Pareja F. Dementia-associated mortality at thirteen years in the NEDICES cohort study. J Alzheimers Dis. 2011;26(3):543–551.
[11] Andersen K, Lolk A, Martinussen T, Kragh-Sørensen P. Very mild to severe dementia and mortality: a 14-year follow-up—the Odense study. Dement Geriatr Cogn Disord. 2010;29(1):61–67. doi:10.1159/000264637
[12] Kuryba AJ, Boyle JM, van der Meulen J, Aggarwal A, Walker K, Fearnhead NS, Braun MS. Severity of dementia and survival in patients diagnosed with colorectal cancer: a national cohort study in England and Wales. Clin Oncol (R Coll Radiol). 2023;35:e67–e76. doi:10.1016/j.clon.2023.04.013
[13] Office for National Statistics. National life tables: UK [Internet]. London: Office for National Statistics; 18 March 2025 [cited 2025 Oct 23]. Available from: https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/nationallifetablesunitedkingdomreferencetables
[14] Chen Y, Bandosz P, Stoye G, Liu Y, Wu Y, Lobanov-Rostovsky S, et al. Dementia incidence trend in England and Wales, 2002–19, and projection for dementia burden to 2040: analysis of data from the English Longitudinal Study of Ageing. Lancet Public Health. 2023;8(11):e859–e867. doi:10.1016/S2468-2667(23)00187-2
[15] Satizabal CL, Beiser AS, Chouraki V, Chêne G, Dufouil C, Seshadri S. Incidence of dementia over three decades in the Framingham Heart Study. N Engl J Med. 2016;374(6):523–532. doi:10.1056/NEJMoa150432
[16] Öksüz N, Ghouri R, Taşdelen B, Uludüz D, Özge A. Mild cognitive impairment progression and Alzheimer’s disease risk: a comprehensive analysis of 3553 cases over 203 months. J Clin Med. 2024;13(2):518. doi:10.3390/jcm13020518
[17] Biondo F, Jewell A, Pritchard M, et al. Brain-age is associated with progression to dementia in memory clinic patients. Neuroimage Clin. 2022;36:103175. doi:10.1016/j.nicl.2022.103175
[18] Geraets AFJ, Leist AK. Sex/gender and socioeconomic differences in modifiable risk factors for dementia. Sci Rep. 2023;13(1):80. doi:10.1038/s41598-022-27368-4.
[19] Dong C, Zhou C, Fu C, et al. Sex differences in the association between cardiovascular diseases and dementia subtypes: a prospective analysis of 464,616 UK Biobank participants. Biol Sex Differ. 2022;13:21. doi:10.1186/s13293-022-00431-5
[20] Gong J, Harris K, Peters SAE, et al. Sex differences in the association between major cardiovascular risk factors in midlife and dementia: a cohort study using data from the UK Biobank. BMC Med. 2021;19:110. doi:10.1186/s12916-021-01980-z
[21] Zhang RQ, Ou YN, Huang SY, Li YZ, Huang YY, Zhang YR, Chen SD, Dong Q, Feng JF, Cheng W, Yu JT. Poor oral health and risk of incident dementia: a prospective cohort study of 425,183 participants. J Alzheimers Dis. 2023;93(3):977–990. doi:10.3233/JAD-230109
[22] Batty GD, Russ TC, Starr JM, et al. Modifiable cardiovascular disease risk factors as predictors of dementia death: pooling of ten general population-based cohort studies. J Negat Results Biomed. 2014;13:8. doi:10.1186/1477-5751-13-8
[23] Kim JH, Lee Y. Dementia and death after stroke in older adults during a 10-year follow-up: results from a competing risk model. J Nutr Health Aging. 2018;22(2):297–301. doi:10.1007/s12603-017-0914-3
[24] Luo J, Rasmussen IJ, Nordestgaard BG, Tybjærg-Hansen A, Thomassen JQ, Frikke-Schmidt R. Cardiovascular diseases and risk of dementia in the general population. Eur J Prev Cardiol. 2025;zwaf129. doi:10.1093/eurjpc/zwaf129
[25] Russ TC, Hamer M, Stamatakis E, Starr JM, Batty GD, Kivimäki M. Does the Framingham cardiovascular disease risk score also have predictive utility for dementia death? An individual participant meta-analysis of 11,887 men and women. Atherosclerosis. 2013;228(1):256–258. doi:10.1016/j.atherosclerosis.2013.02.025
[26] Zhang J, Huang X, Ling Y, et al. Associations of cardiometabolic multimorbidity with all-cause dementia, Alzheimer’s disease, and vascular dementia: a cohort study in the UK Biobank. BMC Public Health. 2025;25:2397. doi:10.1186/s12889-025-23352-5
[27] Beydoun MA, Beydoun HA, Hossain S, El-Hajj ZW, Weiss J, Zonderman AB. Clinical and bacterial markers of periodontitis and their association with incident all-cause and Alzheimer’s disease dementia in a large national survey. J Alzheimers Dis. 2020;75(1):157–172. doi:10.3233/JAD-200064.
[28] HM Treasury. GDP deflators at market prices, and money GDP: March 2025 (Spring Statement & Quarterly National Accounts) [Internet]. London: HM Treasury; 2025 Mar [cited 2025 Oct 24]. Available from: https://www.gov.uk/government/statistics/gdp-deflators-at-market-prices-and-money-gdp-march-2025-spring-statement-quarterly-national-accounts
[29] Mukadam N, Anderson R, Walsh S, Wittenberg R, Knapp M, Brayne C, Livingston G. Benefits of population-level interventions for dementia risk factors: an economic modelling study for England. Lancet Healthy Longev. 2024;5(9):e567–e577. doi:10.1016/S2666-7568(24)00156-3
[30] Kind P, Hardman G, Macran S. UK population norms for EQ-5D. York: Centre for Health Economics, University of York; 1999 Nov. Report No.: 172.
[31] Reed C, Barrett A, Lebrec J, et al. How useful is the EQ-5D in assessing the impact of caring for people with Alzheimer’s disease? Health Qual Life Outcomes. 2017;15:16. doi:10.1186/s12955-017-0591-2


Appendix 
Age
Male
Female
35
0.001129638
0.000631199
36
0.001270807
0.000622194
37
0.001342901
0.000744277
38
0.001447046
0.000870379
39
0.001617307
0.000943445
40
0.0017325
0.001004504
41
0.001872753
0.00107758
42
0.001971943
0.001199719
43
0.00211724
0.001238767
44
0.002277592
0.001387963
45
0.002506138
0.001474086
46
0.002683598
0.00163133
47
0.002940319
0.00191884
48
0.003267332
0.001990981
49
0.003525206
0.002132272
50
0.003829323
0.002372813
51
0.00406224
0.002537216
52
0.004496092
0.002685603
53
0.004703042
0.002974419
54
0.005099983
0.003176038
55
0.00545284
0.00349209
56
0.005884278
0.003817277
57
0.006429626
0.004150602
58
0.006883638
0.004402678
59
0.007414419
0.004811557
60
0.008167261
0.005274888
61
0.009077072
0.005626801
62
0.009761489
0.006349113
63
0.010590886
0.00701253
64
0.011667805
0.007683442
65
0.012624353
0.008298336
66
0.013983312
0.009275888
67
0.015533015
0.010128117
68
0.017026125
0.011140829
69
0.018353398
0.012055375
70
0.020152709
0.013366941
71
0.022175059
0.014398158
72
0.024319332
0.015880429
73
0.026198195
0.017527717
74
0.02844783
0.019302093
75
0.03174044
0.021814209
76
0.035692465
0.024221998
77
0.039667453
0.027727889
78
0.044963894
0.031303894
79
0.049874302
0.035050142
80
0.057289128
0.040569943
81
0.064013859
0.045992631
82
0.072640588
0.052261131
83
0.07986389
0.058631734
84
0.089929084
0.066224208
85
0.10094583
0.075113709
86
0.113810561
0.086476611
87
0.12878951
0.097566523
88
0.14638855
0.112847361
89
0.164157919
0.127565226
90
0.187110606
0.146074877
91
0.207099203
0.166562387
92
0.235637527
0.190102953
93
0.260684476
0.213685916
94
0.29167403
0.237158808
95
0.323528963
0.265026004
96
0.353340509
0.291911002
97
0.397973242
0.327432419
98
0.414895939
0.364047771
99
0.475423088
0.39358644
100
0.514090949
0.438461552
Table 1 Background mortality probabilities by age and sex
