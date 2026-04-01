# Periodontal Disease Research Project Plan
## Timeline: November 12 - December 31, 2025

**Goal**: Complete at least one paper draft by end of year (49 days remaining)

**Status**: As of November 12, 2025
- ✅ Plan approved
- ✅ Documents converted to markdown
- ✅ Excel execution guide created
- ✅ Results template prepared
- 🔄 Ready to begin Week 1 execution

---

## EXECUTIVE SUMMARY

### Prioritization Decision: CVD Study First

**Rationale**:
- CVD study is 80% complete with operational Excel model
- Realistic 4-week completion timeline
- AD study requires 8-11 weeks (microsimulation model must be built from scratch)
- Low technical risk for CVD vs. high risk for AD

**Deliverable**: Complete manuscript draft for CVD study ready for journal submission by December 20, 2025

---

## 🎉 MAJOR UPDATE - FEBRUARY 11, 2026

### AD Study Manuscript v3 — Discussion Now Complete

**New file**: `AD_FullText_v3.md` (converted from `AD_FullText_v3.docx`). This supersedes `AD_FullText_v2_65.md` as the working manuscript.

**Key changes in v3**:
- ✅ **Complete Discussion Section** added (model validation, interpretation & policy implications, strengths & limitations)
- ✅ **Updated title**: now "dementia" throughout (not "Alzheimer's disease")
- ✅ **Updated key findings** (revised model outputs — see below)
- ✅ **Updated references**: 45 citations
- ✅ **Complete Introduction, Methods, and Results** sections retained from v2

**Updated Key Findings (v3)**:
- Reducing PD prevalence from 50% to 25% prevents **114,814** incident dementia cases (2024-2040)
- Total cost savings: **£19.7bn** (£10.9bn formal + £8.8bn informal care)
- Annual average saving: **£1.16bn**
- Cost per case avoided: **~£172,000** (~5 years of dementia-related care)
- Cohort QALYs: minimal variation across scenarios (<0.2% difference); caregiver QALYs inversely related to PD prevalence

**Outstanding Tasks for v3 Completion**:
1. ⏳ **Write Conclusion section** — currently only a header placeholder in `AD_FullText_v3.md`
2. ⏳ **Complete one-way sensitivity analysis results** — results paragraph contains `XXX` placeholder
3. ⏳ **Resolve all `Table XXX` / `Figure XXX` cross-references** — numbering throughout document
4. ⏳ **Draft Abstract** — absent from v3
5. ⏳ **Supplementary Material** — technical appendix with full model parameters, detailed SA tables, validation results
6. ⏳ **Minor formatting** for target journal

---

## AD STUDY - COMPLETION ROADMAP (Updated February 11, 2026)

### IMMEDIATE PRIORITIES (Week of Feb 11-17, 2026)

**Task 1: Write Conclusion Section**
- [ ] Add conclusion to `AD_FullText_v3.md` (currently only a header placeholder)
- [ ] Should cover:
  - Summary of principal findings (cases prevented, cost savings)
  - PD as a weak but ubiquitous risk factor
  - Policy implications for NHS oral health integration
  - Call for further cost-effectiveness research on specific PD interventions

**Task 2: Complete One-Way Sensitivity Analysis Results**
- [ ] Fill in the `XXX` placeholder in the one-way SA results paragraph
- [ ] Confirm tornado diagram values align with `plots/pd_tornado_diagram.png`

**Task 3: Resolve Cross-Reference Placeholders**
- [ ] Replace all `Table XXX` and `Figure XXX` with correct numbering
- [ ] Verify figure captions match figure content (4 figures extracted to `images/`)

**Task 4: Draft Abstract**
- [ ] Write structured abstract (250-300 words):
  - Background (PD-dementia link, rising costs)
  - Methods (individual-level microsimulation, England 65+, 2024-2040, three PD scenarios)
  - Results (114,814 cases prevented, £19.7bn savings, QALY findings)
  - Conclusions (policy implications)

### SECONDARY PRIORITIES (Week of Feb 18-Mar 3, 2026)

**Task 5: Create/Update Supplementary Material**
- [ ] Create or update `Supplementary_Material_AD.md`
- [ ] Include:
  - Full model parameter tables with sources
  - Risk factor hazard ratios (Table XXX referenced in text)
  - Detailed transition probability matrices
  - Cost calculation breakdowns (formal/informal care by setting)
  - QALY utility values by dementia stage and caregiver disutility
  - Extended PSA results tables (referenced in text)
  - Model validation statistics and goodness-of-fit (referenced in Discussion)
  - Institutionalisation rate figures (referenced in Discussion)

**Task 6: Journal Selection and Formatting**
- [ ] Identify target journals:
  - Primary: *The Lancet Public Health* (high impact, UK focus)
  - Secondary: *PLoS Medicine* (health economics, open access)
  - Tertiary: *Lancet Healthy Longev* (aligns with comparable studies cited)
- [ ] Format manuscript to journal guidelines
- [ ] Prepare cover letter

**Task 7: Internal Review and Polish**
- [ ] Complete manuscript read-through for consistency
- [ ] Verify all numbers match across sections (v3 has updated figures — cross-check all)
- [ ] Check reference formatting (45 references, some URLs blank in converted MD)
- [ ] Proofread for grammar/typos
- [ ] Ensure UK spelling throughout

**Target Completion**: March 3, 2026

---

## WEEK-BY-WEEK PLAN

### WEEK 1: Model Execution & Validation (Nov 18-24, 2025)
**Owner**: User (Excel execution required)
**Estimated Time**: 4-6 hours

#### Tasks:
1. **Open Excel Model** (`PD_CVD_markov - PSA On.xlsm`)
   - Enable macros
   - Verify parameters loaded correctly
   - Cross-check against methodology document
   - **Reference**: `EXCEL_MODEL_EXECUTION_GUIDE.md` (Section: PART 2)

2. **Run Base Case Analysis**
   - Execute single deterministic run
   - Extract outputs:
     - Total QALYs (discounted): Treatment vs. No Treatment
     - Total Costs (£, discounted): Treatment vs. No Treatment
     - Clinical events: Stroke, MI, deaths
     - Calculate ICER = (ΔCosts / ΔQALYs)
     - Calculate NMB at £20k and £30k thresholds
   - **Reference**: `EXCEL_MODEL_EXECUTION_GUIDE.md` (Section: PART 3)

3. **Execute Probabilistic Sensitivity Analysis**
   - Run 10,000 Monte Carlo simulations
   - **Expected runtime**: 30 minutes - 2 hours
   - Extract:
     - Mean and 95% CI for incremental costs
     - Mean and 95% CI for incremental QALYs
     - Cost-effectiveness plane data
     - CEAC probabilities
   - **Reference**: `EXCEL_MODEL_EXECUTION_GUIDE.md` (Section: PART 4)

4. **Run One-Way Sensitivity Analysis**
   - Vary each parameter to min/max values
   - Record ICER range for each parameter
   - Identify most influential parameters (for tornado diagram)
   - **Reference**: `EXCEL_MODEL_EXECUTION_GUIDE.md` (Section: PART 5)

5. **Extract Figures**
   - Export cost-effectiveness plane (PNG, 600 DPI)
   - Export CEAC curve (PNG, 600 DPI)
   - Export tornado diagram (PNG, 600 DPI)
   - Export Markov trace (PNG, 600 DPI)
   - **Reference**: `EXCEL_MODEL_EXECUTION_GUIDE.md` (Section: PART 6)

6. **Export Raw Data**
   - Save CSV files for all results tables
   - **Reference**: `EXCEL_MODEL_EXECUTION_GUIDE.md` (Section: PART 7)

#### Deliverables:
- [ ] Base case results table (CSV)
- [ ] PSA raw data (CSV with 10,000 rows)
- [ ] One-way sensitivity results (CSV)
- [ ] 4 high-resolution figures (PNG)
- [ ] Markov trace data (CSV)

#### Success Criteria:
- ICER between £5,000 - £25,000 per QALY (plausible range)
- Incremental QALYs positive (0.1 - 0.3 expected)
- PSA shows reasonable parameter uncertainty
- Base case ICER close to PSA mean ICER

---

### WEEK 2: Complete Supplementary Materials (Nov 25 - Dec 1, 2025)
**Owner**: User + AI assistance
**Estimated Time**: 6-8 hours

#### Tasks:
1. **Fill Parameter Tables**
   - Open `Supplementary_Material_CVD.md`
   - Populate all empty cells in "Model Parameters and Values" table
   - Extract values from Excel Parameters worksheet
   - Include all sources/references
   - **Current Status**: Table structure exists but values missing

2. **Complete Transition Probability Matrices**
   - Fill in all transition probabilities for Base Arm
   - Fill in all transition probabilities for Treatment Arm
   - Verify probabilities sum to 1.0 for each row
   - Show calculations for complex transitions
   - **Current Status**: Empty tables exist in supplementary document

3. **Document Cost Calculations**
   - Verify periodontal treatment cost breakdown (already present: £1,275/year total)
   - Add CVD event cost calculations:
     - Stroke Year 1, Year 2+ (with references)
     - MI Year 1, Year 2+ (with references)
     - Post-both costs (methodology for combining)

4. **Complete CHEERS 2022 Checklist**
   - Fill in "Location where item is reported" column for all 28 items
   - Cross-reference with Introduction, Methodology, Results sections
   - **Current Status**: Checklist structure complete but locations empty

5. **Add Sensitivity Analysis Ranges**
   - Document min/max values for all parameters
   - Specify distribution types (gamma, beta, log-normal)
   - Include rationale for ranges chosen

6. **Create Model Validation Section**
   - Compare results to published CVD cost-effectiveness studies
   - Cross-check event rates against UK epidemiological data
   - Justify any deviations from expected patterns

#### Deliverables:
- [ ] Complete `Supplementary_Material_CVD.md` with all tables filled
- [ ] Parameter justification document
- [ ] Model validation summary

#### Success Criteria:
- All tables have values (no empty cells marked "XXX")
- CHEERS 2022 checklist 100% complete
- Parameters traceable to primary sources
- Transition probabilities mathematically valid

---

### WEEK 3: Write Results Section (Dec 2-8, 2025)
**Owner**: User with AI writing assistance
**Estimated Time**: 10-12 hours

#### Tasks:
1. **Draft Base Case Analysis** (600-800 words)
   - Use `Results_CVD_TEMPLATE.md` as structure
   - Fill in all [PLACEHOLDER] fields with actual values from Week 1
   - Narrative structure:
     - Cohort characteristics
     - Health outcomes (QALYs, life years, events)
     - Cost outcomes (total, by category)
     - ICER interpretation relative to NICE thresholds
     - Net monetary benefit

2. **Write Cohort Distribution Section** (200-300 words)
   - Describe Markov trace findings
   - Compare state proportions at year 10: Treatment vs. No Treatment
   - Interpret protective effect of periodontal therapy

3. **Draft PSA Section** (600-800 words)
   - Present PSA summary statistics
   - Describe cost-effectiveness plane findings
   - Interpret CEAC results
   - Discuss robustness of conclusions

4. **Write One-Way Sensitivity Section** (400-600 words)
   - Present tornado diagram findings
   - Focus on 5 most influential parameters
   - Interpret impact on cost-effectiveness decision
   - Include threshold analysis if conducted

5. **Create All Tables**
   - Table 1: Base case results (costs, QALYs, events)
   - Table 2: Cost breakdown by category
   - Table 3: PSA summary statistics
   - Table 4: One-way sensitivity results
   - Format in markdown tables with proper alignment

6. **Insert Figures**
   - Figure 1: Markov trace (reference PNG file)
   - Figure 2: Cost-effectiveness plane (reference PNG file)
   - Figure 3: CEAC (reference PNG file)
   - Figure 4: Tornado diagram (reference PNG file)
   - Add figure captions with detailed descriptions

7. **Write Summary Paragraph** (200-300 words)
   - Synthesize all findings
   - Clear statement on cost-effectiveness conclusion
   - Foreshadow discussion themes

#### Deliverables:
- [ ] Complete `Results_CVD.md` (2,000-3,000 words)
- [ ] 4 formatted tables
- [ ] 4 figures with captions
- [ ] Internal consistency check

#### Success Criteria:
- Word count: 2,000-3,000 words
- All placeholders filled with actual values
- Clear narrative flow (base case → PSA → sensitivity)
- Figures and tables support text claims
- Numbers match across tables and text
- Appropriate hedging (95% CIs reported, uncertainty acknowledged)

---

### WEEK 4: Discussion, Abstract & Final Polish (Dec 9-15, 2025)
**Owner**: User with AI writing assistance
**Estimated Time**: 12-15 hours

#### Tasks:
1. **Write Discussion Section** (2,000-3,000 words)
   Structure:
   - **Principal Findings** (1 paragraph)
     - Restate ICER and cost-effectiveness conclusion
     - Highlight key clinical outcomes (stroke/MI reduction)

   - **Interpretation** (2-3 paragraphs)
     - Why is periodontal therapy cost-effective for CVD?
     - Mechanisms: reduced inflammation, improved endothelial function
     - Cost offsets: CVD event costs >> periodontal treatment costs

   - **Comparison with Previous Studies** (2 paragraphs)
     - Compare to periodontal-diabetes study (£1,474/QALY - Reference 20)
     - Why is CVD ICER different? (Disease severity, event costs, time horizon)
     - Position within broader periodontal-systemic health literature

   - **Policy and Clinical Implications** (2 paragraphs)
     - Recommendations for NHS commissioning
     - Integration with CVD prevention pathways
     - Potential for targeted screening (65+ with severe PD)
     - Health equity considerations

   - **Strengths** (1 paragraph)
     - NICE-aligned methodology
     - Probabilistic sensitivity analysis
     - Robust treatment effect evidence from longitudinal studies
     - Conservative assumptions (10-year horizon, median HRs)

   - **Limitations** (2 paragraphs)
     - Single cohort (65-year-old males) - generalizability?
     - 10-year horizon may underestimate lifetime benefits
     - Tunnel states simplify recurrent event dynamics
     - Treatment adherence assumptions (perfect compliance)
     - No treatment effect heterogeneity by PD severity
     - Lack of head-to-head RCT data (reliance on observational studies)

   - **Future Research** (1 paragraph)
     - Individual-level microsimulation for heterogeneity
     - Longer time horizons and lifetime modeling
     - Sex-stratified analyses
     - Real-world effectiveness studies
     - Budget impact analysis for NHS
     - Cost-utility of population-level screening programs

   - **Conclusions** (1 paragraph)
     - Clear summary statement
     - Implications for policy and practice
     - Final take-home message

2. **Draft Abstract** (250-300 words)
   Structure (follow journal guidelines, typically):
   - **Background** (2-3 sentences)
     - CVD burden, PD prevalence, evidence for PD-CVD link
     - Gap: no cost-effectiveness analysis of periodontal therapy for CVD

   - **Methods** (2-3 sentences)
     - Markov model, 8 states, 10-year horizon
     - 65-year-old males, severe PD
     - Non-surgical periodontal therapy vs. no treatment
     - NHS perspective, NICE thresholds
     - PSA (10,000 iterations)

   - **Results** (3-4 sentences with KEY NUMBERS)
     - Incremental QALYs: [0.XX] (95% CI: [X-X])
     - Incremental costs: £[X,XXX] (95% CI: [X-X])
     - ICER: £[XX,XXX] per QALY (95% CI: [X-X])
     - Probability cost-effective at £30,000/QALY: [XX%]
     - Event reductions: Stroke [-XX%], MI [-XX%]

   - **Conclusions** (1-2 sentences)
     - Cost-effectiveness statement
     - Policy implication (1 sentence)

3. **Format References**
   - Compile all citations from Introduction, Methodology, Discussion
   - Format in Vancouver or AMA style (check target journal)
   - Verify all references are cited in text
   - Verify all in-text citations have references
   - Use reference manager (EndNote, Zotero, Mendeley) if available

4. **Internal Review and Revisions**
   - Read full manuscript start to finish
   - Check for:
     - Internal consistency (numbers match across sections)
     - Clear logical flow
     - No orphaned references to "Table X" or "Figure X"
     - Consistent terminology (e.g., "non-surgical periodontal therapy" throughout)
     - UK spelling (favour, colour, organisation)
     - Tense consistency (usually past tense for methods/results, present for discussion)
   - Proofread for grammar and typos

5. **Create Cover Letter**
   - Select target journal (prioritize):
     1. Journal of Clinical Periodontology (IF: 5.8) - good fit
     2. Journal of Dental Research (IF: 6.5) - broader scope
     3. Value in Health (IF: 5.9) - health economics focus
   - Draft cover letter (1 page):
     - Why this journal is appropriate
     - Novelty of study (first cost-effectiveness analysis of periodontal therapy for CVD)
     - Significance for readers (clinicians, policymakers)
     - Confirmation of ethics, authorship, conflicts of interest

6. **Assemble Complete Manuscript**
   - Combine all sections in order:
     - Title page
     - Abstract
     - Introduction
     - Methods
     - Results
     - Discussion
     - References
     - Tables
     - Figure legends
     - Supplementary material
   - Format according to target journal guidelines
   - Create single PDF for submission

#### Deliverables:
- [ ] Discussion section (2,000-3,000 words)
- [ ] Abstract (250-300 words)
- [ ] Formatted reference list (30-50 references expected)
- [ ] Cover letter (1 page)
- [ ] Complete manuscript PDF
- [ ] Supplementary material PDF

#### Success Criteria:
- Discussion addresses all key points (interpretation, comparison, implications, limitations)
- Abstract is concise and includes all key numbers
- No missing references or citation errors
- Manuscript reads as cohesive whole (not disjointed sections)
- Ready for submission (no "TBD" or "[PLACEHOLDER]" fields)

---

### BUFFER WEEK: Final Review (Dec 16-20, 2025)
**Owner**: User + co-authors (if applicable)
**Estimated Time**: 4-6 hours

#### Tasks:
1. **Co-author Review** (if applicable)
   - Send draft to collaborators
   - Incorporate feedback
   - Resolve conflicting suggestions

2. **External Peer Review** (informal, optional)
   - Share with trusted colleague not involved in study
   - Get fresh perspective on clarity and interpretation

3. **Final Checks**
   - Run plagiarism check (Turnitin, iThenticate)
   - Verify all data is accurate and traceable
   - Confirm ethical approval statements (if needed)
   - Check author contributions section
   - Verify funding acknowledgment

4. **Journal Submission Preparation**
   - Create journal account (if needed)
   - Prepare author information for all co-authors
   - Gather ORCID IDs
   - Complete online submission forms
   - Upload all files (manuscript, figures, supplementary)

#### Deliverables:
- [ ] Revised manuscript incorporating feedback
- [ ] All journal submission materials ready
- [ ] Optional: Submit to journal

---

## CURRENT PROJECT STATUS

### Completed Artifacts (Ready to Use)

| File | Description | Completeness | Last Updated |
|------|-------------|--------------|--------------|
| `README.md` | Project overview | 100% | Nov 12, 2025 |
| `CVD_Study/Main_Text_CVD_Paper_finalised.tex` | CVD manuscript (LaTeX) | 100% | Original file |
| `CVD_Study/Supplementary_Material_CVD.tex` | CVD supplementary material | 60% (tables empty) | Converted Nov 12 |
| `CVD_Study/PD_CVD_markov - PSA On.xlsm` | CVD Markov model | 95% (needs execution) | Original file |
| `CVD_Study/EXCEL_MODEL_EXECUTION_GUIDE.md` | Model running instructions | 100% | Nov 12, 2025 |
| `AD_Model_v2/AD_FullText_v2_65.md` | AD Study - Full manuscript draft (superseded) | 85% (superseded by v3) | Converted Feb 4, 2026 |
| `AD_Model_v2/AD_FullText_v2.tex` | AD Study - LaTeX version with figures (superseded) | 85% (superseded by v3) | Created Feb 4, 2026 |
| `AD_Model_v3/AD_FullText_v3.md` | **AD Study - Updated manuscript with Discussion** | 90% (needs conclusion, abstract, cross-refs) | Converted Feb 11, 2026 |

### Pending Artifacts (To Be Created)

#### CVD Study
| File | Description | Assigned Week | Owner |
|------|-------------|---------------|-------|
| Model outputs (CSV files) | Raw data from Excel | Week 1 | User |
| Figures (PNG files) | CE plane, CEAC, tornado, trace | Week 1 | User |
| `CVD_Study/Results_CVD.md` | Complete results section | Week 3 | User + AI |
| `CVD_Study/Discussion_CVD.md` | Discussion section | Week 4 | User + AI |
| `CVD_Study/Abstract_CVD.md` | Abstract | Week 4 | User + AI |
| `CVD_Study/MANUSCRIPT_CVD_FULL.md` | Complete manuscript | Week 4 | User + AI |
| `CVD_Study/Cover_Letter_CVD.md` | Journal cover letter | Week 4 | User |

#### AD Study (Priority Shifted - Near Completion)
| File | Description | Timeline | Status |
|------|-------------|----------|--------|
| ~~Discussion section~~ | ~~Add to manuscript~~ | ~~Week 1 (Feb 4-10)~~ | ✅ **Complete (in v3)** |
| Conclusion section | Add to `AD_Model_v3/AD_FullText_v3.md` | Week 1 (Feb 11-17) | ⏳ Pending |
| One-way SA results text | Fill `XXX` placeholder in v3 | Week 1 (Feb 11-17) | ⏳ Pending |
| Figure/Table cross-references | Resolve all `XXX` numbering in v3 | Week 1 (Feb 11-17) | ⏳ Pending |
| `AD_Model_v3/Supplementary_Material_AD.md` | Full technical appendix | Week 2 (Feb 18-24) | ⏳ Pending |
| ~~Figures (PNG files)~~ | ~~5 figures for manuscript~~ | ~~Week 2 (Feb 11-17)~~ | ✅ **Complete** |
| ~~`AD_Model_v2/AD_FullText_v2.tex`~~ | ~~LaTeX manuscript with figures~~ | ~~Week 1 (Feb 4)~~ | ✅ **Superseded by v3** |
| `AD_Model_v3/Abstract_AD.md` | Structured abstract | Week 2 (Feb 11-24) | ⏳ Pending |
| `AD_Model_v3/Cover_Letter_AD.md` | Journal cover letter | Week 4 (Feb 25-Mar 3) | ⏳ Pending |

---

## KEY MILESTONES & DEADLINES

| Date | Milestone | Status |
|------|-----------|--------|
| Nov 12, 2025 | Project plan approved | ✅ Complete |
| Nov 12, 2025 | Documents converted to markdown | ✅ Complete |
| Nov 12, 2025 | Excel guide and results template ready | ✅ Complete |
| Nov 24, 2025 | **Excel model executed, all outputs extracted** | 🔄 In Progress |
| Dec 1, 2025 | **Supplementary materials complete** | ⏳ Pending |
| Dec 8, 2025 | **Results section drafted** | ⏳ Pending |
| Dec 15, 2025 | **Discussion and abstract complete** | ⏳ Pending |
| Dec 20, 2025 | **Full manuscript draft ready** | ⏳ Pending |
| Dec 31, 2025 | **TARGET: Paper draft complete** | 🎯 Goal |
| **Feb 4, 2026** | **AD Study full manuscript converted to MD** | ✅ **Complete** |
| **Feb 11, 2026** | **AD Study v3 (with Discussion) converted to MD** | ✅ **Complete** |
| **Feb 17, 2026** | **AD Study v3 — Conclusion, SA results, cross-refs resolved** | ⏳ Pending |
| **Feb 24, 2026** | **AD Study abstract drafted, supplementary material updated** | ⏳ Pending |
| **Mar 3, 2026** | **AD Study submission-ready** | 🎯 Target |

---

## RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Excel model doesn't run (macro errors) | Low | High | Debug VBA, consult Excel model documentation, rebuild in Python if necessary |
| Results implausible (ICER > £50k/QALY) | Low | Medium | Check parameter values, verify calculations, consider this finding valid if robust |
| PSA takes too long (>4 hours) | Medium | Low | Run overnight, reduce to 5,000 iterations, use more powerful computer |
| Supplementary tables incomplete data | Medium | Medium | Use literature estimates, document assumptions clearly, note as limitation |
| Time overruns (exceeds Dec 31) | Medium | Medium | Prioritize core sections, defer optional analyses, use buffer week |
| Co-author delays | Low | High | Start circulating drafts early in Week 4, set clear deadlines |

---

## SUCCESS METRICS

### Minimum Viable Product (MVP) - December 31, 2025
- [ ] Complete manuscript (Title, Abstract, Intro, Methods, Results, Discussion, References)
- [ ] All tables and figures included
- [ ] Supplementary material with complete parameter tables
- [ ] CHEERS 2022 checklist filled
- [ ] Internally consistent (numbers match across sections)
- [ ] Ready for internal review

### Stretch Goals (Optional)
- [ ] Submit to journal by December 31
- [ ] Informal peer review completed
- [ ] Budget impact analysis (additional section)
- [ ] Scenario analyses (female cohort, younger age, longer horizon)

---

## RESOURCE REQUIREMENTS

### Software/Tools Needed:
- [x] Microsoft Excel (for model execution)
- [x] Markdown editor (VS Code, Typora, or similar)
- [ ] Statistical software for figure refinement (R/Python - optional)
- [ ] Reference manager (EndNote, Zotero, Mendeley - optional)
- [ ] PDF creation tool (built into most markdown editors)

### Data/Literature Access:
- [x] All necessary references already cited in existing documents
- [x] Model parameters already documented
- [ ] Journal submission fees (typically £0 for open access opt-out, or £2,000-£3,000 for open access)

### Time Commitment:
- **Week 1**: 4-6 hours (model execution)
- **Week 2**: 6-8 hours (supplementary materials)
- **Week 3**: 10-12 hours (results writing)
- **Week 4**: 12-15 hours (discussion, abstract, polish)
- **Total**: 32-41 hours over 4 weeks (~8-10 hours/week)

---

## COMMUNICATION PLAN

### Weekly Check-ins:
- **Friday end-of-week**: Review completed tasks, adjust timeline if needed
- **Monday start-of-week**: Confirm upcoming week priorities

### Escalation:
- **Technical issues** (Excel model errors): Troubleshoot using guide, escalate to Excel expert if needed
- **Writing blockers**: Use AI assistance for drafting, but ensure scientific accuracy
- **Timeline slippage**: Reprioritize tasks, use buffer week, adjust scope if necessary

---

## NEXT STEPS (Immediate Actions)

### For User:
1. **Review this project plan** - Confirm timeline is feasible
2. **Block calendar time** - Reserve 8-10 hours/week for next 4 weeks
3. **Prepare workspace** - Ensure Excel is working, macros enabled
4. **Open Excel model** - Familiarize yourself with structure using `EXCEL_MODEL_EXECUTION_GUIDE.md`
5. **Run base case** - Execute first model run by end of this week

### For AI Assistant:
1. **Monitor progress** - Track todo list updates
2. **Provide writing support** - Draft sections when user provides data
3. **Quality check** - Review drafts for consistency and clarity
4. **Technical support** - Troubleshoot issues as they arise

---

## APPENDIX: File Organization

```
periodontal/
├── README.md                               # Project overview
├── PROJECT_PLAN.md                         # This document
├── TODO.md                                 # Outstanding tasks
├── LICENSE
├── convert_word_to_md.py                   # Utility: .docx → .md conversion
├── backend/                                # Web application backend
│
├── CVD_Study/                              # CVD Markov model study
│   ├── PD_CVD_markov - PSA On.xlsm        # Excel Markov model
│   ├── Main_Text_CVD_Paper_finalised.tex   # CVD manuscript (LaTeX)
│   ├── Supplementary_Material_CVD.tex      # CVD supplementary (LaTeX)
│   ├── EXCEL_MODEL_EXECUTION_GUIDE.md      # How to run model
│   ├── CONTRIBUTING.md                     # CVD model contribution guide
│   ├── convert_cvd_to_md.py                # CVD doc conversion script
│   ├── generate_cvd_figures.py             # CVD figure generation
│   ├── images_CVD/                         # CVD manuscript images
│   ├── Results_CVD.md                      # ⏳ To be created (Week 3)
│   ├── Discussion_CVD.md                   # ⏳ To be created (Week 4)
│   ├── Abstract_CVD.md                     # ⏳ To be created (Week 4)
│   ├── MANUSCRIPT_CVD_FULL.md              # ⏳ To be created (Week 4)
│   └── Cover_Letter_CVD.md                 # ⏳ To be created (Week 4)
│
├── AD_Model_v3/                            # AD microsimulation v3 (current)
│   ├── IBM_PD_AD_v3.py                     # Main simulation model
│   ├── run_psa_direct_v3.py                # PSA runner
│   ├── run_pd_tornado.py                   # Tornado diagram runner
│   ├── rerun_pd_tornado_from_export.py     # Re-run tornado from export
│   ├── pd_sensitivity_analysis.py          # Sensitivity analysis
│   ├── combine_hazard_ratios.py            # HR combination utility
│   ├── external_validation.py              # External validation script
│   ├── generate_validation_data.py         # Validation data generator
│   ├── AD_FullText_v3.md                   # ✅ Working manuscript (Discussion added)
│   ├── AD_FullText_v3.docx                 # Source Word document
│   ├── PD_SENSITIVITY_README.md            # Sensitivity analysis guide
│   ├── EXTERNAL_VALIDATION_README.md       # Validation guide
│   ├── psa_results_25_v3/                  # PSA outputs (25% scenario)
│   ├── psa_results_50_v3/                  # PSA outputs (50% scenario)
│   ├── psa_results_75_v3/                  # PSA outputs (75% scenario)
│   ├── plots/                              # Model output plots
│   ├── figures_AD/                         # ✅ Manuscript figures (5 PNGs)
│   ├── images_AD/                          # ✅ Docx-extracted figures (4 PNGs)
│   ├── Abstract_AD.md                      # ⏳ To be created
│   ├── Supplementary_Material_AD.md        # ⏳ To be created
│   └── Cover_Letter_AD.md                  # ⏳ To be created
│
└── AD_Model_v2/                            # AD microsimulation v1/v2 (archived)
    ├── IBM_PD_AD.py                        # v1 simulation model
    ├── IBM_PD_AD_v2.py                     # v2 simulation model
    ├── run_psa_direct.py                   # v1 PSA runner
    ├── run_psa_direct_v2.py                # v2 PSA runner
    ├── example_psa_visualization.py        # PSA visualisation example
    ├── IBM_PD_AD_V2_README.md              # v2 model readme
    ├── AD_FullText_v2_65.md                # v2 manuscript (superseded)
    ├── AD_FullText_v2.tex                  # v2 LaTeX manuscript (superseded)
    ├── AD_FullText_v2_65.docx              # v2 source Word document
    ├── AD_Microsimulation_results/         # v2 PSA results
    └── results/                            # v1 baseline model results
```

Recommended: Create an `outputs/` folder for Week 1 deliverables to keep project organized.

---

## CONTACT & SUPPORT

**Questions or Issues?**
- Refer first to: `EXCEL_MODEL_EXECUTION_GUIDE.md` (technical issues)
- Refer to: `Results_CVD_TEMPLATE.md` (results writing structure)
- Escalate to: AI assistant for writing support, interpretation questions

---

**Project Timeline Visualization:**

```
Nov 12          Nov 24            Dec 1             Dec 8             Dec 15            Dec 20   Dec 31
  |               |                 |                 |                 |                 |        |
  ✅              🔄                ⏳                ⏳                ⏳                🎯       🎯
Setup       Model Exec        Supplement         Results          Discussion        DRAFT    GOAL
Complete     Week 1            Week 2            Week 3            Week 4           READY   COMPLETE

[████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 15% Complete
```

---

**Last Updated**: February 11, 2026
**Next Review**: February 17, 2026 (AD Study Week 1 Check-in)

**PROJECT STATUS SUMMARY (Feb 11, 2026)**:
- **CVD Study**: Awaiting model execution (Week 1 of original plan)
- **AD Study**: ~90% complete — `AD_FullText_v3.md` now has full Discussion; remaining tasks are Conclusion section, one-way SA placeholder text, figure/table cross-reference numbering, abstract, and supplementary material. Targeting March 3, 2026 submission-ready date.
