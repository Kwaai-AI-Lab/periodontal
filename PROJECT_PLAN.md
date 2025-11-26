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
| `Intro_CVD.md` | Introduction section | 100% | Converted Nov 12 |
| `Methodology_CVD.md` | Methods section | 100% | Converted Nov 12 |
| `Supplementary_Material_CVD.md` | Technical appendix | 60% (tables empty) | Converted Nov 12 |
| `PD_CVD_markov - PSA On.xlsm` | Markov model | 95% (needs execution) | Original file |
| `EXCEL_MODEL_EXECUTION_GUIDE.md` | Model running instructions | 100% | Nov 12, 2025 |
| `Results_CVD_TEMPLATE.md` | Results section template | 100% | Nov 12, 2025 |

### Pending Artifacts (To Be Created)

| File | Description | Assigned Week | Owner |
|------|-------------|---------------|-------|
| Model outputs (CSV files) | Raw data from Excel | Week 1 | User |
| Figures (PNG files) | CE plane, CEAC, tornado, trace | Week 1 | User |
| `Results_CVD.md` | Complete results section | Week 3 | User + AI |
| `Discussion_CVD.md` | Discussion section | Week 4 | User + AI |
| `Abstract_CVD.md` | Abstract | Week 4 | User + AI |
| `MANUSCRIPT_CVD_FULL.md` | Complete manuscript | Week 4 | User + AI |
| `Cover_Letter_CVD.md` | Journal cover letter | Week 4 | User |

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
├── EXCEL_MODEL_EXECUTION_GUIDE.md          # How to run model
│
├── CVD Study (Priority 1)
│   ├── Intro_CVD.md                        # ✅ Complete
│   ├── Methodology_CVD.md                  # ✅ Complete
│   ├── Supplementary_Material_CVD.md       # 🔄 In progress (60%)
│   ├── Results_CVD_TEMPLATE.md             # ✅ Template ready
│   ├── Results_CVD.md                      # ⏳ To be created (Week 3)
│   ├── Discussion_CVD.md                   # ⏳ To be created (Week 4)
│   ├── Abstract_CVD.md                     # ⏳ To be created (Week 4)
│   ├── MANUSCRIPT_CVD_FULL.md              # ⏳ To be created (Week 4)
│   ├── PD_CVD_markov - PSA On.xlsm         # Excel model
│   └── outputs/                            # Folder for CSV and PNG outputs
│       ├── CVD_base_case_results.csv
│       ├── CVD_PSA_raw_data.csv
│       ├── CVD_one_way_sensitivity.csv
│       ├── CVD_CE_Plane.png
│       ├── CVD_CEAC.png
│       ├── CVD_Tornado.png
│       └── CVD_Markov_Trace.png
│
├── AD Study (Deferred to Q1 2026)
│   ├── Intro_AD.md                         # ✅ Complete
│   ├── Methodology_AD.md                   # ✅ Complete
│   ├── Results_AD.md                       # ⏳ Pending (empty placeholders)
│   └── Supplementary_Material_AD.md        # ✅ Complete
│
└── Archives (Original files)
    ├── Intro_AD.docx
    ├── Intro_CVD.docx
    ├── Methodology_AD.docx
    ├── Methodology_CVD.docx
    ├── Results_AD.docx
    ├── Supplementary_Material_AD.docx
    └── Supplementary_Material_CVD.docx
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

**Last Updated**: November 12, 2025
**Next Review**: November 24, 2025 (End of Week 1)
