# CVD Paper Submission Plan

**Paper**: Cost-effectiveness analysis of non-surgical periodontal treatment for preventing strokes and myocardial infarctions in the United Kingdom

**Primary Target Journal**: Cost Effectiveness and Resource Allocation

**Plan Created**: 2026-01-07

**Target Submission Date**: ___________ (To be determined)

---

## EXECUTIVE SUMMARY

This document provides a comprehensive, step-by-step plan for preparing and submitting the CVD cost-effectiveness paper to scholarly journals. The plan is organized into 7 major phases with specific tasks, timeline estimates, and quality checkpoints.

**Current Status**: Pre-submission preparation phase
**Estimated Time to Submission-Ready**: 3-5 days of focused work
**Primary Target**: Cost Effectiveness and Resource Allocation (BMC)

---

## TABLE OF CONTENTS

1. [Phase 1: Critical Fixes](#phase-1-critical-fixes)
2. [Phase 2: Document Restructuring](#phase-2-document-restructuring)
3. [Phase 3: Journal-Specific Formatting](#phase-3-journal-specific-formatting)
4. [Phase 4: Quality Assurance](#phase-4-quality-assurance)
5. [Phase 5: Submission Preparation](#phase-5-submission-preparation)
6. [Phase 6: Submission Process](#phase-6-submission-process)
7. [Phase 7: Post-Submission Management](#phase-7-post-submission-management)
8. [Contingency Plans](#contingency-plans)
9. [Resources and Checklists](#resources-and-checklists)

---

## PHASE 1: CRITICAL FIXES
**Estimated Time**: 2-3 hours
**Priority**: HIGHEST - Must complete before proceeding

### Task 1.1: Fix Figure Path References ✓
**Location**: CVD_Paper.md lines 116, 121, 135, 140
**Issue**: References point to `images/figure_X.png` but actual location is `images CVD/figure_X.png`

**Action Options**:
- [ ] **Option A (RECOMMENDED)**: Rename folder from "images CVD" to "images"
  ```bash
  mv "images CVD" images
  ```
  - ✅ No need to edit manuscript
  - ✅ Cleaner file structure
  - ✅ No spaces in folder names (best practice)

- [ ] **Option B**: Update all 4 figure references in manuscript
  - Line 116: `![Figure 1](images CVD/figure_1.png)`
  - Line 121: `![Figure 2](images CVD/figure_2.png)`
  - Line 135: `![Figure 3](images CVD/figure_3.png)`
  - Line 140: `![Figure 4](images CVD/figure_4.png)` (Note: Line 140 caption only)

**Decision**: Choose Option A unless there's a reason to keep the space in the folder name.

**Verification**: After fix, verify all 4 figures render correctly in markdown preview.

---

### Task 1.2: Fix Title Formatting Error ✓
**Location**: CVD_Paper.md line 1
**Current**:
```
Cost-effectiveness analysis of non-surgical periodontal treatment for preventing strokes and myocardial infarctions in the U****nited ****K****ingdom**
```

**Corrected**:
```
Cost-effectiveness analysis of non-surgical periodontal treatment for preventing strokes and myocardial infarctions in the United Kingdom
```

**Action**:
- [ ] Edit line 1 to remove formatting artifacts
- [ ] Remove all asterisks from "United Kingdom"
- [ ] Verify no other title instances have same issue

---

### Task 1.3: Fix Reference Formatting Errors ✓
**Locations**:
- Line 282: "Eur** J Prev Cardiol" → "Eur J Prev Cardiol"
- Line 498: "MDM Policy **Pract" → "MDM Policy Pract"

**Action**:
- [ ] Search for all instances of double asterisks in references
- [ ] Remove formatting artifacts
- [ ] Run spell check on entire references section

---

### Task 1.4: Verify Reference 47 Citation ✓
**Locations**: Lines 146 and 282

**Current Usage**:
- Line 146: "the cost-effectiveness of NSPT compares unfavourably with well-established CVD prevention methods [47]"
- Line 282 (Reference 47): "Mihaylova B, et al. Lifetime effects and cost-effectiveness of standard and higher-intensity statin therapy..."

**Action**:
- [ ] Read reference 47 to verify it discusses CVD prevention methods broadly
- [ ] If reference 47 is only about statins, consider if:
  - Reference 48 (Thom et al., DOACs for stroke prevention) is more appropriate, OR
  - You need to cite both [47,48] for "well-established CVD prevention methods"
- [ ] Document decision and rationale

**Resolution**:
_[To be completed - add notes here after verification]_

---

### Task 1.5: Verify Supplementary Table Cross-References ✓

**Current References in Main Text**:
- Line 65: "Final transition probabilities are in the Supplementary Material Table 3."
- Line 77: "Full costing, QALY utility values and parameter breakdown can be found in Supplementary Material Table 2."
- Line 91: "Full distribution parameters for PSA are in Supplementary Table 6."

**Actual Supplementary Tables**:
- Table 1: CHEERS 2022 Checklist (line 297)
- Table 2: All model parameters and values (line 340)
- Table 3: Base Markov model transition matrices (line 375)
- Table 4: Full breakdown of periodontal costs (line 401)
- Table 5: Breakdown of costs per Markov state (line 417)
- Table 6: Parameters and related values used in sensitivity analysis (line 435)

**Action**:
- [ ] Verify line 65 reference: "Table 3" - ✓ CORRECT (Table 3 contains transition matrices)
- [ ] Verify line 77 reference: "Table 2" - ✓ CORRECT (Table 2 contains parameters and utilities)
- [ ] Verify line 91 reference: "Table 6" - ✓ CORRECT (Table 6 contains PSA distributions)
- [ ] All cross-references appear correct ✓

---

## PHASE 2: DOCUMENT RESTRUCTURING
**Estimated Time**: 4-6 hours
**Priority**: HIGH - Required for submission

Most journals require separate files for different components. The current CVD_Paper.md combines everything.

### Task 2.1: Analyze Current Document Structure ✓

**Current Structure** (CVD_Paper.md):
```
Lines 1-16:   Title, Author info, Declarations
Lines 17-27:  Cover Letter
Lines 29-40:  Abstract
Lines 41-169: Main Manuscript
  - Background (41-49)
  - Methods (51-92)
  - Results (94-143)
  - Discussion (144-155)
  - Limitations (156-165)
  - Conclusions (166-169)
Lines 170-188: List of Abbreviations
Lines 189-291: References (main text)
Lines 292-336: Supplementary - CHEERS Checklist
Lines 337-481: Supplementary - Tables 2-6
Lines 482-507: Supplementary - References
```

**Required Separate Files**:
1. Cover Letter (for journal system)
2. Title Page with Author Information
3. Main Manuscript (blinded if required)
4. Supplementary Materials
5. Figures (separate high-resolution files)

---

### Task 2.2: Create Cover Letter Document ✓

**File to Create**: `CVD_Cover_Letter.docx`

**Content to Include**:
```
Dear Editors,

[Lines 18-27 from current CVD_Paper.md]

Yours sincerely,
Edward Coote
Edward.coote@21d.co.uk
21D Clinical Limited
[Full postal address if required]

Date: [Submission date]
```

**Actions**:
- [ ] Extract lines 17-27 from CVD_Paper.md
- [ ] Create new Word document
- [ ] Format as formal business letter
- [ ] Update date to actual submission date
- [ ] Add full contact details and postal address
- [ ] Check journal-specific cover letter requirements:
  - [ ] Statement of originality ✓ (already included, line 25)
  - [ ] Conflict of interest statement ✓ (already included, line 25)
  - [ ] Author contributions ✓ (mentioned as sole author)
  - [ ] Suggested reviewers (see Phase 5)
  - [ ] Funding statement (if required separately)
- [ ] Save as: `CVD_Cover_Letter.docx`

**Template Additions for Cost Effectiveness and Resource Allocation**:
- [ ] Add statement: "This manuscript is not under consideration elsewhere"
- [ ] Add statement: "All authors have approved the manuscript for submission"
- [ ] Add word count (if required)

---

### Task 2.3: Create Title Page Document ✓

**File to Create**: `CVD_Title_Page.docx`

**Content to Include**:
1. **Title**: [Line 1, corrected]
2. **Author Information**:
   - Full name: Edward James Brastock Coote, MSc
   - Affiliation: 21D Clinical Limited
   - Email: Edward.coote@21d.co.uk
   - ORCID ID: [Add if available]
   - Corresponding author: Yes
   - Full postal address
   - Phone number (if required)

3. **Declarations** [Lines 5-13]:
   - Ethics approval
   - Consent for publication
   - Availability of data and materials
   - Competing interests
   - Funding
   - Authors' contributions

4. **Word Count**:
   - Abstract: ~260 words
   - Main text: ~5,000 words (excluding references)
   - Number of tables: 2 (main text), 6 (supplementary)
   - Number of figures: 4

5. **Keywords**:
   - Cost-effectiveness
   - Periodontal disease
   - Cardiovascular disease
   - NHS
   - Markov model
   - [Add 2-3 more if required by journal]

**Actions**:
- [ ] Create title page document
- [ ] Get ORCID ID if not already registered: https://orcid.org/
- [ ] Verify all contact information is current
- [ ] Add any additional keywords required by journal
- [ ] Format according to journal guidelines
- [ ] Save as: `CVD_Title_Page.docx`

---

### Task 2.4: Create Main Manuscript Document ✓

**File to Create**: `CVD_Manuscript.docx`

**Content Structure**:
```
1. Title (if not blinded)
2. Abstract (Lines 29-40)
   - Background
   - Methods
   - Results
   - Conclusions
   - Keywords
3. Background (Lines 41-49)
4. Methods (Lines 51-92)
   - Model structure
   - Health utilities and costs
   - Outcomes
   - Sensitivity Analysis
5. Results (Lines 94-143)
   - Base case analysis (with Table 1)
   - Sensitivity analysis (with Figures 1-4, Table 2)
6. Discussion (Lines 144-155)
7. Limitations (Lines 156-165)
8. Conclusions (Lines 166-169)
9. List of Abbreviations (Lines 170-188)
10. References (Lines 189-291)
```

**Formatting Requirements** (Check journal-specific):
- [ ] Line numbers: YES (for review)
- [ ] Line spacing: Double-spaced
- [ ] Font: Times New Roman 12pt or Arial 11pt
- [ ] Margins: 1 inch (2.54 cm) all sides
- [ ] Page numbers: Bottom center or top right
- [ ] Paragraph indentation: None (use spacing instead)
- [ ] Heading levels: Clearly defined
- [ ] Tables: In-text or at end (check journal preference)
- [ ] Figure legends: After references or with figures
- [ ] Supplementary references: In main reference list or separate

**Actions**:
- [ ] Create Word document from markdown
- [ ] Apply formatting requirements
- [ ] Add line numbers (Word: Layout → Line Numbers → Continuous)
- [ ] Insert Table 1 (lines 100-105) in Results section
- [ ] Insert Table 2 (lines 128-131) in Results section
- [ ] Insert Figure 1-4 placeholders with legends
- [ ] Format all headings consistently
- [ ] Format references according to journal style (likely Vancouver)
- [ ] Add page numbers
- [ ] Verify all in-text citations match reference list
- [ ] Save as: `CVD_Manuscript.docx`

**Special Considerations**:
- [ ] Check if blinding required (Cost Effectiveness and Resource Allocation: NO)
- [ ] Decide: Tables in-text vs. end-of-document (recommend: in-text for readability)
- [ ] Verify figure callouts are in correct locations

---

### Task 2.5: Create Supplementary Materials Document ✓

**File to Create**: `CVD_Supplementary_Materials.docx`

**Content Structure**:
```
Title: Online Supplementary Material
Main Title: Cost-effectiveness analysis of non-surgical periodontal treatment for preventing strokes and myocardial infarctions in the UK

Table of Contents:
- Table S1: CHEERS 2022 Checklist
- Table S2: All model parameters and values
- Table S3: Base Markov model transition matrices
- Table S4: Full breakdown of periodontal costs
- Table S5: Breakdown of costs per Markov state
- Table S6: Parameters and values used in sensitivity analysis
- References for Supplementary Material

[Then all content from lines 292-507]
```

**Actions**:
- [ ] Create supplementary document
- [ ] Add "Supplementary" or "S" prefix to all table numbers
- [ ] Ensure table titles are complete and standalone
- [ ] Format all tables consistently
- [ ] Add supplementary figure if any (check if Excel model screenshots needed)
- [ ] Include supplementary references section (lines 482-507)
- [ ] Verify all table cross-references in main manuscript use "Supplementary Table" or "Table S#"
- [ ] Add page numbers
- [ ] Save as: `CVD_Supplementary_Materials.docx`

**Additional Supplementary Content to Consider**:
- [ ] Model validation information (if available)
- [ ] Additional sensitivity analyses (if performed)
- [ ] Excel model file (see Task 2.6)
- [ ] PRISMA checklist (if applicable - probably not for modeling study)
- [ ] Additional methodology details (if main text word limited)

---

### Task 2.6: Prepare Figure Files ✓

**Current Figures**:
1. `images CVD/figure_1.png` - Tornado Plot and Table Of One-Way Sensitivity Analysis
2. `images CVD/figure_2.png` - One-way sensitivity analysis of treatment effects
3. `images CVD/figure_3.png` - Cost-effectiveness plane
4. `images CVD/figure_4.png` - Cost-effectiveness acceptability curve

**Journal Requirements for Figures** (typical):
- **Format**: PNG, TIFF, or EPS (PNG acceptable for most journals)
- **Resolution**: Minimum 300 DPI at final print size
- **Color mode**: RGB for online, CMYK for print (RGB fine for online-only journals)
- **File naming**: Figure1.png, Figure2.png, etc.
- **Size**: Width typically 84mm (single column) or 174mm (double column)

**Actions**:
- [ ] Check current figure resolution:
  ```bash
  file "images CVD/figure_1.png"
  # Or use: identify -verbose "images CVD/figure_1.png" | grep Resolution
  ```
- [ ] If resolution < 300 DPI, regenerate figures from source code
- [ ] Rename files to journal convention:
  - [ ] figure_1.png → Figure1.png or Fig1.png
  - [ ] figure_2.png → Figure2.png or Fig2.png
  - [ ] figure_3.png → Figure3.png or Fig3.png
  - [ ] figure_4.png → Figure4.png or Fig4.png
- [ ] Create figure legends document or add to manuscript
- [ ] Verify all text in figures is readable at print size
- [ ] Check color-blind accessibility (important for cost-effectiveness planes)
- [ ] Create low-resolution versions for manuscript if required
- [ ] Keep high-resolution originals for production

**Figure Legends** (to verify/refine):
- Figure 1: Tornado plot and table showing results of one-way sensitivity analysis for selected variables. Each bar shows the impact on the incremental cost-effectiveness ratio (ICER) when varying parameter values across their specified ranges.

- Figure 2: One-way sensitivity analysis of treatment effects (stroke and MI hazard ratios) showing impact on net monetary benefit at willingness-to-pay thresholds of £20,000 and £30,000 per QALY gained.

- Figure 3: Cost-effectiveness plane showing results of 10,000 probabilistic sensitivity analysis iterations. Each point represents one simulation. Quadrants indicate dominant (southeast), dominated (northwest), cost-effective (depends on threshold), or cost-ineffective zones.

- Figure 4: Cost-effectiveness acceptability curve showing the probability that NSPT is cost-effective across a range of willingness-to-pay thresholds from £0 to £100,000 per QALY gained.

---

### Task 2.7: Prepare Excel Model File (Optional but Recommended) ✓

**Purpose**: Many health economics journals encourage or require sharing the model

**Actions**:
- [ ] Locate the original Excel model file (appears to be: `PD_CVD_markov - PSA On (1).xlsm`)
- [ ] Clean up the model:
  - [ ] Remove any unnecessary worksheets
  - [ ] Add clear documentation sheet explaining:
    - Model structure
    - How to use the model
    - Input parameters locations
    - Results locations
    - How to run PSA
  - [ ] Add cell comments for key formulas
  - [ ] Ensure no broken links
  - [ ] Remove any personal or confidential information
  - [ ] Test that model runs on fresh open
- [ ] Save clean version as: `CVD_Model_Supplement.xlsx` (or .xlsm if macros needed)
- [ ] Consider creating model screenshot for supplementary materials
- [ ] Decide whether to:
  - Include as supplementary file with submission, OR
  - Upload to public repository (Zenodo, Figshare, GitHub) and reference, OR
  - Make "available upon request"

**Data Sharing Statement Options**:
1. "The Excel model used in this analysis is available as Supplementary File X."
2. "The Excel model is available at [DOI/URL]."
3. "The Excel model is available from the corresponding author upon reasonable request."

---

## PHASE 3: JOURNAL-SPECIFIC FORMATTING
**Estimated Time**: 3-4 hours
**Priority**: HIGH - Must meet journal requirements

### Task 3.1: Download and Review Journal Guidelines ✓

**Target Journal**: Cost Effectiveness and Resource Allocation

**Actions**:
- [ ] Visit: https://resource-allocation.biomedcentral.com/submission-guidelines
- [ ] Download submission guidelines PDF or save as reference
- [ ] Review all sections:
  - [ ] Manuscript preparation guidelines
  - [ ] File formats accepted
  - [ ] Word count limits
  - [ ] Abstract structure requirements
  - [ ] Reference style (likely Vancouver)
  - [ ] Figure and table specifications
  - [ ] Supplementary material policies
  - [ ] Checklist requirements (CHEERS already done)
  - [ ] Data availability requirements
  - [ ] Open access policies

**Key Information to Extract**:
- Maximum word count: ___________
- Abstract word limit: ___________
- Reference style: ___________
- Reference manager format: ___________
- Figure file formats: ___________
- Tables: (in-text / end-of-document / separate files): ___________
- Line numbering required: ___________
- Blinding required: ___________

---

### Task 3.2: Complete CHEERS 2022 Checklist ✓

**Location**: Supplementary Table 1 (lines 294-336)
**Current Status**: Template complete, "Location" column EMPTY

**Action**: Fill in the "Location where item is reported" column for all 28 items

**Methodology**:
1. Open final paginated manuscript
2. For each CHEERS item, identify page/section/paragraph
3. Use specific format: "Page X, [Section Name], paragraph Y" or "Line XX-XX"

**Example Entries**:

| No. | Item | Location where item is reported |
|-----|------|--------------------------------|
| 1 | Identify the study as an economic evaluation... | Title, page 1 |
| 2 | Provide a structured summary... | Abstract, page 1, lines 29-40 |
| 3 | Give the context for the study... | Background, page 2, lines 41-49 |
| 4 | Indicate whether a health economic analysis plan was developed... | Not applicable; no pre-specified plan registered |
| 5 | Describe characteristics of the study population... | Methods, page 3, lines 59-60 |
| ... | ... | ... |

**Actions**:
- [ ] Create working spreadsheet to fill in locations
- [ ] Review manuscript and map each CHEERS item
- [ ] Use "Not applicable" where appropriate with brief justification
- [ ] Common "Not applicable" items for modeling studies:
  - Item 4 (pre-specified analysis plan) - unless registered
  - Item 21 (patient engagement) - unless patient advisory group involved
  - Item 25 (effect of engagement) - unless engagement occurred
- [ ] Transfer completed checklist to supplementary document
- [ ] Verify all 28 items addressed

**Timeline**: Complete this AFTER manuscript is in final format with page numbers

---

### Task 3.3: Format References to Journal Style ✓

**Likely Style**: Vancouver (numbered citations, listed in order of appearance)

**Current Status**: References appear to be in Vancouver style already

**Actions**:
- [ ] Verify journal uses Vancouver style (BMC journals typically do)
- [ ] Check reference format requirements:
  - Journal article format
  - Book chapter format
  - Website citation format
  - Government document format
- [ ] Verify all references follow consistent format:
  - [ ] Authors (list all or max 6 then "et al.")
  - [ ] Article title (sentence case)
  - [ ] Journal title (abbreviated or full - check journal preference)
  - [ ] Year
  - [ ] Volume(Issue)
  - [ ] Page numbers
  - [ ] DOI (required if available)
- [ ] Check all URLs are current and accessible
- [ ] Verify all in-text citations match reference list
- [ ] Check for duplicate references
- [ ] Verify reference numbering is sequential

**Reference Manager**:
- [ ] If using EndNote/Zotero/Mendeley, import journal style
- [ ] Reformat all references automatically
- [ ] Check for formatting errors

**Spot Check References** (High priority ones to verify):
- [ ] Reference 16 (Solowiej-Wedderburn - previous UK CEA)
- [ ] Reference 36 (NICE guidelines)
- [ ] Reference 29 (Sanz et al. consensus)
- [ ] Reference 47 (resolved in Phase 1, Task 1.4)

---

### Task 3.4: Verify Word Counts ✓

**Components to Check**:

| Component | Current | Limit | Status |
|-----------|---------|-------|--------|
| Abstract | ~260 words | Check journal (usually 250-300) | ✓ Likely OK |
| Main text | ~5,000 words | Check journal (usually 5,000-6,000) | ✓ Likely OK |
| Tables (main) | 2 tables | Check journal | ✓ Usually OK |
| Figures | 4 figures | Check journal | ✓ Usually OK |
| References | ~51 refs | Check journal | ✓ Usually OK |
| Supplementary Tables | 6 tables | Usually unlimited | ✓ OK |

**Actions**:
- [ ] Count abstract words (exclude title, subheadings, keywords)
- [ ] Count main text words (exclude abstract, references, tables, figure legends)
- [ ] If over limit, identify sections to shorten:
  - [ ] Combine/shorten background
  - [ ] Move detailed methods to supplementary
  - [ ] Condense discussion
- [ ] Verify table and figure counts acceptable
- [ ] Document final counts for submission form

---

### Task 3.5: Prepare Keywords ✓

**Current Keywords** (line 39): Cost-effectiveness, periodontal disease, cardiovascular disease, NHS, Markov model

**Actions**:
- [ ] Check journal requirements for number of keywords (usually 3-10)
- [ ] Verify keywords are appropriate
- [ ] Consider additional keywords:
  - [ ] Quality-adjusted life years / QALYs
  - [ ] Non-surgical periodontal treatment / NSPT
  - [ ] Stroke
  - [ ] Myocardial infarction
  - [ ] Health economics
  - [ ] Economic evaluation
  - [ ] United Kingdom
  - [ ] Primary prevention
- [ ] Check if keywords should use MeSH terms
- [ ] Finalize 5-8 keywords for submission
- [ ] Add to title page and submission form

**Recommended Final Keywords**:
1. Cost-effectiveness analysis
2. Periodontal disease
3. Cardiovascular disease
4. Markov model
5. Quality-adjusted life years
6. National Health Service
7. Non-surgical periodontal treatment

---

## PHASE 4: QUALITY ASSURANCE
**Estimated Time**: 4-6 hours
**Priority**: CRITICAL - Final checks before submission

### Task 4.1: Comprehensive Proofreading ✓

**Process**:
1. **First Read** - Content and flow
   - [ ] Read entire manuscript start to finish
   - [ ] Check logical flow between sections
   - [ ] Verify arguments are clear and supported
   - [ ] Ensure conclusions match results

2. **Second Read** - Technical accuracy
   - [ ] Verify all numbers in abstract match Results
   - [ ] Check all numbers in text match tables/figures
   - [ ] Verify all statistical tests are correct
   - [ ] Check all formulas and calculations mentioned
   - [ ] Verify all units are correct (£, QALYs, years, %)

3. **Third Read** - Grammar and style
   - [ ] Run spell check (British English for UK journal)
   - [ ] Run grammar check
   - [ ] Check for consistency in:
     - [ ] Terminology (e.g., "cost-effectiveness" vs "cost effectiveness")
     - [ ] Abbreviations (define at first use, then use consistently)
     - [ ] Tense (mostly past tense for methods/results, present for discussion)
     - [ ] Voice (active vs passive)
   - [ ] Check sentence length (avoid overly long sentences)
   - [ ] Check paragraph length (aim for 4-8 sentences)

4. **Fourth Read** - Formatting
   - [ ] Check all headings are formatted consistently
   - [ ] Check all tables are formatted consistently
   - [ ] Check all figures are high quality
   - [ ] Check all references are formatted consistently
   - [ ] Check page numbers are present
   - [ ] Check line numbers are continuous

**Tools to Use**:
- Microsoft Word spell/grammar check
- Grammarly (free or premium)
- Hemingway Editor (for readability)
- Manual reading (best for catching logical errors)

---

### Task 4.2: Verify All Cross-References ✓

**Tables**:
- [ ] Table 1 mentioned in text (line ~96-97)
- [ ] Table 2 mentioned in text (line ~125)
- [ ] Supplementary Table 1 (CHEERS) mentioned if needed
- [ ] Supplementary Table 2 mentioned in text (line 77)
- [ ] Supplementary Table 3 mentioned in text (line 65)
- [ ] Supplementary Table 4 mentioned if needed
- [ ] Supplementary Table 5 mentioned if needed
- [ ] Supplementary Table 6 mentioned in text (line 91)

**Figures**:
- [ ] Figure 1 mentioned in text (line ~111)
- [ ] Figure 2 mentioned in text (line ~113)
- [ ] Figure 3 mentioned in text (line ~125)
- [ ] Figure 4 mentioned in text (line ~125 or mention with Figure 3)

**References**:
- [ ] All in-text citations [#] have corresponding reference
- [ ] All references are cited in text (or remove if not cited)
- [ ] No citation numbers skipped
- [ ] No duplicate citation numbers

**Supplementary Materials**:
- [ ] All supplementary materials mentioned in main text
- [ ] All mentions in main text exist in supplementary

---

### Task 4.3: Verify Data Consistency ✓

**Key Numbers to Cross-Check**:

1. **Base Case Results** (Abstract vs. Table 1 vs. Text):
   - [ ] Incremental cost: £6,487
   - [ ] Incremental QALYs: 0.15
   - [ ] ICER: £44,858
   - [ ] Stroke reduction: 41%
   - [ ] MI reduction: 24%
   - [ ] Event-free survival: 56% vs 44%
   - [ ] Mortality reduction: 13%

2. **PSA Results** (Abstract vs. Table 2 vs. Text):
   - [ ] Mean incremental cost: £8,096
   - [ ] Mean incremental QALYs: 0.28
   - [ ] Mean ICER: £34,723
   - [ ] Cost-effective at £20,000: 25%
   - [ ] Cost-effective at £30,000: 52%

3. **25-Year Results** (Text only):
   - [ ] Incremental cost: £13,189
   - [ ] Incremental QALYs: 0.82
   - [ ] ICER: £16,121
   - [ ] Event-free survival: 22% vs 11%

4. **Treatment Effects** (Methods vs. Supplementary):
   - [ ] Stroke HR: 0.55 (range 0.29-0.81)
   - [ ] MI HR: 0.70 (range 0.44-0.95)

5. **Other Key Parameters**:
   - [ ] Discount rate: 3.5%
   - [ ] Time horizon: 10 years (base), 25 years (scenario)
   - [ ] Starting age: 65 years
   - [ ] Cycle length: 1 year
   - [ ] NICE threshold: £20,000-£30,000

**Action**:
- [ ] Create checklist of all key numbers
- [ ] Verify each appears consistently throughout
- [ ] If discrepancies found, investigate and correct

---

### Task 4.4: Check Figures and Tables ✓

**For Each Figure**:
- [ ] Figure 1:
  - [ ] High resolution (300 DPI)
  - [ ] All text readable
  - [ ] Legend is complete
  - [ ] Axes labeled with units
  - [ ] Referenced in text
  - [ ] File name correct
- [ ] Figure 2:
  - [ ] (Same checks as Figure 1)
- [ ] Figure 3:
  - [ ] (Same checks as Figure 1)
  - [ ] Check color-blind accessibility
  - [ ] Verify quadrants are labeled or explained
- [ ] Figure 4:
  - [ ] (Same checks as Figure 1)
  - [ ] Threshold lines marked (£20k, £30k)

**For Each Table**:
- [ ] Table 1:
  - [ ] All columns aligned
  - [ ] Headers clear
  - [ ] Units in headers or specified
  - [ ] Numbers match text
  - [ ] Footnotes if needed
  - [ ] Title is descriptive and standalone
- [ ] Table 2:
  - [ ] (Same checks as Table 1)
- [ ] Supplementary Tables 1-6:
  - [ ] (Same checks as Table 1)
  - [ ] Numbered as S1-S6 or Supplementary Table 1-6

---

### Task 4.5: External Review (Strongly Recommended) ✓

**Option 1: Colleague Review**
- [ ] Identify 1-2 colleagues with health economics expertise
- [ ] Ask for review of:
  - [ ] Overall clarity and flow
  - [ ] Methodological soundness
  - [ ] Appropriateness of assumptions
  - [ ] Clarity of presentation
- [ ] Provide 1-2 week timeline
- [ ] Incorporate feedback

**Option 2: Statistical Review**
- [ ] If available, have statistician review:
  - [ ] PSA methodology
  - [ ] Confidence intervals
  - [ ] Sensitivity analysis approach
- [ ] Incorporate feedback

**Option 3: English Language Review**
- [ ] If English is not first language, consider professional editing service
- [ ] BMC journals offer English language editing service

**Actions**:
- [ ] Decide which review options to pursue
- [ ] Send manuscript to reviewers
- [ ] Set deadline
- [ ] Incorporate feedback
- [ ] Thank reviewers in acknowledgments (if appropriate)

---

## PHASE 5: SUBMISSION PREPARATION
**Estimated Time**: 3-4 hours
**Priority**: HIGH - Final preparation

### Task 5.1: Create Submission Checklist ✓

**Documents Ready**:
- [ ] Cover Letter (CVD_Cover_Letter.docx)
- [ ] Title Page (CVD_Title_Page.docx)
- [ ] Main Manuscript (CVD_Manuscript.docx)
- [ ] Supplementary Materials (CVD_Supplementary_Materials.docx)
- [ ] Figure 1 (high resolution)
- [ ] Figure 2 (high resolution)
- [ ] Figure 3 (high resolution)
- [ ] Figure 4 (high resolution)
- [ ] CHEERS Checklist (completed, in supplementary materials)
- [ ] Excel Model (optional: CVD_Model_Supplement.xlsx)

**Information Ready**:
- [ ] All author information (name, affiliation, email, ORCID)
- [ ] Corresponding author designated
- [ ] Suggested reviewers (3-5 names) - see Task 5.2
- [ ] Opposed reviewers (if any)
- [ ] Keywords (5-8)
- [ ] Funding information
- [ ] Conflict of interest statements
- [ ] Data availability statement
- [ ] Word counts
- [ ] Manuscript classification/type: Research Article / Original Research

---

### Task 5.2: Prepare Suggested Reviewers List ✓

**Requirements**: Most journals ask for 3-5 suggested reviewers

**Criteria for Suggested Reviewers**:
- ✓ Expertise in health economics and/or cost-effectiveness analysis
- ✓ Knowledge of CVD or periodontal disease preferred
- ✓ Have published in relevant area
- ✓ No conflicts of interest (no co-authors, not same institution, no personal relationships)
- ✓ Ideally diverse (geography, career stage, gender)

**Information Needed for Each Reviewer**:
1. Full name
2. Title/Position
3. Institution
4. Email address
5. Reason for suggestion (brief statement of expertise)

**Suggested Reviewers Template**:

**Reviewer 1:**
- Name: [To be determined]
- Institution:
- Email:
- Expertise: Health economics, cost-effectiveness analysis in dentistry
- Reason: Has published on cost-effectiveness of periodontal interventions

**Reviewer 2:**
- Name: [To be determined]
- Institution:
- Email:
- Expertise: Cardiovascular disease prevention, health economics
- Reason: Expert in CVD prevention cost-effectiveness

**Reviewer 3:**
- Name: [To be determined]
- Institution:
- Email:
- Expertise: Markov modeling, health technology assessment
- Reason: Expert in modeling methodology for chronic disease

**Reviewer 4:**
- Name: [To be determined]
- Institution:
- Email:
- Expertise: Health economics, NHS policy
- Reason: UK-based health economist familiar with NICE methods

**Reviewer 5:**
- Name: [To be determined]
- Institution:
- Email:
- Expertise: Dental public health, health services research
- Reason: Expert in oral health and systemic disease connections

**How to Find Reviewers**:
- [ ] Search for authors of papers cited in your references
- [ ] Exclude anyone cited too frequently (potential conflict)
- [ ] Search PubMed for: "cost-effectiveness periodontal" OR "cost-effectiveness cardiovascular prevention"
- [ ] Check editorial boards of relevant journals
- [ ] Consider authors from systematic reviews in your field
- [ ] Verify email addresses are current (check recent publications)

**Opposed Reviewers** (if any):
- Name:
- Reason for opposition:

**Actions**:
- [ ] Research and compile list of 5-7 potential reviewers
- [ ] Get complete contact information
- [ ] Verify no conflicts of interest
- [ ] Prioritize top 5 for submission form
- [ ] Write brief justification for each

---

### Task 5.3: Prepare Data Availability Statement ✓

**Current Statement** (line 11):
"The datasets used and/or analysed during the current study are available from the corresponding author on reasonable request (email). All data used was from previously published articles or national sources."

**Options to Enhance**:

**Option A - Model Available Upon Request** (Conservative):
```
All data used in this study were obtained from previously published articles and national sources, as cited in the references. The Excel-based Markov model used for analysis is available from the corresponding author upon reasonable request.
```

**Option B - Model in Supplementary Materials**:
```
All data used in this study were obtained from previously published articles and national sources, as cited in the references. The Excel-based Markov model used for analysis is provided as Supplementary File 1.
```

**Option C - Model in Public Repository** (Best for Open Science):
```
All data used in this study were obtained from previously published articles and national sources, as cited in the references. The Excel-based Markov model used for analysis has been deposited in [Zenodo/Figshare/GitHub] and is available at [DOI/URL].
```

**Actions**:
- [ ] Decide which option aligns with your preferences and journal requirements
- [ ] If choosing Option C, upload to repository and get DOI (see Task 5.4)
- [ ] Update data availability statement in title page
- [ ] Ensure consistency with cover letter

**Recommendation**: Option B or C for maximum transparency and to facilitate reproducibility, which is increasingly expected for health economics research.

---

### Task 5.4: Upload Model to Repository (If Choosing Public Sharing) ✓

**Recommended Repository**: Zenodo (free, integrates with journals, provides DOI)

**Steps**:
1. **Create Zenodo Account**:
   - [ ] Go to https://zenodo.org/
   - [ ] Sign up or log in
   - [ ] Connect to ORCID if available

2. **Prepare Model Package**:
   - [ ] Clean Excel model (see Task 2.7)
   - [ ] Create README.txt explaining:
     - Model purpose
     - How to use
     - Input parameters
     - Output interpretation
     - Software requirements (Excel version)
     - License (e.g., CC-BY 4.0)
   - [ ] Test model runs correctly
   - [ ] Consider adding example screenshots

3. **Upload to Zenodo**:
   - [ ] Click "New Upload"
   - [ ] Upload files
   - [ ] Add metadata:
     - Title: "Markov Model: Cost-effectiveness of NSPT for CVD Prevention"
     - Authors: Edward Coote
     - Description: [Brief description]
     - Keywords: [Same as paper]
     - License: Creative Commons Attribution 4.0
     - Related identifiers: [Link to paper when published]
   - [ ] Save draft or publish
   - [ ] Get DOI

4. **Update Paper**:
   - [ ] Add DOI to data availability statement
   - [ ] Consider adding to supplementary materials
   - [ ] Add acknowledgment if required by repository

**Alternative Repositories**:
- **Figshare**: https://figshare.com/ (similar to Zenodo)
- **GitHub**: https://github.com/ (good for version control, can link to Zenodo)
- **Open Science Framework (OSF)**: https://osf.io/ (good for project management)

---

### Task 5.5: Create Submission Email Draft (Backup) ✓

Most journals use online submission systems, but it's good to have an email draft as backup.

**Template**:
```
Subject: Manuscript Submission - Cost-effectiveness of NSPT for CVD Prevention

Dear Editor-in-Chief,

Please find attached our manuscript titled "Cost-effectiveness analysis of non-surgical
periodontal treatment for preventing strokes and myocardial infarctions in the United
Kingdom" for consideration for publication in Cost Effectiveness and Resource Allocation.

This manuscript presents the first UK cost-effectiveness analysis examining non-surgical
periodontal treatment as a cardiovascular disease prevention strategy. Using a Markov
model with probabilistic sensitivity analysis, we found that while NSPT reduces stroke
and MI incidence substantially, it is not cost-effective over a 10-year horizon under
base-case assumptions, though longer-term analysis suggests potential value.

The manuscript includes:
- Main manuscript (5,000 words)
- 4 figures
- 2 tables (main text)
- Supplementary materials with 6 tables and completed CHEERS checklist
- [Optional: Excel model file]

All authors have approved this submission. This work has not been published previously
and is not under consideration elsewhere. We have no conflicts of interest to declare.

We believe this work is well-suited to Cost Effectiveness and Resource Allocation given
its focus on health economics methodology and exploration of cost-effectiveness in an
under-evaluated area of healthcare.

We suggest the following reviewers: [list 3-5]

Thank you for considering our manuscript.

Sincerely,
Edward Coote, MSc
21D Clinical Limited
Edward.coote@21d.co.uk
[Phone]
```

**Actions**:
- [ ] Customize template
- [ ] Add specific details
- [ ] Save as draft
- [ ] Update after online submission begun

---

### Task 5.6: Final File Organization ✓

**Create Submission Folder Structure**:
```
CVD_Paper_Submission/
├── 01_Cover_Letter/
│   └── CVD_Cover_Letter.docx
├── 02_Title_Page/
│   └── CVD_Title_Page.docx
├── 03_Manuscript/
│   └── CVD_Manuscript.docx
├── 04_Supplementary/
│   └── CVD_Supplementary_Materials.docx
├── 05_Figures/
│   ├── Figure1.png
│   ├── Figure2.png
│   ├── Figure3.png
│   └── Figure4.png
├── 06_Model/
│   ├── CVD_Model_Supplement.xlsx
│   └── Model_README.txt
└── 07_Submission_Info/
    ├── Suggested_Reviewers.docx
    ├── Submission_Checklist.docx
    └── Response_to_Reviewers_Template.docx (for future use)
```

**Actions**:
- [ ] Create folder structure
- [ ] Copy all final files to appropriate folders
- [ ] Create backup (cloud storage, external drive)
- [ ] Create ZIP archive for submission
- [ ] Verify all files open correctly
- [ ] Check file sizes are within journal limits

---

## PHASE 6: SUBMISSION PROCESS
**Estimated Time**: 2-3 hours
**Priority**: MEDIUM - When ready to submit

### Task 6.1: Create Journal Account ✓

**Journal**: Cost Effectiveness and Resource Allocation
**Publisher**: BioMed Central (BMC)
**Submission System**: Editorial Manager (typical for BMC journals)

**Actions**:
- [ ] Go to: https://resource-allocation.biomedcentral.com/submission-guidelines
- [ ] Click "Submit manuscript" or similar
- [ ] Create account or log in
- [ ] Complete profile:
  - [ ] Full name
  - [ ] Affiliation
  - [ ] Email
  - [ ] ORCID (link if available)
  - [ ] Areas of expertise
  - [ ] Willing to review (optional)
- [ ] Verify email address
- [ ] Save login credentials

---

### Task 6.2: Complete Online Submission Form ✓

**Typical Sections** (may vary):

**Section 1: Article Type**
- [ ] Select: Research Article / Original Article
- [ ] Confirm scope fits journal

**Section 2: Title and Abstract**
- [ ] Enter title (plain text, no formatting)
- [ ] Paste abstract (may require structured format)
- [ ] Enter keywords

**Section 3: Authors**
- [ ] Add corresponding author (Edward Coote)
  - Full name
  - Email
  - Affiliation
  - ORCID
  - Mark as corresponding
- [ ] Add any co-authors (N/A for this paper - sole author)
- [ ] Confirm author order
- [ ] Verify all information

**Section 4: Files**
- [ ] Upload cover letter (PDF or Word)
- [ ] Upload title page (PDF or Word)
- [ ] Upload manuscript (Word with line numbers)
- [ ] Upload figures (separate high-res files)
- [ ] Upload supplementary materials
- [ ] Upload CHEERS checklist (if separate from supplementary)
- [ ] Upload model file (if sharing)
- [ ] Label each file appropriately
- [ ] Verify all files uploaded correctly

**Section 5: Declarations**
- [ ] Ethics approval: Not applicable
- [ ] Consent: Not applicable
- [ ] Data availability: [Insert statement]
- [ ] Competing interests: [Insert statement from line 13]
- [ ] Funding: [Insert statement from line 13]
- [ ] Authors' contributions: Sole author
- [ ] Acknowledgments: [If any - acknowledge data sources, colleagues who reviewed]

**Section 6: Suggested Reviewers**
- [ ] Add Reviewer 1: [Name, email, institution, expertise]
- [ ] Add Reviewer 2: [Name, email, institution, expertise]
- [ ] Add Reviewer 3: [Name, email, institution, expertise]
- [ ] Add Reviewer 4: [Name, email, institution, expertise]
- [ ] Add Reviewer 5: [Name, email, institution, expertise]
- [ ] Add opposed reviewers if any

**Section 7: Additional Information**
- [ ] Word count
- [ ] Number of figures
- [ ] Number of tables
- [ ] Has this been presented at conference? (Yes/No)
- [ ] Is this a resubmission? (No)
- [ ] Any additional comments for editor

**Section 8: Review and Submit**
- [ ] Review all sections
- [ ] Generate PDF preview
- [ ] Review PDF carefully
- [ ] Check all files are included
- [ ] Approve submission
- [ ] Accept copyright/license agreement
- [ ] Submit

---

### Task 6.3: Post-Submission Immediate Actions ✓

**Within 1 Hour of Submission**:
- [ ] Save submission confirmation email
- [ ] Note manuscript ID number: _____________
- [ ] Note submission date: _____________
- [ ] Download submission receipt/PDF
- [ ] Create folder for submission records
- [ ] Update submission tracking spreadsheet (if using)
- [ ] Notify any co-authors (N/A for sole author)
- [ ] Update CV with "submitted" status

**Within 24 Hours**:
- [ ] Verify you received automated email from journal
- [ ] Verify email confirms files were received
- [ ] If no confirmation, contact editorial office
- [ ] Add calendar reminder to check status in 2 weeks

**Documentation to Keep**:
```
Submission_Records/
├── Confirmation_Email.pdf
├── Submission_Receipt.pdf
├── Manuscript_ID_[XXXXX]/
│   ├── Cover_Letter_Submitted.docx
│   ├── Manuscript_Submitted.docx
│   ├── Supplementary_Submitted.docx
│   └── Figures_Submitted/
└── Submission_Timeline.xlsx
```

---

## PHASE 7: POST-SUBMISSION MANAGEMENT
**Estimated Time**: Ongoing
**Priority**: MEDIUM - Monitoring and preparation

### Task 7.1: Monitor Submission Status ✓

**Expected Timeline** (Cost Effectiveness and Resource Allocation):
- Submission date: ___________
- Admin check: 1-2 days
- Initial decision (desk reject or send to review): ~10 days
- Peer review: 4-8 weeks
- Author revisions: 2-4 weeks (your timeline)
- Re-review: 2-4 weeks
- Final decision: ~3-6 months total
- Publication: 2-4 weeks after acceptance

**Monitoring Schedule**:
- [ ] Week 1: Verify administrative check passed
- [ ] Week 2: Check for editor decision (desk reject vs. peer review)
- [ ] Week 4: Check for peer review updates
- [ ] Week 8: Check for reviewer reports
- [ ] Monthly thereafter until decision

**Actions**:
- [ ] Log in to submission system weekly
- [ ] Check email daily for editor communications
- [ ] Update status tracking spreadsheet
- [ ] Respond promptly to any editor requests

**Status Updates to Track**:
- Submitted
- Admin check complete
- Editor assigned
- With editor
- Under review
- Reviews complete
- Decision pending
- Revisions required
- Revisions submitted
- Accepted
- In production
- Published

---

### Task 7.2: Prepare for Peer Review Outcomes ✓

**Possible Outcomes**:

**Outcome 1: Desk Reject** (within 2 weeks)
- Not suitable for journal scope
- Major methodological concerns
- Below publication standard

**Action Plan**:
- [ ] Review editor comments carefully
- [ ] Decide: Revise for different journal OR dispute decision
- [ ] If revising, address editor concerns
- [ ] Select next journal from target list (see CVD_Paper_Target_Journals.md)
- [ ] Reformat and resubmit within 1-2 weeks

**Outcome 2: Major Revisions**
- Significant issues but potentially publishable
- Usually 2-4 weeks to revise

**Action Plan**:
- [ ] Read all reviewer comments carefully
- [ ] Create point-by-point response document
- [ ] Address all comments (agree or respectfully disagree with justification)
- [ ] Revise manuscript
- [ ] Update figures/tables if needed
- [ ] Submit within deadline

**Outcome 3: Minor Revisions**
- Small issues, likely acceptance after revision
- Usually 1-2 weeks to revise

**Action Plan**:
- [ ] Address all reviewer comments
- [ ] Create response letter
- [ ] Revise manuscript
- [ ] Resubmit quickly

**Outcome 4: Accept as Is** (rare on first submission)
- [ ] Celebrate!
- [ ] Respond to acceptance email
- [ ] Complete copyright forms
- [ ] Pay APC if not waived
- [ ] Review proofs carefully

**Outcome 5: Reject** (after peer review)
- Major concerns that cannot be addressed
- Work not suitable for journal

**Action Plan**:
- [ ] Review comments to determine if concerns are valid
- [ ] Decide whether to revise for another journal
- [ ] Select next target journal
- [ ] Revise and resubmit elsewhere

---

### Task 7.3: Prepare Response to Reviewers Template ✓

Create this now, before you receive reviews, so you're ready.

**Template**: `Response_to_Reviewers_Template.docx`

```
Response to Reviewers

Manuscript ID: [INSERT]
Title: Cost-effectiveness analysis of non-surgical periodontal treatment for preventing
strokes and myocardial infarctions in the United Kingdom

We thank the editor and reviewers for their thoughtful comments on our manuscript.
We have carefully addressed all comments and believe the manuscript is significantly
improved. Below we provide point-by-point responses to each comment. Changes in the
manuscript are highlighted in yellow.

EDITOR COMMENTS:
[Insert editor comments]

Response to Editor:
[Your response]

REVIEWER 1:
[Insert reviewer comments]

Response to Reviewer 1:

Comment 1.1: [Quote exact comment]
Response: [Your response, indicating changes made or respectful disagreement with justification]
Changes made: [Quote new text or indicate "See manuscript page X, lines Y-Z"]

Comment 1.2: [Quote exact comment]
Response: [Your response]
Changes made: [Description]

[Continue for all comments]

REVIEWER 2:
[Insert reviewer comments]

Response to Reviewer 2:
[Continue same format]

SUMMARY OF CHANGES:
1. [Major change 1]
2. [Major change 2]
3. [Minor formatting/typo corrections]

We hope these revisions have adequately addressed the reviewers' concerns and that
the manuscript is now suitable for publication in Cost Effectiveness and Resource Allocation.

Sincerely,
Edward Coote
```

**Actions**:
- [ ] Create template file
- [ ] Save in submission folder
- [ ] When reviews received, fill in comments
- [ ] Draft responses
- [ ] Have colleague review response before submitting

---

### Task 7.4: Plan for Publication Costs ✓

**Cost Effectiveness and Resource Allocation**:
- **APC**: £1,990 / $2,690 / €2,290
- **Waiver Available**: Check eligibility (some countries, institutions)

**Payment Planning**:
- [ ] Check if your institution has BMC membership (may reduce cost)
- [ ] Check if your institution will pay APC (library, research funds)
- [ ] Check if funder requires open access (and will pay)
- [ ] Check waiver eligibility: https://resource-allocation.biomedcentral.com/submission-guidelines/fees-and-funding
- [ ] Apply for waiver if eligible (before or at submission)
- [ ] Budget personal/company funds if needed
- [ ] Payment typically due after acceptance, before publication

**Invoice Information Needed** (prepare in advance):
- [ ] Billing name
- [ ] Billing address
- [ ] VAT/Tax ID (if applicable)
- [ ] Purchase order number (if institutional payment)
- [ ] Funding acknowledgment (if funder paying)

---

### Task 7.5: Plan for Promotion Upon Publication ✓

**Pre-Publication**:
- [ ] Draft plain language summary (for social media, press release)
- [ ] Identify key "takeaway" messages
- [ ] Create 1-2 figures suitable for social media
- [ ] Prepare lay summary (~200 words)

**Upon Acceptance**:
- [ ] Update CV
- [ ] Update institution profile
- [ ] Update ResearchGate/Academia.edu/ORCID profile
- [ ] Add to LinkedIn

**Upon Publication**:
- [ ] Share on Twitter/X with:
  - [ ] Link to paper
  - [ ] Key findings (thread or single tweet)
  - [ ] Tag journal (@BMC_series, @CERAJOURNAL if available)
  - [ ] Use hashtags: #HealthEconomics #CostEffectiveness #CVD #PeriodontaDisease #OpenAccess
- [ ] Share on LinkedIn (longer post with context)
- [ ] Email colleagues, collaborators
- [ ] Notify institution press office (if applicable)
- [ ] Submit to institution repository (if required)
- [ ] Add to Google Scholar profile (should auto-index)
- [ ] Consider blog post explaining findings for general audience
- [ ] Consider contacting health policy organizations (NICE, dental associations)

**Long-Term**:
- [ ] Monitor citations (Google Scholar alerts)
- [ ] Respond to any comments/questions
- [ ] Build on this work for future research
- [ ] Use for grant applications, promotion cases

---

## CONTINGENCY PLANS

### Plan A: Rejection from Primary Target (Cost Effectiveness and Resource Allocation)

**Timeline**: Resubmit within 1-2 weeks

**Actions**:
1. [ ] Carefully review rejection reasons
2. [ ] Determine if concerns are about fit or quality
3. [ ] If fit issue: Select next journal from target list without major revisions
4. [ ] If quality issue: Address concerns before resubmission
5. [ ] Reformat for new journal:
   - [ ] New cover letter (journal-specific)
   - [ ] Reformat references if different style
   - [ ] Check word limits
   - [ ] Update any journal-specific requirements
6. [ ] Submit to next target within 2 weeks

**Next Target**: Applied Health Economics and Health Policy OR Value in Health

---

### Plan B: Major Revisions Requested

**Timeline**: Complete within deadline (usually 2-4 weeks, can request extension)

**Actions**:
1. [ ] Read all comments immediately
2. [ ] Categorize comments:
   - [ ] Easy fixes (grammar, clarification)
   - [ ] Moderate fixes (additional analysis, table/figure)
   - [ ] Difficult fixes (major re-analysis, new data)
3. [ ] Create timeline for addressing each
4. [ ] Start with difficult fixes first
5. [ ] If new analysis needed:
   - [ ] Update Excel model
   - [ ] Re-run analyses
   - [ ] Update tables/figures
   - [ ] Update text
6. [ ] Address all comments systematically
7. [ ] Create point-by-point response
8. [ ] Have colleague review revisions
9. [ ] Submit within deadline

**If Cannot Meet Deadline**:
- [ ] Request extension (1-2 weeks) with justification
- [ ] Most journals will grant reasonable extension

---

### Plan C: Concerns About Study Limitations

**Common Reviewer Concerns for Modeling Studies**:
1. "Treatment effect estimates from non-UK studies"
2. "Uncertainty around treatment effects is too large"
3. "Time horizon too short"
4. "Model structure too simple"
5. "Missing important costs or outcomes"

**Response Strategies**:

**For #1 (Non-UK data)**:
- Acknowledge limitation explicitly
- Explain why UK data not available
- Argue effects likely similar in UK population
- Use sensitivity analysis to show impact
- Call for UK research in discussion

**For #2 (Too much uncertainty)**:
- Acknowledge uncertainty
- Explain this is precisely why research is needed
- Show PSA captures this uncertainty
- Frame as "value of information" issue
- Emphasize this is exploratory/methodological contribution

**For #3 (Time horizon)**:
- You already have 25-year analysis as sensitivity
- Can expand on this if needed
- Explain rationale for 10-year primary analysis
- Acknowledge lifetime analysis would be ideal

**For #4 (Model structure)**:
- Justify parsimony vs. complexity trade-off
- Explain what was excluded and why
- Show model captures key health states
- Cite similar models in literature

**For #5 (Missing outcomes)**:
- Acknowledge in limitations (already done)
- Explain scope was CVD-specific
- Note oral health benefits would improve value
- Suggest future research to include broader outcomes

---

### Plan D: Methodological Disagreements with Reviewers

**Principle**: Be respectful but stand your ground if you're confident in your methods

**Template Response**:
```
We thank the reviewer for this comment. We respectfully disagree with this suggestion
for the following reasons: [1, 2, 3]. Our approach is consistent with NICE guidance
[ref] and has been used in similar analyses [refs]. We believe our method is appropriate
because [justification]. However, we have added text to the limitations section to
acknowledge this as an area of methodological debate.
```

**When to Disagree**:
- ✓ Reviewer suggests method not aligned with NICE/ISPOR guidelines
- ✓ Reviewer suggests unfeasible analysis (requires data you don't have)
- ✓ Reviewer's suggestion would fundamentally change scope
- ✓ Reviewer's suggestion contradicts standard practice

**When to Agree**:
- ✓ Reviewer identifies genuine error
- ✓ Reviewer suggests improvement that's feasible
- ✓ Reviewer's concern is shared by multiple reviewers
- ✓ Reviewer suggests clarification that improves paper

---

### Plan E: Multiple Rejections

**After 2 Rejections**:
- [ ] Carefully review all reviewer comments across journals
- [ ] Look for common themes
- [ ] Consider if fundamental revision needed
- [ ] Seek external expert opinion
- [ ] Consider whether findings are publication-worthy or if re-analysis needed

**Options**:
1. **Major Revision**: Address common concerns, improve methodology
2. **Scope Change**: Narrow scope to what's most robust
3. **Alternative Outlets**:
   - Conference presentation first (get feedback)
   - Technical report / working paper
   - Blog post to get feedback before resubmission
4. **Pivot**: Use model for different research question

---

## RESOURCES AND CHECKLISTS

### Essential Resources

**Journal Guidelines**:
- Cost Effectiveness and Resource Allocation: https://resource-allocation.biomedcentral.com/submission-guidelines
- CHEERS 2022 Statement: https://www.equator-network.org/reporting-guidelines/cheers/
- NICE Methods Guide: https://www.nice.org.uk/process/pmg6/

**Reference Management**:
- EndNote (paid): https://endnote.com/
- Zotero (free): https://www.zotero.org/
- Mendeley (free): https://www.mendeley.com/

**Writing Tools**:
- Grammarly: https://www.grammarly.com/
- Hemingway Editor: http://www.hemingwayapp.com/
- Microsoft Editor (free in Word)

**Figure Preparation**:
- Inkscape (free vector graphics): https://inkscape.org/
- GIMP (free image editing): https://www.gimp.org/
- Python matplotlib/seaborn (if regenerating from code)

**Repository Options**:
- Zenodo: https://zenodo.org/
- Figshare: https://figshare.com/
- OSF: https://osf.io/
- GitHub: https://github.com/

---

### Master Checklist

**Pre-Submission (All must be ✓)**:
- [ ] All critical fixes completed (Phase 1)
- [ ] Documents separated and formatted (Phase 2)
- [ ] Journal-specific formatting applied (Phase 3)
- [ ] Quality assurance checks passed (Phase 4)
- [ ] All submission materials prepared (Phase 5)
- [ ] Colleague review completed (if doing)
- [ ] All files backed up
- [ ] Submission checklist completed
- [ ] Confidence level: High / Medium / Need more work

**Submission (Complete on submission day)**:
- [ ] Journal account created
- [ ] Online form completed
- [ ] All files uploaded
- [ ] Submission confirmed
- [ ] Confirmation email received
- [ ] Manuscript ID recorded

**Post-Submission (Ongoing)**:
- [ ] Status monitored regularly
- [ ] Response templates prepared
- [ ] Payment plan ready
- [ ] Promotion plan ready

---

### Timeline Template

| Phase | Tasks | Estimated Time | Target Date | Actual Date | Status |
|-------|-------|----------------|-------------|-------------|--------|
| Phase 1 | Critical Fixes | 2-3 hours | | | ⬜ Not started |
| Phase 2 | Document Restructuring | 4-6 hours | | | ⬜ Not started |
| Phase 3 | Journal Formatting | 3-4 hours | | | ⬜ Not started |
| Phase 4 | Quality Assurance | 4-6 hours | | | ⬜ Not started |
| Phase 5 | Submission Prep | 3-4 hours | | | ⬜ Not started |
| Phase 6 | Submit | 2-3 hours | | | ⬜ Not started |
| **TOTAL** | | **18-26 hours** | | | |

**Working Schedule Options**:

**Option A - Intensive (3-5 days)**:
- Day 1: Phases 1-2 (6-9 hours)
- Day 2: Phase 3 (3-4 hours)
- Day 3: Phase 4 (4-6 hours)
- Day 4: Phase 5 (3-4 hours)
- Day 5: Phase 6 (submit)

**Option B - Moderate (1-2 weeks)**:
- Week 1: Phases 1-3
- Week 2: Phases 4-6

**Option C - Relaxed (2-4 weeks)**:
- Week 1-2: Phases 1-2
- Week 3: Phases 3-4
- Week 4: Phases 5-6

---

### Submission Day Final Checklist

**Morning of Submission**:
- [ ] Good night's sleep ✓
- [ ] All files ready and tested
- [ ] Submission information compiled
- [ ] 2-3 hours of uninterrupted time blocked

**Before Clicking Submit**:
- [ ] Cover letter reviewed one final time
- [ ] Manuscript PDF generated and reviewed
- [ ] All figures display correctly
- [ ] All author information correct
- [ ] All declarations complete and accurate
- [ ] Suggested reviewers list ready
- [ ] No typos in submission form

**After Submission**:
- [ ] Confirmation email saved
- [ ] Manuscript ID recorded
- [ ] Submission date noted
- [ ] Files archived
- [ ] Calendar reminders set
- [ ] Take a break - you earned it! 🎉

---

## NOTES AND DECISIONS LOG

Use this section to document decisions made during preparation:

**Date**: [Date]
**Decision**: [What was decided]
**Rationale**: [Why]
**Action Taken**: [What you did]

---

**Example**:
Date: 2026-01-07
Decision: Primary target journal is Cost Effectiveness and Resource Allocation
Rationale: Best fit for scope, appropriate impact factor, mentioned in original cover letter, reasonable APC
Action Taken: Downloaded submission guidelines, created target journals list

---

**Plan Status**:
- Created: 2026-01-07
- Last Updated: 2026-01-07
- Current Phase: Pre-Phase 1
- Target Submission Date: [TBD]
- Progress: 0% complete

---

## APPENDIX: Contact Information

**Journal Editorial Office**:
- Journal: Cost Effectiveness and Resource Allocation
- Publisher: BioMed Central
- Email: [Check journal website]
- Submission site: https://resource-allocation.biomedcentral.com/

**Author Contact**:
- Name: Edward Coote
- Email: Edward.coote@21d.co.uk
- Institution: 21D Clinical Limited

**Support Contacts**:
- BMC Author Support: [Check website]
- English Editing Service: [If needed]
- Repository Support: [If using]

---

**END OF SUBMISSION PLAN**

Good luck with your submission! This is high-quality work that will make an important contribution to the health economics literature on periodontal disease and CVD prevention.
