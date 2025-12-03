# CVD Markov Model Web Application - TODO List

## Project Overview
Converting the Excel-based PD_CVD_markov model into a FastAPI + React web application for academic research.

**Timeline:** 7-8 weeks
**Tech Stack:** Python (FastAPI), React, NumPy, Chart.js, Plotly

---

## Phase 1: Backend - Core Markov Engine (Week 1-2)

### 1.1 Project Setup
- [x] Create backend directory structure
- [x] Create `requirements.txt` with dependencies
- [ ] Setup virtual environment documentation
- [ ] Create `.gitignore` for Python

**Assignable to:** Backend Developer
**Estimated time:** 2-3 hours

### 1.2 Parameter Models (`backend/models/parameters.py`)
- [ ] Implement `BaselineHazards` Pydantic model
  - Fields: stroke, MI, death hazards
  - Validation: positive values, realistic ranges
- [ ] Implement `TreatmentEffects` Pydantic model
  - Fields: stroke_hr (0.29-0.81), mi_hr (0.44-0.95)
  - Validation: within published confidence intervals
- [ ] Implement `Costs` Pydantic model
  - Fields: periodontal treatment, stroke Y1/Y2+, MI Y1/Y2+, both Y1/Y2+
  - Validation: positive values, post-both = max(stroke, MI)
- [ ] Implement `Utilities` Pydantic model
  - Fields: base, post-stroke Y1/Y2+, post-MI Y1/Y2+, disutilities
  - Validation: 0-1 range
- [ ] Implement `ModelParameters` aggregate model
- [ ] Add docstrings and type hints
- [ ] Write unit tests for parameter validation

**Assignable to:** Backend Developer
**Estimated time:** 1 day
**Dependencies:** None

### 1.3 Calculations Utilities (`backend/utils/calculations.py`)
- [ ] Implement `hazard_to_probability(hazard, time)` function
  - Formula: `p = 1 - exp(-hazard * time)`
- [ ] Implement `calculate_icer(incremental_cost, incremental_qalys)` function
  - Handle division by zero, negative incrementals
- [ ] Implement `calculate_nmb(incremental_qalys, incremental_cost, wtp)` function
  - Formula: `NMB = incremental_QALYs × WTP - incremental_cost`
- [ ] Implement `count_events(markov_trace)` function
- [ ] Add docstrings and unit tests

**Assignable to:** Backend Developer (can work in parallel with 1.2)
**Estimated time:** 4-6 hours
**Dependencies:** None

### 1.4 Markov Engine (`backend/models/markov_engine.py`)
- [ ] Create `MarkovCVDModel` class structure
- [ ] Implement `build_transition_matrix(cycle, treatment)` method
  - 8×8 probability matrix
  - Handle tunnel state transitions
  - Apply competing risks (stroke, MI, death)
  - Treatment HR application
- [ ] Implement `simulate_cohort(treatment)` method
  - Initialize state vector (all in state 0)
  - Matrix multiplication for 10 cycles
  - Return markov_trace
- [ ] Implement `calculate_outcomes(markov_trace, treatment)` method
  - Apply costs by state
  - Apply utilities by state
  - Calculate discounting (3.5% annual)
  - Return total costs, QALYs, life years
- [ ] Implement `run_simulation(treatment)` method
  - Combine cohort simulation and outcomes
- [ ] Add comprehensive docstrings
- [ ] Create `SimulationResults` dataclass

**Assignable to:** Senior Backend Developer (core algorithm)
**Estimated time:** 3-4 days
**Dependencies:** 1.2 (parameters), 1.3 (calculations)

### 1.5 Unit Tests (`backend/tests/test_markov_engine.py`)
- [ ] Test hazard-to-probability conversion
- [ ] Test transition matrix properties (row sums = 1.0)
- [ ] Test tunnel state transitions
- [ ] Test competing risks logic
- [ ] Test discounting calculations
- [ ] Test full simulation run

**Assignable to:** QA/Testing Developer
**Estimated time:** 1-2 days
**Dependencies:** 1.4 (markov engine)

### 1.6 Validation Against Excel
- [ ] Extract base case parameters from Excel model
- [ ] Run Python simulation with identical parameters
- [ ] Compare total QALYs (tolerance: <0.1%)
- [ ] Compare total costs (tolerance: <0.1%)
- [ ] Compare ICER (tolerance: <0.1%)
- [ ] Compare Markov trace state-by-state
- [ ] Document validation results

**Assignable to:** Research/Validation Specialist
**Estimated time:** 2-3 days
**Dependencies:** 1.4 (markov engine), 1.5 (tests passing)

---

## Phase 2: Backend - PSA Implementation (Week 2-3)

### 2.1 Distribution Functions (`backend/utils/distributions.py`)
- [ ] Implement `gamma_params_from_mean_rel_sd(mean, rel_sd)`
  - Convert mean and relative SD to shape/scale
- [ ] Implement `beta_params_from_mean_rel_sd(mean, rel_sd)`
  - Convert mean and relative SD to alpha/beta
- [ ] Implement `lognormal_params_from_ci(point, lower, upper)`
  - Extract mu/sigma from 95% CI bounds
- [ ] Implement `sample_gamma(mean, rel_sd, rng)`
- [ ] Implement `sample_beta(mean, rel_sd, rng)`
- [ ] Implement `sample_lognormal(point, lower, upper, rng)`
- [ ] Add docstrings and unit tests

**Assignable to:** Data Science/Stats Developer
**Estimated time:** 1 day
**Dependencies:** None (can start during Phase 1)

### 2.2 PSA Engine (`backend/models/psa_engine.py`)
- [ ] Create `PSAEngine` class structure
- [ ] Implement `sample_parameters(n_iterations, seed)` method
  - Sample costs (gamma distribution)
  - Sample utilities (beta distribution)
  - Sample HRs (lognormal from CI)
  - Return list of ModelParameters
- [ ] Implement `run_single_iteration(params)` static method
  - Run base and treatment simulations
  - Calculate incrementals
  - Return results dict
- [ ] Implement `run_psa(n_iterations, n_jobs)` method
  - Use multiprocessing.Pool for parallelization
  - Progress tracking
  - Return raw results array
- [ ] Implement `calculate_psa_statistics(results)` method
  - Mean, median, 95% CI for costs and QALYs
  - Probability cost-effective at £20k, £30k
- [ ] Implement `calculate_ceac(results, wtp_range)` method
  - Calculate probability CE across WTP thresholds
- [ ] Create `PSAResults` dataclass
- [ ] Add docstrings

**Assignable to:** Senior Backend Developer
**Estimated time:** 3-4 days
**Dependencies:** 1.4 (markov engine), 2.1 (distributions)

### 2.3 PSA Tests (`backend/tests/test_psa_engine.py`)
- [ ] Test parameter sampling distributions
- [ ] Test single PSA iteration
- [ ] Test PSA statistics calculation
- [ ] Test CEAC calculation
- [ ] Test parallelization (compare serial vs parallel results)

**Assignable to:** QA/Testing Developer
**Estimated time:** 1 day
**Dependencies:** 2.2 (PSA engine)

### 2.4 PSA Validation Against Excel
- [ ] Run PSA with same seed as Excel (if possible)
- [ ] Compare mean incremental cost ± 95% CI
- [ ] Compare mean incremental QALYs ± 95% CI
- [ ] Compare probability CE at £20k threshold
- [ ] Compare probability CE at £30k threshold
- [ ] Visualize and compare CE planes
- [ ] Document validation results

**Assignable to:** Research/Validation Specialist
**Estimated time:** 2 days
**Dependencies:** 2.2 (PSA engine), 2.3 (tests passing)

---

## Phase 3: Backend - FastAPI Endpoints (Week 3-4)

### 3.1 FastAPI Application (`backend/main.py`)
- [ ] Initialize FastAPI app
- [ ] Configure CORS middleware (allow localhost:3000)
- [ ] Setup global exception handling
- [ ] Create in-memory simulation store (dict)
- [ ] Add health check endpoint (`GET /health`)
- [ ] Add API documentation metadata
- [ ] Create startup/shutdown event handlers

**Assignable to:** Backend Developer
**Estimated time:** 4-6 hours
**Dependencies:** None (can start during Phase 1-2)

### 3.2 Parameter Endpoints (`backend/api/parameters.py`)
- [ ] Implement `GET /api/parameters/default` endpoint
  - Return default ModelParameters
- [ ] Implement `POST /api/parameters/validate` endpoint
  - Validate user-provided parameters
  - Return validation errors or success
- [ ] Add response models (Pydantic)
- [ ] Add docstrings and examples

**Assignable to:** Backend Developer
**Estimated time:** 4 hours
**Dependencies:** 1.2 (parameters model)

### 3.3 Simulation Endpoints (`backend/api/simulation.py`)
- [ ] Implement `POST /api/simulate/deterministic` endpoint
  - Accept ModelParameters
  - Run base and treatment simulations
  - Calculate incrementals
  - Return detailed results
- [ ] Implement `POST /api/simulate/psa` endpoint
  - Accept ModelParameters and PSAConfig
  - Generate simulation_id
  - Start background task
  - Return simulation_id and status
- [ ] Implement `GET /api/simulate/psa/{id}/status` endpoint
  - Return status, progress, estimated time
- [ ] Implement `GET /api/simulate/psa/{id}/results` endpoint
  - Return complete PSA results
- [ ] Implement background task handler for PSA
- [ ] Add response models and error handling

**Assignable to:** Backend Developer
**Estimated time:** 2-3 days
**Dependencies:** 1.4 (markov engine), 2.2 (PSA engine), 3.1 (FastAPI app)

### 3.4 Export Utilities (`backend/utils/excel_export.py`)
- [ ] Implement `create_summary_sheet(workbook, results)`
- [ ] Implement `create_parameters_sheet(workbook, params)`
- [ ] Implement `create_markov_trace_sheet(workbook, trace)`
- [ ] Implement `create_psa_results_sheet(workbook, psa_results)`
- [ ] Implement `generate_excel_workbook(results)` main function
- [ ] Add formatting (bold headers, number formats)

**Assignable to:** Backend Developer
**Estimated time:** 1 day
**Dependencies:** None (can work in parallel)

### 3.5 Export Endpoints (`backend/api/export.py`)
- [ ] Implement `POST /api/export/excel` endpoint
  - Accept simulation_id
  - Generate Excel file
  - Return FileResponse
- [ ] Add filename generation with timestamp
- [ ] Add error handling for missing simulations

**Assignable to:** Backend Developer
**Estimated time:** 4 hours
**Dependencies:** 3.4 (excel export utils)

### 3.6 API Tests (`backend/tests/test_api.py`)
- [ ] Test parameter endpoints
- [ ] Test deterministic simulation endpoint
- [ ] Test PSA endpoint flow (create → status → results)
- [ ] Test export endpoint
- [ ] Test error handling (invalid parameters, missing simulations)
- [ ] Test CORS headers

**Assignable to:** QA/Testing Developer
**Estimated time:** 1-2 days
**Dependencies:** 3.2, 3.3, 3.5 (all endpoints)

---

## Phase 4: Frontend - React Application (Week 4-6)

### 4.1 React Setup
- [ ] Run `npx create-react-app frontend`
- [ ] Install dependencies (Redux, React-Bootstrap, Chart.js, Plotly, axios, yup)
- [ ] Configure Redux store
- [ ] Setup React Router
- [ ] Create basic layout structure
- [ ] Add global styles

**Assignable to:** Frontend Developer
**Estimated time:** 1 day
**Dependencies:** None (can start during Phase 3)

### 4.2 API Client (`frontend/src/services/api.js`)
- [ ] Create axios instance with base URL
- [ ] Implement `getDefaultParameters()`
- [ ] Implement `validateParameters(params)`
- [ ] Implement `runDeterministicSimulation(params)`
- [ ] Implement `runPSASimulation(params, config)`
- [ ] Implement `getPSAStatus(simulationId)`
- [ ] Implement `getPSAResults(simulationId)`
- [ ] Implement `exportToExcel(simulationId)`
- [ ] Add error handling and response transformation

**Assignable to:** Frontend Developer
**Estimated time:** 1 day
**Dependencies:** 4.1 (React setup)

### 4.3 Redux Store
- [ ] Create `parametersSlice.js` (manage parameter state)
- [ ] Create `resultsSlice.js` (manage simulation results)
- [ ] Create `uiSlice.js` (manage UI state, active tabs, errors)
- [ ] Configure Redux store with slices
- [ ] Add Redux DevTools integration

**Assignable to:** Frontend Developer
**Estimated time:** 1 day
**Dependencies:** 4.1 (React setup)

### 4.4 Parameter Forms (`frontend/src/components/ParameterForm/`)
- [ ] Create `ParameterForm.jsx` (main container with tabs)
- [ ] Create `BaselineHazardsForm.jsx`
  - Input fields for stroke, MI, death hazards
  - Tooltips with literature references
- [ ] Create `TreatmentEffectsForm.jsx`
  - Input fields for stroke_hr, mi_hr
  - Show valid ranges
- [ ] Create `CostsForm.jsx`
  - Input fields for all cost parameters
  - Currency formatting
- [ ] Create `UtilitiesForm.jsx`
  - Input fields for utilities and disutilities
  - 0-1 range validation
- [ ] Implement real-time validation with yup
- [ ] Add "Reset to Defaults" button
- [ ] Add "Run Base Case" and "Run PSA" buttons
- [ ] Style with React-Bootstrap

**Assignable to:** Frontend Developer
**Estimated time:** 3-4 days
**Dependencies:** 4.2 (API client), 4.3 (Redux store)

### 4.5 Simulation Control
- [ ] Create `SimulationControls.jsx` component
- [ ] Implement PSA polling mechanism (2-second intervals)
- [ ] Create `ProgressBar.jsx` component (0-100%)
- [ ] Create `LoadingSpinner.jsx` component
- [ ] Add error toast notifications
- [ ] Handle simulation submission

**Assignable to:** Frontend Developer
**Estimated time:** 1 day
**Dependencies:** 4.2 (API client), 4.4 (parameter forms)

### 4.6 Results Components (`frontend/src/components/Results/`)
- [ ] Create `ResultsTabs.jsx` (tab container)
- [ ] Create `DeterministicResults.jsx`
  - Summary table (QALYs, costs, life years, events)
  - ICER calculation with NICE threshold comparison
  - NMB badges at £20k, £30k
- [ ] Create `MarkovTrace.jsx`
  - Stacked area chart with Chart.js
  - 8 states over 11 cycles
  - Color-coded legend
- [ ] Create `CEPlane.jsx`
  - Interactive scatter plot with Plotly
  - 10,000 PSA iteration points
  - WTP threshold lines (£20k, £30k)
  - Quadrant labels
- [ ] Create `CEAC.jsx`
  - Line chart with Chart.js
  - Probability CE vs WTP threshold
  - Annotations at £20k, £30k
- [ ] Create `PSASummary.jsx`
  - Summary statistics (mean ± 95% CI)
  - Probability CE at NICE thresholds
- [ ] Add export buttons for each chart (PNG)

**Assignable to:** 2 Frontend Developers (can split components)
**Estimated time:** 5-6 days
**Dependencies:** 4.2 (API client), 4.3 (Redux store)

### 4.7 Pages (`frontend/src/pages/`)
- [ ] Create `HomePage.jsx`
  - Project overview
  - Model description
  - Quick start guide
- [ ] Create `SimulationPage.jsx`
  - Integrate ParameterForm (left sidebar)
  - Integrate SimulationControls (right panel)
  - Layout with React-Bootstrap Grid
- [ ] Create `ResultsPage.jsx`
  - Integrate ResultsTabs
  - Handle empty state (no results yet)
- [ ] Create `DocumentationPage.jsx`
  - Methodology summary
  - Model equations
  - References

**Assignable to:** Frontend Developer
**Estimated time:** 2 days
**Dependencies:** 4.4 (forms), 4.5 (controls), 4.6 (results)

### 4.8 Styling and Polish
- [ ] Implement responsive design (mobile, tablet, desktop)
- [ ] Create consistent color scheme (professional/academic)
- [ ] Add loading states and transitions
- [ ] Implement accessibility (ARIA labels, keyboard navigation)
- [ ] Add tooltips for all parameters
- [ ] Polish animations and interactions
- [ ] Cross-browser testing

**Assignable to:** Frontend Developer / UI Designer
**Estimated time:** 2-3 days
**Dependencies:** 4.7 (all pages complete)

### 4.9 Frontend Tests (`frontend/src/`)
- [ ] Component tests for parameter forms (Jest + React Testing Library)
- [ ] Component tests for results visualizations
- [ ] Integration tests for simulation workflow
- [ ] API client tests (mock API responses)
- [ ] Redux store tests

**Assignable to:** QA/Testing Developer
**Estimated time:** 2-3 days
**Dependencies:** 4.4-4.7 (all components)

---

## Phase 5: Integration and Testing (Week 7)

### 5.1 End-to-End Testing
- [ ] Test full workflow: parameter input → base case → results
- [ ] Test full workflow: parameter input → PSA → results → export
- [ ] Test PSA cancellation
- [ ] Test error scenarios (invalid parameters, network errors)
- [ ] Test with various parameter values
- [ ] Performance testing (PSA completion time)
- [ ] Cross-browser testing (Chrome, Firefox, Safari, Edge)
- [ ] Mobile responsiveness testing

**Assignable to:** QA/Testing Team
**Estimated time:** 2-3 days
**Dependencies:** Phase 3 and 4 complete

### 5.2 Final Validation
- [ ] Run multiple scenarios and compare with Excel
- [ ] Validate edge cases (HR=1.0, HR=0.0, different time horizons)
- [ ] Statistical validation of PSA distributions
- [ ] Verify all NICE guidelines compliance
- [ ] Check CHEERS 2022 reporting standards
- [ ] Document any discrepancies

**Assignable to:** Research/Validation Specialist
**Estimated time:** 2 days
**Dependencies:** 5.1 (E2E tests passing)

### 5.3 Documentation
- [ ] Write USER_GUIDE.md with screenshots
  - How to input parameters
  - How to run simulations
  - How to interpret results
  - How to export data
- [ ] Write API_DOCUMENTATION.md (or use FastAPI auto-docs)
  - Endpoint descriptions
  - Request/response examples
  - Error codes
- [ ] Write DEVELOPER_GUIDE.md
  - Code structure explanation
  - How to add new features
  - Testing guidelines
- [ ] Write DEPLOYMENT.md
  - Local development setup
  - Production deployment (Docker, cloud)
  - Environment variables
- [ ] Update main README.md with project overview

**Assignable to:** Technical Writer / Developer
**Estimated time:** 2-3 days
**Dependencies:** Phase 3 and 4 complete

### 5.4 Deployment Preparation
- [ ] Create Dockerfile for backend
- [ ] Create Dockerfile for frontend
- [ ] Create docker-compose.yml
- [ ] Add environment variable configuration
- [ ] Test Docker deployment locally
- [ ] Write deployment instructions for cloud platforms (AWS, GCP, Azure)

**Assignable to:** DevOps / Backend Developer
**Estimated time:** 1-2 days
**Dependencies:** Phase 3 and 4 complete

---

## Additional Tasks (Ongoing)

### Code Quality
- [ ] Add type hints to all Python functions
- [ ] Add comprehensive docstrings
- [ ] Run linting (pylint, flake8)
- [ ] Run code formatting (black, prettier)
- [ ] Add pre-commit hooks

**Assignable to:** All developers
**Estimated time:** Ongoing

### Bug Fixes
- [ ] Track bugs in GitHub Issues
- [ ] Prioritize and assign bugs
- [ ] Fix and test
- [ ] Document fixes

**Assignable to:** All developers
**Estimated time:** Ongoing

---

## Task Assignment Legend

- **Backend Developer:** Python, FastAPI, NumPy, scientific computing
- **Senior Backend Developer:** Complex algorithms, architecture decisions
- **Frontend Developer:** React, Redux, JavaScript, CSS
- **Data Science/Stats Developer:** Statistical distributions, PSA methodology
- **QA/Testing Developer:** Writing tests, test automation, validation
- **Research/Validation Specialist:** Domain expertise, Excel comparison, NICE guidelines
- **Technical Writer:** Documentation, guides, tutorials
- **DevOps:** Deployment, containerization, CI/CD
- **UI Designer:** Visual design, UX, accessibility

---

## Getting Help

- See `CONTRIBUTING.md` for contribution guidelines
- See `DEVELOPER_GUIDE.md` for code structure (once created)
- Refer to the implementation plan at `.claude/plans/distributed-discovering-balloon.md`
- Check existing issues and PRs on GitHub

---

## Progress Tracking

Update this file as tasks are completed. Mark completed tasks with `[x]` and add completion date and contributor name in comments.

Example:
```markdown
- [x] Create backend directory structure <!-- Completed 2025-12-03 by @contributor1 -->
```
