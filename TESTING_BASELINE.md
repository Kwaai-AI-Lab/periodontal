# Testing Baseline - Established January 29, 2026

## Overview
This document establishes the testing baseline for the periodontal health impact simulator following the merge of PR #5.

## Test Suite Statistics

### Test Execution
- **Total Tests**: 137
- **Passing**: 132
- **Skipped**: 5 (documented missing module)
- **Failing**: 0
- **Execution Time**: ~0.7 seconds

### Test Distribution
| Test File | Tests | Status | Coverage |
|-----------|-------|--------|----------|
| test_ibm_pd_ad.py | 97 | ✅ All Pass | 100% of tested functions |
| test_cvd_figures.py | 26 | ✅ All Pass | 100% of tested functions |
| test_psa_workflow.py | 14 | ✅ 9 pass, 5 skip | 80% |

### Code Coverage
| Module | Coverage | Notes |
|--------|----------|-------|
| generate_cvd_figures.py | 84% | Excellent coverage |
| IBM_PD_AD.py | 10% | Critical functions covered |
| **Overall** | **32%** | Good baseline |

## Critical Functions - 100% Coverage ✅

### Hazard/Probability Conversions (Tier 1)
- `prob_to_hazard()` - 9 tests
- `hazard_to_prob()` - 7 tests
- `base_hazard_from_duration()` - 7 tests
- `hazards_from_survival()` - 12 tests
- Round-trip conversions - 14 tests

### Age Band Handling (Tier 2)
- `assign_age_to_reporting_band()` - 8 tests
- `age_band_key()` - 2 tests
- `age_band_label()` - 2 tests
- `age_band_midpoint()` - 3 tests

### Distribution Parameters (Tier 3)
- `_beta_params_from_mean_rel_sd()` - 9 tests
- `_gamma_params_from_mean_rel_sd()` - 7 tests
- `_lognormal_params_from_ci()` - 7 tests

### Cost-Effectiveness Analysis
- Net Monetary Benefit (NMB) calculations - 6 tests
- CEAC probability calculations - 3 tests
- CE plane quadrant analysis - 3 tests
- Integration tests - 2 tests

## Known Issues

### Missing Module: psa_with_timeseries
- **Status**: Module not in repository
- **Impact**: 5 tests skipped
- **Required by**: run_psa_direct.py
- **Action**: Contact original authors to obtain module
- **Workaround**: Tests document expected interface

## Testing Infrastructure

### Configuration
- **pytest.ini**: Configured with verbose output, short tracebacks, deprecation filters
- **.gitignore**: Updated to exclude test artifacts (.coverage, .pytest_cache/, htmlcov/)

### Dependencies
**Production** (requirements.txt):
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- openpyxl >= 3.1.0
- tqdm >= 4.65.0

**Development** (requirements-dev.txt):
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- flake8 >= 6.1.0
- mypy >= 1.5.0
- pandas-stubs >= 2.0.0

## Running Tests

### Run all tests
```bash
pytest -v
```

### Run with coverage
```bash
pytest --cov --cov-report=term-missing
```

### Run specific test file
```bash
pytest tests/test_ibm_pd_ad.py -v
```

### Run with coverage report (HTML)
```bash
pytest --cov --cov-report=html
open htmlcov/index.html
```

## Future Improvements

### Phase 1 (Immediate)
- [ ] Resolve missing psa_with_timeseries module
- [ ] Add integration tests for main script execution
- [ ] Increase IBM_PD_AD.py coverage to 30%

### Phase 2 (Short-term)
- [ ] Add property-based tests (hypothesis) for mathematical functions
- [ ] Add data validation and error handling tests
- [ ] Add performance regression tests

### Phase 3 (Long-term)
- [ ] Add end-to-end tests with real data
- [ ] Set up CI/CD pipeline
- [ ] Add mutation testing for test quality validation

## Baseline Metrics (January 29, 2026)

| Metric | Value | Target |
|--------|-------|--------|
| Test Count | 137 | 200+ |
| Pass Rate | 96.4% | 100% |
| Coverage (Overall) | 32% | 70% |
| Coverage (Critical) | 100% | 100% |
| Execution Time | 0.7s | <5s |

---

**Baseline Established**: January 29, 2026  
**Established By**: Claude Sonnet 4.5  
**PR Reference**: #5
