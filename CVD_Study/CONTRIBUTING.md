# Contributing to CVD Markov Model Web Application

Thank you for your interest in contributing to this project! This guide will help you get started.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Task Assignment](#task-assignment)
8. [Communication](#communication)

---

## Project Overview

This project converts an Excel-based Markov cohort model for cardiovascular disease (CVD) cost-effectiveness analysis into a modern web application. The goal is to enable researchers to adjust parameters, run simulations, and visualize results without requiring Excel expertise.

**Tech Stack:**
- **Backend:** Python 3.11+, FastAPI, NumPy, pandas
- **Frontend:** React 18+, Redux Toolkit, Chart.js, Plotly
- **Testing:** pytest (backend), Jest + React Testing Library (frontend)

**Key Objectives:**
- Maintain scientific accuracy (validate against Excel model)
- Follow NICE guidelines for health economic evaluation
- Provide an intuitive user interface for academic researchers
- Support probabilistic sensitivity analysis (10,000 Monte Carlo simulations)

---

## Getting Started

### Prerequisites

**Backend:**
- Python 3.11 or higher
- pip and virtualenv

**Frontend:**
- Node.js 18+ and npm

**General:**
- Git
- Code editor (VS Code recommended)

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kwaai-AI-Lab/periodontal.git
   cd periodontal
   ```

2. **Backend setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend setup (when available):**
   ```bash
   cd frontend
   npm install
   ```

4. **Verify installation:**
   ```bash
   # Backend
   cd backend
   pytest

   # Frontend
   cd frontend
   npm test
   ```

---

## Development Workflow

### 1. Pick a Task

- Review `TODO.md` for available tasks
- Look for tasks marked with `[ ]` (incomplete)
- Choose a task that matches your skills (see Task Assignment Legend in TODO.md)
- Comment on the task or create a GitHub Issue to claim it

### 2. Create a Branch

Use descriptive branch names following this convention:
```
<type>/<short-description>

Types:
- feature/  : New feature
- fix/      : Bug fix
- docs/     : Documentation
- test/     : Adding tests
- refactor/ : Code refactoring
```

Examples:
```bash
git checkout -b feature/markov-engine
git checkout -b fix/parameter-validation
git checkout -b docs/api-documentation
```

### 3. Write Code

- Follow the code standards (see below)
- Write tests for new functionality
- Update documentation as needed
- Keep commits small and focused

### 4. Commit Changes

Write clear, descriptive commit messages:
```
<type>: <subject>

<body (optional)>

Types:
- feat:     New feature
- fix:      Bug fix
- docs:     Documentation changes
- test:     Adding or updating tests
- refactor: Code refactoring
- style:    Code style changes (formatting, etc.)
- chore:    Maintenance tasks
```

Examples:
```bash
git commit -m "feat: implement MarkovCVDModel class with transition matrix"
git commit -m "fix: handle division by zero in ICER calculation"
git commit -m "docs: add API endpoint documentation"
git commit -m "test: add unit tests for parameter validation"
```

### 5. Push and Create Pull Request

```bash
git push origin <your-branch-name>
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what was changed and why
- Reference to related TODO items or issues
- Screenshots (for UI changes)
- Test results

---

## Code Standards

### Python (Backend)

**Style Guide:**
- Follow PEP 8
- Use type hints for function parameters and return values
- Maximum line length: 100 characters

**Naming Conventions:**
- Variables and functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

**Example:**
```python
from typing import Optional, Tuple
import numpy as np

class MarkovCVDModel:
    """8-state Markov cohort model for CVD cost-effectiveness analysis.

    This model simulates a cohort of individuals over 10 years to estimate
    the cost-effectiveness of non-surgical periodontal therapy for CVD prevention.

    Attributes:
        params: ModelParameters containing all model inputs
        n_states: Number of health states (8)
        n_cycles: Number of simulation cycles (10)
    """

    def __init__(self, params: ModelParameters) -> None:
        self.params = params
        self.n_states = 8
        self.n_cycles = 10

    def hazard_to_probability(self, hazard: float, time: float = 1.0) -> float:
        """Convert hazard rate to cycle probability.

        Args:
            hazard: Annual hazard rate
            time: Time period in years (default: 1.0)

        Returns:
            Probability of event occurring in time period

        Formula:
            p = 1 - exp(-hazard × time)
        """
        if hazard <= 0.0:
            return 0.0
        return 1.0 - np.exp(-hazard * time)
```

**Docstrings:**
- Use Google-style docstrings
- Include description, arguments, returns, and raises sections
- Add examples for complex functions

**Imports:**
- Group imports: standard library, third-party, local
- Sort alphabetically within groups
- Use absolute imports

### JavaScript/React (Frontend)

**Style Guide:**
- Use ES6+ syntax
- Use functional components with hooks
- Use Prettier for formatting
- Maximum line length: 100 characters

**Naming Conventions:**
- Components: `PascalCase` (e.g., `ParameterForm.jsx`)
- Variables and functions: `camelCase`
- Constants: `UPPER_SNAKE_CASE`
- CSS classes: `kebab-case`

**Example:**
```javascript
import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Form, Button, Alert } from 'react-bootstrap';
import { setParameters } from '../../store/parametersSlice';

/**
 * Parameter form component for baseline hazards input.
 *
 * Allows users to specify stroke, MI, and death hazards for the base case population.
 * Values are validated in real-time against literature-based ranges.
 *
 * @component
 */
const BaselineHazardsForm = ({ values, onChange, errors }) => {
  const [touched, setTouched] = useState({});

  const handleBlur = (field) => {
    setTouched({ ...touched, [field]: true });
  };

  return (
    <div className="baseline-hazards-form">
      <Form.Group className="mb-3">
        <Form.Label>Baseline Stroke Hazard (annual)</Form.Label>
        <Form.Control
          type="number"
          step="0.0001"
          value={values.stroke}
          onChange={(e) => onChange('stroke', e.target.value)}
          onBlur={() => handleBlur('stroke')}
          isInvalid={touched.stroke && errors.stroke}
        />
        <Form.Control.Feedback type="invalid">
          {errors.stroke}
        </Form.Control.Feedback>
        <Form.Text className="text-muted">
          Literature range: 0.015 - 0.035 for 65-year-old with severe PD
        </Form.Text>
      </Form.Group>
      {/* More form fields... */}
    </div>
  );
};

export default BaselineHazardsForm;
```

**Component Structure:**
- Keep components small and focused (single responsibility)
- Extract reusable logic into custom hooks
- Use PropTypes or TypeScript for type checking
- Avoid inline styles (use CSS modules or styled-components)

---

## Testing Guidelines

### Backend Testing

**Unit Tests:**
- Test individual functions and methods
- Mock external dependencies
- Use pytest fixtures for setup/teardown
- Aim for >80% code coverage

**Example:**
```python
import pytest
from backend.models.markov_engine import MarkovCVDModel
from backend.models.parameters import ModelParameters

@pytest.fixture
def default_params():
    """Fixture providing default model parameters."""
    return ModelParameters(
        baseline_hazards=BaselineHazards(stroke=0.025, mi=0.030, death=0.0135),
        # ... other parameters
    )

def test_hazard_to_probability(default_params):
    """Test hazard to probability conversion."""
    model = MarkovCVDModel(default_params)

    # Test with known values
    prob = model.hazard_to_probability(0.01, time=1.0)
    assert abs(prob - 0.00995) < 1e-5  # Expected: 1 - exp(-0.01)

    # Test edge case: zero hazard
    prob_zero = model.hazard_to_probability(0.0)
    assert prob_zero == 0.0

def test_transition_matrix_properties(default_params):
    """Test that transition matrix rows sum to 1.0."""
    model = MarkovCVDModel(default_params)
    T = model.build_transition_matrix(cycle=0, treatment=False)

    # Check row sums
    row_sums = T.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)

    # Check death state is absorbing
    assert T[7, 7] == 1.0
    assert np.all(T[7, :7] == 0.0)
```

**Running Tests:**
```bash
cd backend
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest tests/test_markov_engine.py  # Specific file
pytest --cov=backend            # With coverage report
```

### Frontend Testing

**Component Tests:**
- Test component rendering
- Test user interactions
- Mock API calls
- Test Redux actions and reducers

**Example:**
```javascript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import BaselineHazardsForm from './BaselineHazardsForm';

const mockStore = configureStore([]);

describe('BaselineHazardsForm', () => {
  let store;

  beforeEach(() => {
    store = mockStore({
      parameters: {
        baseline_hazards: { stroke: 0.025, mi: 0.030, death: 0.0135 }
      }
    });
  });

  it('renders all input fields', () => {
    render(
      <Provider store={store}>
        <BaselineHazardsForm />
      </Provider>
    );

    expect(screen.getByLabelText(/Baseline Stroke Hazard/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Baseline MI Hazard/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Background Death Hazard/i)).toBeInTheDocument();
  });

  it('validates input values', async () => {
    const { container } = render(
      <Provider store={store}>
        <BaselineHazardsForm />
      </Provider>
    );

    const strokeInput = screen.getByLabelText(/Baseline Stroke Hazard/i);

    // Enter invalid value
    fireEvent.change(strokeInput, { target: { value: '-0.01' } });
    fireEvent.blur(strokeInput);

    await waitFor(() => {
      expect(screen.getByText(/must be positive/i)).toBeInTheDocument();
    });
  });
});
```

**Running Tests:**
```bash
cd frontend
npm test                        # Run all tests
npm test -- --coverage          # With coverage
npm test -- BaselineHazardsForm # Specific file
```

---

## Pull Request Process

### Before Submitting

1. **Ensure tests pass:**
   ```bash
   # Backend
   pytest

   # Frontend
   npm test
   ```

2. **Run linting:**
   ```bash
   # Backend
   flake8 backend/

   # Frontend
   npm run lint
   ```

3. **Update TODO.md:**
   - Mark completed tasks with `[x]`
   - Add completion date and your name

4. **Update documentation:**
   - Add docstrings to new functions
   - Update relevant .md files
   - Add inline comments for complex logic

### PR Template

Use this template for your pull request description:

```markdown
## Description
Brief description of what this PR does.

## Related Issue/TODO
- Fixes #123
- Implements TODO item: Phase 1.2 - Parameter Models

## Changes Made
- Added BaselineHazards Pydantic model
- Added TreatmentEffects Pydantic model
- Added validation tests
- Updated TODO.md

## Testing
- [ ] Unit tests added and passing
- [ ] Integration tests passing (if applicable)
- [ ] Manual testing completed
- [ ] Validated against Excel model (if applicable)

## Screenshots (for UI changes)
[Add screenshots here]

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] TODO.md updated
```

### Review Process

1. At least one reviewer must approve the PR
2. All CI checks must pass
3. Address reviewer feedback promptly
4. Resolve merge conflicts if they arise
5. Squash commits if requested

---

## Task Assignment

### How to Claim a Task

1. **Check TODO.md** for available tasks
2. **Comment on GitHub Issue** or create a new issue for the task
3. **Assign yourself** (or ask to be assigned)
4. **Update TODO.md** by adding your name:
   ```markdown
   - [ ] Implement MarkovCVDModel class <!-- @yourname - In Progress -->
   ```

### Task Types by Role

**Backend Developers:**
- Python implementation (models, utilities, API endpoints)
- Backend testing
- Database/storage integration

**Frontend Developers:**
- React components
- Redux state management
- UI/UX implementation
- Frontend testing

**Data Science/Stats Developers:**
- PSA implementation
- Statistical distributions
- Validation against literature

**QA/Testing:**
- Test writing (unit, integration, E2E)
- Test automation
- Bug reporting and verification

**Research/Validation:**
- Excel model comparison
- NICE guidelines compliance
- Methodology documentation

**DevOps:**
- Deployment setup (Docker, CI/CD)
- Infrastructure configuration

**Technical Writers:**
- Documentation (user guides, API docs)
- Tutorials and examples

---

## Communication

### Channels

- **GitHub Issues:** Bug reports, feature requests, task discussions
- **Pull Requests:** Code reviews, implementation discussions
- **TODO.md comments:** Quick updates on task progress

### Best Practices

- **Be respectful and professional**
- **Ask questions** if anything is unclear
- **Provide context** when reporting issues or requesting features
- **Be responsive** to feedback on your PRs
- **Help others** when you can

### Reporting Bugs

Use this template for bug reports:

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Enter '...'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., macOS 14.0]
- Browser: [e.g., Chrome 120] (for frontend bugs)
- Python version: [e.g., 3.11.5] (for backend bugs)
- Node version: [e.g., 18.17.0] (for frontend bugs)

## Screenshots
[Add screenshots if applicable]

## Additional Context
Any other relevant information.
```

---

## Code of Conduct

### Our Standards

- **Respectful:** Treat all contributors with respect
- **Inclusive:** Welcome people of all backgrounds and experience levels
- **Collaborative:** Work together, share knowledge, help each other
- **Professional:** Maintain professional conduct in all interactions
- **Constructive:** Provide constructive feedback, focus on solutions

### Unacceptable Behavior

- Harassment, discrimination, or offensive language
- Personal attacks or trolling
- Spam or irrelevant content
- Sharing private information without consent

### Enforcement

Violations of the code of conduct should be reported to the project maintainers. All reports will be reviewed and investigated promptly.

---

## Questions?

If you have questions:

1. Check the `README.md` for general information
2. Review the implementation plan at `.claude/plans/distributed-discovering-balloon.md`
3. Check existing GitHub Issues
4. Create a new GitHub Issue with your question

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

## Thank You!

Thank you for contributing to this project! Your efforts help make cost-effectiveness research more accessible to the academic community.
