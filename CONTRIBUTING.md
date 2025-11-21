# Contributing to Deep Learning for Biology Codebook

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Getting Started](#getting-started)
4. [Contribution Workflow](#contribution-workflow)
5. [Notebook Guidelines](#notebook-guidelines)
6. [Code Style](#code-style)
7. [Documentation](#documentation)
8. [Review Process](#review-process)

## Code of Conduct

This project follows a simple code of conduct:

- **Be respectful**: Treat all contributors with respect and kindness
- **Be constructive**: Provide helpful, actionable feedback
- **Be inclusive**: Welcome contributors of all backgrounds and skill levels
- **Be patient**: Remember that everyone is here to learn
- **Be collaborative**: Work together to improve the project

## How Can I Contribute?

### Reporting Bugs

If you find a bug in a notebook or documentation:

1. **Check existing issues** to see if it's already reported
2. **Create a new issue** with:
   - Clear, descriptive title
   - Detailed description of the bug
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (OS, Python version, package versions)
   - Screenshots if applicable

### Suggesting Enhancements

Have an idea for improvement?

1. **Check existing issues** for similar suggestions
2. **Open a new issue** tagged with "enhancement"
3. **Describe**:
   - The current limitation
   - Your proposed solution
   - Why this would be useful
   - Alternative approaches considered

### Contributing Notebooks

We welcome new notebooks on topics related to deep learning in biology!

**Good topics include:**
- Novel architectures for biological data
- Applications to emerging datasets
- Comparisons of different approaches
- Tutorials on specific tools or techniques
- Real-world case studies

**Before starting:**
1. Open an issue to discuss your proposed notebook
2. Check if similar content already exists
3. Get feedback on scope and approach

### Improving Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add missing information
- Improve examples
- Update outdated content
- Translate to other languages

### Fixing Bugs

Browse the [issue tracker](https://github.com/tonyliang19/deep-learning-biology-codebook/issues) for bugs labeled "good first issue" or "help wanted".

## Getting Started

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/deep-learning-biology-codebook.git
   cd deep-learning-biology-codebook
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/tonyliang19/deep-learning-biology-codebook.git
   ```

### Set Up Development Environment

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Install pre-commit hooks** (if configured):
   ```bash
   pre-commit install
   ```

## Contribution Workflow

### 1. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features or notebooks
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

### 2. Make Your Changes

- Write clear, well-commented code
- Follow existing style and structure
- Test your changes thoroughly
- Update documentation as needed

### 3. Commit Your Changes

```bash
# Add files
git add path/to/changed/files

# Commit with descriptive message
git commit -m "Add: Brief description of changes"
```

**Commit message guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues: "Fix #123: Description"
- Use prefixes:
  - `Add:` - New features or content
  - `Fix:` - Bug fixes
  - `Update:` - Updates to existing content
  - `Refactor:` - Code refactoring
  - `Docs:` - Documentation changes
  - `Test:` - Test additions or changes

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then on GitHub:
1. Navigate to your fork
2. Click "New Pull Request"
3. Fill out the PR template
4. Link related issues
5. Request review

## Notebook Guidelines

### Structure

All notebooks should follow this structure:

```markdown
# Title

## Overview
Brief description of the notebook's purpose and goals

## Learning Objectives
- Objective 1
- Objective 2
- Objective 3

## Prerequisites
- Required knowledge
- Required packages
- Required data

## Background
Biological context and problem description

## Implementation

### 1. Import Libraries
[Code cell with imports]

### 2. Load Data
[Code cells for data loading]

### 3. Data Exploration
[Code cells for EDA]

### 4. Preprocessing
[Code cells for data preparation]

### 5. Model Definition
[Code cells defining the model]

### 6. Training
[Code cells for training]

### 7. Evaluation
[Code cells for evaluation]

### 8. Interpretation
[Analysis and biological insights]

## Exercises (Optional)
Challenges for readers to try

## References
Citations for methods, data, and papers

## Next Steps
Suggestions for further exploration
```

### Best Practices

**Code Quality:**
- Use clear variable names
- Add docstrings to functions
- Include comments for complex logic
- Keep cells focused and not too long
- Use type hints where appropriate

**Documentation:**
- Explain the "why" not just the "what"
- Include biological context
- Add visualizations to illustrate concepts
- Link to relevant resources

**Data:**
- Use small sample datasets when possible
- Provide scripts to download full datasets
- Document data sources and licenses
- Include data validation steps

**Reproducibility:**
- Set random seeds
- Document package versions
- Include environment specifications
- Make notebooks self-contained

### Testing Notebooks

Before submitting:

1. **Restart kernel and run all cells**
   ```python
   # In Jupyter: Kernel > Restart & Run All
   ```

2. **Check for errors**
   - No cell should raise exceptions
   - All outputs should be generated
   - Warnings should be addressed

3. **Verify outputs**
   - Plots render correctly
   - Results are reasonable
   - Metrics are calculated properly

4. **Test with different environments**
   - Try on CPU and GPU (if applicable)
   - Test with minimum required package versions

5. **Clear outputs before committing** (optional)
   ```bash
   jupyter nbconvert --clear-output --inplace notebook.ipynb
   ```

## Code Style

### Python

Follow [PEP 8](https://pep8.org/) style guide:

```python
# Good
def calculate_accuracy(predictions, targets):
    """Calculate classification accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Accuracy as float between 0 and 1
    """
    correct = (predictions == targets).sum()
    total = len(targets)
    return correct / total


# Avoid
def calc_acc(p,t):
    return (p==t).sum()/len(t)
```

### Markdown

- Use proper heading hierarchy
- Include blank lines between sections
- Use code fences with language specification
- Keep lines under 100 characters when possible

### Notebooks

- One logical operation per cell
- Clear cell outputs before committing (optional)
- Use markdown cells generously for explanations
- Keep notebook file size reasonable (<10 MB if possible)

## Documentation

### Updating README

When adding new content:
- Update the "Topics Covered" section
- Add to the "Repository Structure" if needed
- Update installation instructions for new dependencies

### Updating RESOURCES.md

When adding references:
- Place in appropriate category
- Include full citation information
- Add links to freely accessible versions
- Briefly describe why it's useful

### Creating Examples

Good examples should:
- Be self-contained
- Use realistic data
- Show best practices
- Include error handling
- Have clear explanations

## Review Process

### For Contributors

After submitting a PR:

1. **Respond to feedback** promptly and constructively
2. **Make requested changes** in new commits
3. **Ask questions** if feedback is unclear
4. **Be patient** - reviews take time
5. **Thank reviewers** for their time

### For Reviewers

When reviewing PRs:

1. **Be constructive** and specific
2. **Explain the "why"** behind suggestions
3. **Praise good work** along with critiques
4. **Focus on substance** over style (unless style matters)
5. **Test the changes** if possible
6. **Approve promptly** when changes meet standards

### Review Criteria

PRs are evaluated on:

- **Correctness**: Code works as intended
- **Clarity**: Easy to understand and follow
- **Completeness**: Includes necessary documentation
- **Consistency**: Matches project style and structure
- **Quality**: Well-tested and robust
- **Value**: Provides benefit to users

## Questions?

- **General questions**: Open a [Discussion](https://github.com/tonyliang19/deep-learning-biology-codebook/discussions)
- **Bug reports**: Open an [Issue](https://github.com/tonyliang19/deep-learning-biology-codebook/issues)
- **Security concerns**: Email the maintainer directly

## Recognition

Contributors will be:
- Listed in the README (if desired)
- Credited in release notes
- Thanked in the community

Thank you for contributing! 🎉

---

**Questions about these guidelines?** Open an issue and we'll clarify.
