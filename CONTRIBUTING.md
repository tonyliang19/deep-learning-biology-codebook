# Contributing to Deep Learning Biology Codebook

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find errors, typos, or have suggestions:

1. Check if the issue already exists
2. Open a new issue with:
   - Clear title and description
   - Steps to reproduce (if applicable)
   - Expected vs actual behavior
   - Relevant screenshots or code snippets

### Suggesting Enhancements

We welcome suggestions for:
- New topics to cover
- Additional examples
- Biology-specific applications
- Improved explanations

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Notebook Guidelines

When adding or modifying notebooks:

### Structure
- Start with a clear title and introduction
- Use markdown cells for explanations
- Include code cells with comments
- End with a summary and exercises

### Content
- **Theory**: Explain concepts with mathematical foundations
- **Code**: Working, well-commented implementations
- **Visualizations**: Plots and diagrams to aid understanding
- **Biology Applications**: Real-world examples when possible

### Style
- Use consistent formatting
- Keep code cells focused (one concept per cell)
- Add section headers for navigation
- Include TOC at the beginning

### Testing
Before submitting:
```bash
# Validate notebook JSON
python3 -m json.tool notebooks/your_notebook.ipynb > /dev/null

# Run the notebook
jupyter nbconvert --execute --to notebook notebooks/your_notebook.ipynb
```

## Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep lines under 100 characters when possible

## Documentation

- Update README.md if adding new features
- Add entries to _toc.yml for new notebooks
- Update requirements.txt for new dependencies

## Biology Applications

When adding biology examples:
- Cite relevant papers/datasets
- Explain biological context
- Use realistic data when possible
- Mention ethical considerations if applicable

## Questions?

Feel free to open an issue for:
- Clarification on guidelines
- Discussion of potential contributions
- General questions about the project

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for helping make this resource better! 🧬🤖
