# Contributing to Mix-and-Match Pruning

We welcome contributions to the Mix-and-Match Pruning framework! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, PyTorch version, OS, etc.)
- Any relevant code snippets or error messages

### Suggesting Enhancements

We welcome suggestions for new features or improvements:
- Create an issue with the tag "enhancement"
- Clearly describe the proposed feature and its benefits
- Provide examples of how it would be used
- Explain why this enhancement would be useful to most users

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Follow existing code style and conventions
   - Add comments to complex logic
   - Update documentation as needed
3. **Test your changes**:
   - Ensure existing tests pass
   - Add tests for new functionality
   - Verify on at least one architecture (VGG, ResNet, LeViT, or Swin)
4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference related issues in commits
5. **Push to your fork** and submit a pull request

### Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex algorithms or non-obvious logic

### Adding New Architectures

To add support for a new architecture:

1. Create three new files following the naming pattern:
   - `{architecture}_multi_strategy.py` - Main pruning script
   - `{architecture}_sensitivity_simple.py` - Sensitivity analysis
   - `{architecture}_layer_classifier.py` - Layer classification

2. Implement the required functions:
   - `load_model()` - Model loading
   - `evaluate_model()` - Accuracy evaluation
   - `get_layer_info()` - Layer extraction and ordering
   - `apply_pruning_mask()` - Pruning execution
   - `fine_tune_model()` - Fine-tuning with masks

3. Update the README.md with:
   - Architecture description
   - Dataset information
   - Baseline results
   - Recommended sensitivity metric

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex code
- Update requirements.txt if adding dependencies
- Include examples for new features

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/mix-and-match-pruning.git
cd mix-and-match-pruning

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Testing

Before submitting a pull request:

```bash
# Run existing tests (if available)
pytest tests/

# Check code style
flake8 .

# Format code
black .
```

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the tag "question"
- Reach out to the maintainers

Thank you for contributing to Mix-and-Match Pruning!
