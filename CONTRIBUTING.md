# Contributing to Kartezio

Thank you for your interest in contributing to Kartezio! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/kartezio.git
   cd kartezio
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```
   
   Or with uv (recommended):
   ```bash
   uv sync --extra dev
   ```

3. **Verify installation**:
   ```bash
   python -c "import kartezio; print('Setup successful!')"
   ```

## 🧪 Testing

Run the test suite before submitting contributions:

```bash
# Quick tests (recommended for development)
uv run python tests/test_runner.py --quick

# Security tests
uv run python scripts/dev.py security

# Full test suite with our custom runner
uv run python tests/test_runner.py

# Individual test suites
uv run python -m unittest tests.test_core_components
uv run python -m unittest tests.test_security

# Using pytest (if preferred)
uv run python -m pytest tests/
```

## 🎨 Code Quality

We maintain high code quality standards:

```bash
# Format code
uv run ruff format src/ tests/

# Lint and auto-fix
uv run ruff check --fix src/ tests/

# Use development scripts for comprehensive checks
uv run python scripts/dev.py format
uv run python scripts/dev.py lint
uv run python scripts/dev.py typecheck
```

## 📝 Contribution Types

### 🐛 Bug Reports

When reporting bugs, please include:
- Python version and operating system
- Kartezio version (`python -c "import kartezio; print(kartezio.__version__)"`)
- Minimal code example reproducing the issue
- Expected vs. actual behavior
- Error messages and stack traces

### ✨ Feature Requests

For feature requests, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)
- Willingness to implement (if any)

### 🔧 Code Contributions

1. **Fork the repository** and create a feature branch
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## 📚 Component Development

### Adding New Primitives

```python
from kartezio.core.components import Primitive, register
from kartezio.types import Matrix

@register(Primitive)
class MyCustomFilter(Primitive):
    def __init__(self):
        super().__init__([Matrix], Matrix, n_parameters=1)
    
    def call(self, x, args):
        # Your implementation here
        return processed_image
```

### Adding New Fitness Functions  

```python
from kartezio.core.components import Fitness, register
import numpy as np

@register(Fitness)
class MyCustomMetric(Fitness):
    def evaluate(self, y_true, y_pred):
        # Your evaluation logic here
        return score_array
```

## 📖 Documentation

- Update docstrings for any new public APIs
- Add examples for new features
- Update README.md if adding major functionality
- Follow Google-style docstring format

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the same license as the project (Proprietary - Non-Commercial/Academic use only).

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment
- Follow scientific and academic standards of conduct

## 📞 Questions?

- 💬 **Discord**: [Join our community](https://discord.gg/KnJ4XWdQMK)
- 📧 **Email**: kevin.cortacero@protonmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/kartezio/issues)

---

**Thank you for contributing to Kartezio and advancing explainable computer vision!** 🚀