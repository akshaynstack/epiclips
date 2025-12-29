# Contributing to Epiclips

Thank you for your interest in contributing to Epiclips! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/epiclips.git
   cd epiclips
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Setup

### Backend (Python)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black isort mypy

# Run the server
uvicorn app.main:app --reload
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

## 📝 Code Style

### Python
- Use **Black** for formatting
- Use **isort** for import sorting
- Follow **PEP 8** guidelines

```bash
# Format code
black app/
isort app/

# Check types
mypy app/
```

### TypeScript/JavaScript
- Use **ESLint** and **Prettier**
- Follow existing patterns in the codebase

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 📤 Submitting Changes

1. **Commit your changes** with clear messages:
   ```bash
   git commit -m "feat: add new caption style"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub

### Commit Message Format

We use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

## 🐛 Reporting Issues

When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

## 💡 Feature Requests

Feature requests are welcome! Please:
- Check existing issues first
- Describe the use case
- Explain why it would be valuable

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
