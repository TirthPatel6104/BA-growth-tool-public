# Contributing Guidelines

Thank you for your interest in contributing to the **Basal Area Growth Tool**!

## Development Workflow

1. **Fork & Clone**: Fork the repository and clone it to your local machine.
2. **Branching**: Create a feature branch off `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Local Testing**:
   - Backend: Run `pytest tests/backend/`
   - Frontend: Run `npm run test` or `npx playwright test`
4. **Code Quality**: Ensure your code passes all linters (`flake8`, `mypy`, `npm run lint`).
5. **Pull Request**: Push your branch and open a PR against `main`.

## Coding Standards

- **Python**: Follow PEP 8 guidelines (formatted via Black / Ruff). Include docstrings for all API endpoints and calculations.
- **TypeScript**: Use strict typing and React hooks best practices.
- **Security**: Never submit real credentials, hostnames, or confidential inventory dataset files in pull requests.
