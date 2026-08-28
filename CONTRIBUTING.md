# Contributing

Thanks for contributing. Short checklist to get started:

- Create a branch named `feature/your-topic`.
- Add tests for new behavior and run `pytest`.
- Follow the coding style (ruff config present).
- Open a pull request against `main`.

For development:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
pytest
```
