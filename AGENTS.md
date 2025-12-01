# Repository Guidelines

This guide explains how to contribute effectively to the `viewcreator-genesis` FastAPI service while staying aligned with the repository’s conventions.

## Project Structure & Module Organization
- `app/` holds the application source code.
- `app/routers/` defines FastAPI route handlers; organize by domain (e.g., `ai_clipping`, `detection`).
- `app/services/` contains business logic and heavy lifting (AI, video processing); keep services focused and single-responsibility.
- `app/schemas/` defines Pydantic models for data validation and API contracts.
- `models/` stores machine learning model weights; do not commit large binary files (use git-lfs if necessary or download scripts).
- `tests/` contains Pytest specs; mirror the `app/` structure where possible.

## Build, Test, and Development Commands
- `docker-compose up --build` — launch the service with hot-reload and dependencies.
- `uvicorn app.main:app --reload` — run locally without Docker (requires local env setup).
- `pip install -r requirements.txt` — install Python dependencies.
- `pytest` — execute unit and integration tests.

## Coding Style & Naming Conventions
- Follow **PEP 8** for Python code style.
- Use **Type Hints** strictly for all function signatures and class attributes.
- Use `snake_case` for variables, functions, and file names.
- Use `PascalCase` for Classes and Pydantic Models.
- Use `UPPER_SNAKE_CASE` for constants and environment variables.
- Keep imports organized (standard lib, third-party, local app).

## Testing Guidelines
- Write tests using `pytest`.
- Mock external API calls (Groq, OpenRouter, S3) using `unittest.mock` or `pytest-mock` to avoid cost and latency in tests.
- Ensure critical pipeline stages (Download -> Transcribe -> Plan -> Render) have integration coverage.

## Commit & Pull Request Guidelines
- Write atomic commits with imperative summaries (e.g., `feat(detection): add yolo fallback`).
- Reference linked tickets in the commit body or PR description.
- PRs should include: purpose summary, testing results, and details on any new dependencies or env vars.

## Streaming Context & Sensitive Data
- Output is broadcast live to the user’s stream audience; double-check wording so nothing sensitive leaks into chat.
- **NEVER** echo or enumerate environment variables, secrets (API keys), or token values in responses.
- When in doubt, summarize sensitive configuration steps and point maintainers to secure channels for specifics.

## Security & Configuration Tips
- Secrets live in environment variables; never hard-code keys in source.
- Use `config.py` (Pydantic Settings) to load and validate environment variables.
- Ensure `yt-dlp` is kept up to date to handle YouTube changes.
- Validate all inputs in `routers` using Pydantic schemas before processing.
