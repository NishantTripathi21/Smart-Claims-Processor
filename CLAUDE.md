# Smart Claims Processor contributor notes

## Project overview

This repository contains the completed backend proof of concept for an
insurance claims processor. LangGraph orchestrates the six-agent pipeline and
CrewAI runs the focused fraud-analysis sub-crew. The backend uses FastAPI,
SQLite, ChromaDB memory, Gemini/Groq providers, and US/India YAML profiles.

## Common commands

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn api.main:app --port 8000
pytest tests/ -q
python scripts/seed_policies.py
python scripts/clean_data.py
```

## Architecture

The completed LangGraph routes intake to fraud, damage, policy, settlement,
and communication. Invalid claims go directly to communication, low-value
claims use fast mode, and confirmed fraud goes to auto-rejection.

`ClaimsState` in `src/models/state.py` is the shared typed state. Each agent
returns a structured Pydantic output and appends an audit trace. Memory is
managed by `src/memory/manager.py`; guardrails are managed by
`src/guardrails/manager.py`; country-aware PII masking is in
`src/security/pii_masker.py`.

The evaluator, manual HITL queue, and React frontend are not included in this
completed subset.

## Configuration

Provider and country selection come from `.env`. Shared tunables are in
`configs/base.yaml`; country overrides are in `configs/countries/`. Runtime
settings overrides take precedence over environment and YAML values.
