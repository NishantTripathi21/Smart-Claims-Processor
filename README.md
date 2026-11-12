# Smart Claims Processor

Smart Claims Processor is a backend proof of concept for automated insurance
claim processing. It combines LangGraph orchestration with a CrewAI fraud
sub-crew, structured agent outputs, country-aware configuration, memory,
guardrails, and a FastAPI API.

This repository contains the implemented scope:

- six production pipeline agents: intake, fraud, damage, policy, settlement,
  and communication;
- normal, fast-mode, intake-failure, and confirmed-fraud routes;
- short-term LangGraph state plus long-term and episodic ChromaDB memory;
- PII masking, token/cost limits, timeout checks, and confidence validation;
- US and India country profiles;
- FastAPI authentication, claim persistence, analytics, appeals, policies, and
  settings endpoints.

The evaluator agent, manual HITL queue, and React frontend are planned
follow-up work and are intentionally not part of this completed subset.

## Architecture

LangGraph owns the claim state and routes each claim:

```text
Intake
  |-- invalid ------------------------> Communication
  |-- fast mode (< configured amount) -> Settlement -> Communication
  `-- normal -------------------------> Fraud Crew
                                            |-- confirmed fraud -> Auto Reject
                                            `-- normal -> Damage -> Policy
                                                         -> Settlement
                                                         -> Communication
```

CrewAI is used only inside the fraud stage. Its pattern analyst, anomaly
detector, and social validator run as a focused fraud sub-crew; LangGraph
remains the top-level orchestrator.

## Project layout

```text
api/                 FastAPI routes, authentication, and SQLite models
configs/             Shared and country-specific YAML configuration
src/agents/          Six LangGraph nodes and the CrewAI fraud sub-crew
src/guardrails/      Per-claim budget, timeout, loop, and confidence checks
src/memory/          ChromaDB long-term claims and episodic event memory
src/models/          Typed state and Pydantic agent output schemas
src/security/        Country-aware PII masking and audit logging
src/tools/           Policy, fraud-pattern, and claim helper tools
tests/               Focused tests for implemented components
scripts/             Data seeding and cleanup utilities
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --port 8000
```

Set `LLM_PROVIDER` (`groq` or `gemini`), the matching API key, `COUNTRY`
(`us` or `india`), and `API_SECRET_KEY` in `.env`.

Run tests:

```bash
pytest tests/ -q
```

Seed sample policies or clean generated data:

```bash
python scripts/seed_policies.py
python scripts/clean_data.py
```

## Agent contracts

Each agent receives the shared `ClaimsState` and returns a typed state update.
The update is appended to the pipeline trace and passed to the next node.
The API persists the final decision, settlement, fraud result, trace, and
communication output in SQLite.

## Country profiles

`configs/countries/us.yaml` and `configs/countries/india.yaml` define currency,
claim types, PII fields, depreciation, settlement rules, required documents,
and communication text. Runtime overrides take precedence over `.env`, country
configuration, and shared base configuration.

## Remaining work

The next implementation phase is to add evaluator quality gates, durable
human review pause/resume, a reviewer interface, and end-to-end deployment
validation.
