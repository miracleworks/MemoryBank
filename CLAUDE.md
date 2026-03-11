# CLAUDE.md

## Project Overview
Streamlit app demonstrating Vertex AI Memory Bank using the Agent Engine SDK.

## Tech Stack
- Python 3.12+, Streamlit, `google-cloud-aiplatform` (Vertex AI SDK)
- Package manager: `uv`
- Config: `.env` file (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`)
- Auth: `gcloud auth application-default login` (Application Default Credentials)

## Key Files
- `main.py` — Streamlit app (single-page, all Memory Bank workflow steps)

## Commands
- Run app: `uv run streamlit run main.py`
- Install deps: `uv sync`

## Architecture
- Uses `vertexai.Client` for all API calls (not the older `aiplatform` init pattern)
- Memory Bank types are aliased from `vertexai.types` (e.g., `MemoryBankConfig`, `SimilaritySearchConfig`, `GenerationConfig`)
- Agent engines are selected/created through the UI — no hardcoded ENGINE_ID
- Chat responses use similarity search to retrieve top 5 relevant memories as context before generating model replies

## Conventions
- No sidebar — all UI is on the main page in sequential steps
- Each step wrapped in `st.container(border=True)` for visual separation
- Session state keys prefixed with `mb_` (e.g., `mb_agent_engine_name`, `mb_session_name`, `mb_conversation`)
- User input fields use placeholder text, no hardcoded defaults for IDs

## Google API Notes
- Google AI APIs change frequently — always verify against latest docs before modifying API calls
- Match working patterns already in the codebase before trying different approaches
