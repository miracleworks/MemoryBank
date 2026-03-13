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
- Memory generation uses `direct_contents_source` with explicit scope (not `vertex_session_source`) to ensure topic customization is respected
- Models, memory topics, and TTL config are applied together in a single `agent_engines.update()` call to avoid overwriting each other
- Model dropdowns appear both at engine creation (initial config) and in the customization section (post-creation updates)

## Conventions
- No sidebar — all UI is on the main page in sequential steps
- Each step wrapped in `st.container(border=True)` for visual separation
- Session state keys prefixed with `mb_` (e.g., `mb_agent_engine_name`, `mb_session_name`, `mb_conversation`, `mb_selected_topics`)
- User input fields use placeholder text, no hardcoded defaults for IDs
- Disconnect/delete/new-session handlers must reset all relevant `mb_` session state keys
- All buttons use teal theme via CSS; all API operations have `st.spinner` with custom brain animation
- TTL timestamps shown on memories (created, expires, remaining) when TTL is configured
- `_format_memory_ttl()` helper used consistently across existing memories, generated memories, and retrieved memories

## Google API Notes
- Google AI APIs change frequently — always verify against latest docs before modifying API calls
- Match working patterns already in the codebase before trying different approaches
