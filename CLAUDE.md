# CLAUDE.md

## Project Overview
Memory Bank Playground — Streamlit app demonstrating Vertex AI Memory Bank using the Agent Engine SDK and ADK (Agent Development Kit).

## Tab Structure
The app uses `st.tabs()` with two tabs sharing the same `vertexai.Client` and engine pool:

### Tab 1: Vertex AI Agent Engine
1. Agent Engine — select existing or create new
2. Memory Bank Customization — models, topics, TTL (single Apply)
3. Session — create or reset
4. Chat — memory-aware conversation
5. Generate — extract memories from conversation
6. Retrieve — scope-based or similarity search
- Existing Memories (expandable) — view/delete stored memories
- Cleanup (expandable) — delete engine

### Tab 2: Agent Development Kit
1. Agent Engine — shared `_render_engine_section()` with Tab 1 (select/create), independent state (`mb_adk_engine_name`)
2. Agent Configuration — model, agent name, system instruction
3. Memory Configuration — retrieval strategy (Preload `PreloadMemoryTool` / Tool-based `LoadMemoryTool` / Custom callback / None), auto-generate mode, scope keys
4. Active Callbacks (expandable) — read-only summary of which callbacks are active
5. Session & Chat — Build Agent button, chat with transparency panel (system instruction, memories, tool calls, auto-gen status)

## Tech Stack
- Python 3.12+, Streamlit, `google-cloud-aiplatform` (Vertex AI SDK), `google-adk` (Agent Development Kit)
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
- `_render_engine_section(key_prefix, engine_state_key, on_disconnect)` is shared between both tabs for engine select/create parity
- Chat responses use similarity search to retrieve top 5 relevant memories as context before generating model replies
- Memory generation uses `direct_contents_source` with explicit scope (not `vertex_session_source`) to ensure topic customization is respected
- Models, memory topics, and TTL config are applied together in a single `agent_engines.update()` call to avoid overwriting each other
- Model dropdowns appear both at engine creation (initial config) and in the customization section (post-creation updates)

### Agent Development Kit (Tab 2)
- Uses `google.adk` (`LlmAgent`, `Runner`, `InMemorySessionService`, `VertexAiMemoryBankService`)
- `os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"` is set after `load_dotenv()`
- `VertexAiMemoryBankService` created from engine resource name: `agent_engine_id = name.split("/")[-1]`
- `Runner.run()` is synchronous and returns an event iterator; `InMemorySessionService` methods are async (wrapped with `_run_async()`)
- Four retrieval strategies: PreloadMemoryTool, LoadMemoryTool, custom `before_model_callback`, or none
- Two auto-generate modes: full session via `after_agent_callback` + `add_session_to_memory`, or last message via `before_agent_callback` + `direct_contents_source`
- Callbacks write debug info to a mutable dict in `st.session_state._adk_turn_debug_ref`, captured after each turn for the transparency panel
- Agent is rebuilt when config changes (tracked via `mb_adk_config_hash`)

## Conventions
- No sidebar — all UI is on the main page in sequential steps
- Each step wrapped in `st.container(border=True)` for visual separation
- Session state keys prefixed with `mb_` for Tab 1 (e.g., `mb_agent_engine_name`, `mb_session_name`, `mb_conversation`) and `mb_adk_` for Tab 2 (e.g., `mb_adk_engine_name`, `mb_adk_runner`, `mb_adk_conversation`)
- User input fields use placeholder text, no hardcoded defaults for IDs
- Disconnect/delete/new-session handlers must reset all relevant `mb_` session state keys
- All buttons use teal theme via CSS; all API operations have `st.spinner` with custom brain animation
- TTL timestamps shown on memories (created, expires, remaining) when TTL is configured
- `_format_memory_ttl()` helper used consistently across existing memories, generated memories, and retrieved memories

## Google API Notes
- Google AI APIs change frequently — always verify against latest docs before modifying API calls
- Match working patterns already in the codebase before trying different approaches
