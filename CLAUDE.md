# CLAUDE.md

## Project Overview
Memory Bank Playground — Streamlit app demonstrating Vertex AI Memory Bank using the Agent Engine SDK and ADK (Agent Development Kit).

## Tab Structure
The app uses `st.tabs()` with two tabs sharing the same `vertexai.Client` and engine pool:

### Tab 1: Vertex AI Agent Engine
1. **Agent Engine** — select existing or create new
   - `client.agent_engines.list()` — list available engines
   - `client.agent_engines.create(config={...})` — create engine with `MemoryBankConfig`
   - `client.agent_engines.get(name=...)` — fetch engine config (accessed via `.api_resource.context_spec.memory_bank_config`)
   - Existing Memories (expandable) — view/delete stored memories
     - `client.agent_engines.memories.retrieve(name=..., scope=...)` — list memories for a scope
     - `client.agent_engines.memories.get(name=...)` — get full memory detail (including TTL)
     - `client.agent_engines.memories.delete(name=...)` — delete individual memory
2. **Memory Bank Customization** — models, topics, TTL (single Apply)
   - `client.agent_engines.update(name=..., config={"context_spec": {"memory_bank_config": ...}})` — update models, topics, and TTL together
3. **Session** — create or reset
   - `client.agent_engines.sessions.create(name=..., config={"user_id": ...})` — create a new session
4. **Chat** — memory-aware conversation
   - `client.agent_engines.sessions.events.append(name=..., event=...)` — send user message
   - `client.agent_engines.memories.retrieve(name=..., scope=..., similarity_search_params=...)` — retrieve top-5 relevant memories as context
   - `GenerativeModel(model).generate_content(contents)` — generate reply with memory context
5. **Generate** — extract memories from conversation
   - `client.agent_engines.memories.generate(name=..., scope=..., direct_contents_source=..., config=...)` — generate memories from conversation events
   - `client.agent_engines.memories.get(name=...)` — fetch generated memory details
6. **Retrieve** — scope-based or similarity search
   - `client.agent_engines.memories.retrieve(name=..., scope=..., similarity_search_params=...)` — retrieve memories by scope or similarity
- Cleanup (expandable)
  - `client.agent_engines.delete(name=..., force=True)` — delete engine

### Tab 2: Agent Development Kit
1. **Agent Engine** — shared `_render_engine_section()` with Tab 1 (select/create), independent state (`mb_adk_engine_name`)
   - Memory Bank Settings (expandable) — read-only view of current engine config (models, topics, TTL) with pointer to Tab 1 for changes
     - `client.agent_engines.get(name=...)` — fetch engine config for display
   - Existing Memories (expandable) — view/delete stored memories
     - `client.agent_engines.memories.retrieve(name=..., scope=...)` — list memories
     - `client.agent_engines.memories.get(name=...)` — get full memory detail
     - `client.agent_engines.memories.delete(name=...)` — delete memory
2. **Agent Configuration** — model, agent name, system instruction
3. **Memory Configuration** — retrieval strategy, auto-generate mode, active callbacks display, scope keys
   - Retrieval strategy radio: Preload `PreloadMemoryTool` / Tool-based `LoadMemoryTool` / Custom callback / None
   - Auto-generate radio: Off / After each turn (full session) / After each turn (last message only)
   - Active callbacks display — inline bullet list (no heading, no separator) that dynamically updates based on retrieval + auto-gen selections:
     - Custom callback → `before_model_callback`; Preload/Tool-based → `memory_service`
     - Full session auto-gen → `after_agent_callback`; Last message auto-gen → `before_agent_callback`
     - `after_tool_callback` always shown (tool logging)
     - If retrieval=None and auto-gen=Off → shows `st.info` "Baseline mode" instead of bullet list
   - Scope configuration: user_id + app_name (always on)
4. **Session & Chat** — Build Agent button, chat with transparency panel (system instruction, memories, tool calls, auto-gen status)
   - `VertexAiMemoryBankService(agent_engine_id=..., project=..., location=...)` — create ADK memory service
   - `Runner(agent=..., app_name=..., session_service=..., memory_service=...)` — create ADK runner
   - `InMemorySessionService().create_session(app_name=..., user_id=...)` — create ADK session (async)
   - `runner.run_async(user_id=..., session_id=..., new_message=...)` — run agent turn (async iterator)
   - Custom callback: `client.agent_engines.memories.retrieve(name=..., scope=...)` — manual memory retrieval in `before_model_callback`
   - Auto-gen (full session): `memory_service.add_session_to_memory(session)` — in `after_agent_callback` (async)
   - Auto-gen (last message): `client.agent_engines.memories.generate(name=..., scope=..., direct_contents_source=...)` — in `before_agent_callback`

## Tech Stack
- Python 3.12+, Streamlit, `google-cloud-aiplatform` (Vertex AI SDK), `google-adk` (Agent Development Kit)
- Package manager: `uv`
- Config: `.env` file (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`)
- Auth: `gcloud auth application-default login` (Application Default Credentials)

## Key Files
- `main.py` — Entry point: page config, cached `vertexai.Client`, CSS, tab shell
- `shared.py` — Constants, type aliases, helpers (`_render_engine_section`, `_run_async`, `_format_memory_ttl`, `_populate_engine_config`, `get_client`)
- `tab_vertex.py` — Tab 1: Vertex AI Agent Engine (sessions, chat, generate, retrieve)
- `tab_adk.py` — Tab 2: Agent Development Kit (agent build, ADK runner, callbacks)

## Commands
- Run app: `uv run streamlit run main.py`
- Install deps: `uv sync`

## Architecture
- Uses `vertexai.Client` for all API calls (not the older `aiplatform` init pattern), cached via `@st.cache_resource` to avoid re-init on every Streamlit rerun
- Memory Bank types are aliased from `vertexai.types` (e.g., `MemoryBankConfig`, `SimilaritySearchConfig`, `GenerationConfig`)
- Agent engines are selected/created through the UI — no hardcoded ENGINE_ID
- `_render_engine_section(key_prefix, engine_state_key, on_disconnect)` is shared between both tabs for engine select/create parity
- Chat responses use similarity search to retrieve top 5 relevant memories as context before generating model replies
- Memory generation uses `direct_contents_source` with explicit scope (not `vertex_session_source`) to ensure topic customization is respected
- Models, memory topics, and TTL config are applied together in a single `agent_engines.update()` call to avoid overwriting each other
- Model dropdowns appear both at engine creation (initial config) and in the customization section (post-creation updates)
- `_populate_engine_config(client, engine_name)` fetches engine config via `agent_engines.get()` and pre-populates the customization UI (models, topics, TTL) in session state; runs once per engine connection (tracked via `mb_config_loaded_for`); clears stale widget keys before setting new values to avoid Streamlit caching old widget state
- Engine config is accessed via `engine_info.api_resource.context_spec.memory_bank_config` (not directly on `engine_info`)

### Agent Development Kit (Tab 2)
- Uses `google.adk` (`LlmAgent`, `Runner`, `InMemorySessionService`, `VertexAiMemoryBankService`)
- `os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"` is set after `load_dotenv()`
- `VertexAiMemoryBankService` created from engine resource name: `agent_engine_id = name.split("/")[-1]`
- `Runner.run()` is synchronous and returns an event iterator; `InMemorySessionService` methods are async (wrapped with `_run_async()`)
- Four retrieval strategies: PreloadMemoryTool, LoadMemoryTool, custom `before_model_callback`, or none
- Two auto-generate modes: full session via `after_agent_callback` + `add_session_to_memory`, or last message via `before_agent_callback` + `direct_contents_source`
- Active callbacks display is inline within the Memory Configuration section (not a separate expander); dynamically shows which ADK callbacks will be wired based on current retrieval + auto-gen selections; baseline mode (retrieval=None, auto-gen=Off) shows an info message instead
- Callbacks write debug info to a mutable dict in `st.session_state._adk_turn_debug_ref`, captured after each turn for the transparency panel
- Agent is rebuilt when config changes (tracked via `mb_adk_config_hash`)

## Conventions
- No sidebar — all UI is on the main page in sequential steps
- Each step wrapped in `st.container(border=True)` for visual separation
- Session state keys prefixed with `mb_` for Tab 1 (e.g., `mb_agent_engine_name`, `mb_session_name`, `mb_conversation`) and `mb_adk_` for Tab 2 (e.g., `mb_adk_engine_name`, `mb_adk_runner`, `mb_adk_conversation`)
- User input fields use placeholder text, no hardcoded defaults for IDs
- Disconnect/delete/new-session handlers must reset all relevant `mb_` session state keys (including `mb_config_loaded_for` to force config re-fetch on reconnect)
- All buttons use teal theme via CSS; all API operations have `st.spinner` with custom brain animation
- TTL timestamps shown on memories (created, expires, remaining) when TTL is configured
- `_format_memory_ttl()` helper used consistently across existing memories, generated memories, and retrieved memories

## Google API Notes
- Google AI APIs change frequently — always verify against latest docs before modifying API calls
- Match working patterns already in the codebase before trying different approaches
