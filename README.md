# 🧠 Memory Bank Playground
A Streamlit app demonstrating Vertex AI Memory Bank using both the Agent Engine SDK and the Agent Development Kit (ADK).

## Prerequisites
- Python 3.12+
- A GCP project with the Vertex AI API enabled
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed

## Setup
1. Authenticate with GCP:
   ```bash
   gcloud auth application-default login
   ```
   This stores credentials locally so the app can access Vertex AI. Each user needs to run this once.

2. Copy the example env file and fill in your values:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your GCP project ID and location.
3. Install dependencies: `uv sync`
4. Run the app: `uv run streamlit run main.py`

## App Structure

The app has two tabs sharing the same engine pool:

### Tab 1 — Vertex AI Agent Engine
Demonstrates how to make API calls directly to Vertex AI Agent Engine Sessions and Memory Bank using the Vertex AI Agent Engine SDK. Use the Vertex AI Agent Engine SDK if you don't want an agent framework to orchestrate calls for you, or you want to integrate Sessions and Memory Bank with agent frameworks other than Agent Development Kit (ADK).

#### 1. Agent Engine
Two modes via radio toggle:
- **Select existing**: Lists all available Agent Engines in the project with a dropdown (display name + resource name). Includes a Refresh button to reload the list.
- **Create new**: Provide an Engine Name, select Embedding Model and Generation Model, then provision a new Agent Engine.
- **Disconnect**: Detach from the current engine without deleting it.
- API calls:
  - `client.agent_engines.list()`
  - `client.agent_engines.create(config={...})`

#### Existing Memories (expandable)
- Retrieve previously stored memories by **User ID**
- **Load Memories**: Fetches all memories scoped to the given user
- Displays TTL info (created, expires, remaining time) when TTL is configured
- **Delete All Memories**: Removes all memories for the user
- API calls:
  - `client.agent_engines.memories.retrieve(name=..., scope={...})`
  - `client.agent_engines.memories.get(name=...)` (for TTL timestamps)
  - `client.agent_engines.memories.delete(name=...)`

#### 2. Memory Bank Customization
Combined configuration section for models, memory topics, and TTL — all applied together via a single **Apply Configuration** button.

**Models** — Change the generation or embedding model after engine creation:
- **Embedding Model**: `text-embedding-005` (default), `text-embedding-004`, `text-multilingual-embedding-002`
- **Generation Model**: `gemini-2.5-flash` (default), plus `gemini-3.1-pro`, `gemini-3-flash`, `gemini-2.5-pro`, and others

**Memory Topics** — Select which types of information Memory Bank should extract:
- `USER_PERSONAL_INFO`: Names, relationships, hobbies, important dates
- `USER_PREFERENCES`: Likes, dislikes, preferred styles, patterns
- `KEY_CONVERSATION_DETAILS`: Milestones, conclusions, task outcomes
- `EXPLICIT_INSTRUCTIONS`: Explicit remember/forget instructions from the user

All topics are active by default. Selecting a subset restricts extraction to only those topics.

**Memory TTL** — Set expiration times for memories (three modes):
- **None**: Memories persist indefinitely (default)
- **Default TTL**: Single duration applied to all create/update operations
- **Granular TTL**: Per-operation durations:
  - `CreateMemory TTL`: Manually created memories
  - `GenerateMemories (new) TTL`: Newly generated memories
  - `GenerateMemories (updated) TTL`: Updated memories during consolidation
- Each TTL supports seconds, minutes, or hours for easy testing

API call: `client.agent_engines.update(name=..., config={...})` (models + topics + TTL sent together)

#### 3. Session
- **User ID**: Text input for identifying the user in the session
- **Session Display Name**: Text input for naming the session
- **New Session**: Reset the current session to start fresh without disconnecting from the engine
- API call: `client.agent_engines.sessions.create(name=..., user_id=..., config={...})`

#### 4. Chat
- **Memory-aware responses**: The model retrieves relevant memories (top 5 via similarity search) before generating each reply
- **Load Sample Conversation**: Pre-populates a hotel check-in scenario (Emma Chen) with 8 turns
- **Chat interface**: Scrollable message container with `st.chat_message` bubbles. Text input clears after sending via keyboard or button.
- Both user and model turns are appended to the Memory Bank session via `sessions.events.append()`
- New memories are not created automatically — use Step 5 to generate them
- API calls:
  - `client.agent_engines.sessions.events.append(name=..., author=..., ...)`
  - `client.agent_engines.memories.retrieve(name=..., scope={...}, similarity_search_params={...})`

#### 5. Generate
- Triggers memory generation using `direct_contents_source` with explicit `scope={"user_id": ...}` to respect topic customization
- Displays extracted memories with NEW/UPDATED labels and TTL timestamps
- Performs both extraction (fact extraction from conversation) and consolidation (intelligent merge with existing memories)
- Only extracts facts matching the active memory topics configured in the customization section
- TTL configuration is applied automatically to generated memories
- API calls:
  - `client.agent_engines.memories.generate(name=..., scope={...}, direct_contents_source={...}, config={...})`
  - `client.agent_engines.memories.get(name=...)`

#### 6. Retrieve
- **Scope-based**: Returns ALL memories for the user
- **Similarity search**: Returns TOP K most relevant memories for a query
  - **Search Query**: Text input
  - **Top K**: Number input (1–20, default 3)
  - Displays similarity scores and TTL info alongside results
- API call: `client.agent_engines.memories.retrieve(name=..., scope={...}, similarity_search_params={...})`

#### Cleanup (expandable)
- **Delete Agent Engine** button
- Safely clears all session state after successful deletion
- API call: `client.agent_engines.delete(name=..., force=True)`

### Tab 2 — Agent Development Kit
Demonstrates how you can use Memory Bank with ADK to manage long-term memories. After you configure your Agent Development Kit (ADK) agent to use Memory Bank, your agent orchestrates calls to Memory Bank to manage long-term memories for you.

Shares the same engine pool as Tab 1.

> **Scope note:** Tab 2 uses ADK's `VertexAiMemoryBankService`, which always scopes memories to `{user_id, app_name}`. Memories created in Tab 1 (scoped to `{user_id}` only) are in a different scope and won't be visible here. Use Tab 2's Existing Memories panel to verify what's available before chatting.

### Agent Engine Usage in ADK Tab

> **Important Concept:** In Tab 2 (Agent Development Kit), the agent is run locally using ADK’s `LlmAgent` and `Runner`. The Agent Engine is **not** used to execute the agent itself. Instead, the Agent Engine is leveraged for memory management via the `VertexAiMemoryBankService`. This separation allows flexible agent orchestration while utilizing Google’s managed memory infrastructure.

- **Agent Execution:** Local (Streamlit + ADK)
- **Memory Management:** Delegated to Agent Engine (Vertex AI Memory Bank)
- **Why:** This separation allows flexible agent orchestration while utilizing Google’s managed memory infrastructure.

**Example:**
When building the agent, the `Runner` is initialized with a memory service that connects to the Agent Engine for memory operations, but the agent logic itself is handled locally:

```python
runner = Runner(
    agent=agent,
    app_name=adk_app_name,
    session_service=session_service,
    memory_service=VertexAiMemoryBankService(
        agent_engine_id=adk_engine_id,
        project=PROJECT_ID,
        location=LOCATION,
    ),
)
```

#### UI Sections

##### 1. Agent Engine
Same UI as Tab 1 — radio toggle between **Select existing** and **Create new**:
- **Select existing**: Lists available engines, click **Use this Engine** to connect
- **Create new**: Provide an Engine Name, select Embedding Model and Generation Model, then provision a new Agent Engine
- **Disconnect**: Detach and reset all ADK state (runner, session, conversation)
- **Refresh list**: Reload available engines
- Both tabs share the same engine list cache

##### Existing Memories (expandable)
- Check what memories exist for a given `{user_id, app_name}` scope before chatting
- Both **User ID** and **App Name** are required (matches the scope ADK uses)
- Displays memories with TTL info, with option to **Delete All**

##### 2. Agent Configuration
- **Model**: Generation model for the ADK agent
- **Agent Name**: Internal name for the `LlmAgent` (default: `memory_agent`)
- **System Instruction**: The agent's base system prompt

##### 3. Memory Configuration

**Retrieval Strategy** — how the agent accesses memories each turn:

| Strategy                          | How it works                                                                                                                                                      | When to use                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Preload — `PreloadMemoryTool`** | Silently runs similarity search using the user's message and injects matching memories into the system instruction before every model call. No visible tool call. | Default choice. Good for always-on memory context.                    |
| **Tool-based — `LoadMemoryTool`** | Available as a tool the agent can call. The agent decides whether to invoke it based on the conversation. Shows as a tool call in transparency.                   | When you want the agent to selectively use memory only when relevant. |
| **Custom callback**               | `before_model_callback` retrieves all memories for the scope via Agent Engine SDK (no similarity search). Lower latency.                                          | When you want all memories loaded without similarity ranking.         |
| **None**                          | No memory retrieval at all.                                                                                                                                       | Baseline comparison to see how the agent behaves without memory.      |

**Auto-Generate Memories** — when to automatically create memories from conversation:

| Mode                                    | How it works                                                                             | Trade-offs                                                                       |
| --------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Off**                                 | No automatic generation.                                                                 | Use when testing retrieval only, or generating manually in Tab 1.                |
| **After each turn (full session)**      | `after_agent_callback` sends the entire session to `add_session_to_memory`.              | Better context for extraction, but cost grows with session length. Non-blocking. |
| **After each turn (last message only)** | `before_agent_callback` sends only the latest user message via `direct_contents_source`. | Lower cost, but loses conversational context. Non-blocking.                      |

**Active Callbacks** — displayed inline below the retrieval/auto-generate selections (no separate section or heading). Dynamically updates as you change options:
- Preload or Tool-based → `memory_service` — `VertexAiMemoryBankService`
- Custom callback → `before_model_callback` — retrieves memories via Agent Engine SDK and injects into system instruction
- Auto-gen full session → `after_agent_callback` — sends full session to `add_session_to_memory`
- Auto-gen last message → `before_agent_callback` — sends latest user message via `direct_contents_source`
- `after_tool_callback` — always active (logs tool calls for the transparency panel)
- If retrieval=None and auto-gen=Off → shows "Baseline mode" info message instead of the bullet list

**Scope Configuration**: Both `user_id` and `app_name` are always included in the scope. The **App Name** value (default: `memory_playground`) determines the `app_name` scope key.

##### 4. Session & Chat
- **User ID**: Identifies the user for memory scoping
- **Build Agent**: Constructs the `LlmAgent` + `Runner` + `VertexAiMemoryBankService` from current config. Must be clicked before chatting. Auto-rebuilds if config changes.
- **New Session**: Creates a fresh ADK session (clears conversation but keeps the agent)
- **Chat interface**: Scrollable message container with chat bubbles
- **Last Turn Details** (expandable, below chat): Shows what happened during the most recent turn:
  - System instruction sent to model (with any injected memories)
  - Memories retrieved (for Custom callback mode)
  - Tool calls made (for Tool-based mode)
  - Whether auto-generate was triggered

#### Walkthrough: Testing Each Feature

##### Getting started
1. **Create an engine** in Tab 1 (or use an existing one)
2. Switch to **Tab 2** and **Connect** to the same engine
3. Enter a **User ID** (e.g., `testuser`)

##### Testing Preload (auto)
1. First, create some memories:
   - Set Retrieval to **None**, Auto-Generate to **After each turn (full session)**
   - Click **Build Agent**
   - Chat: `"Hi, I'm Alex. I love rock climbing and I'm allergic to peanuts."`
   - Wait a few seconds for memory generation to complete in the background
2. Verify memories exist:
   - Open **Existing Memories**, enter the same User ID and App Name, click **Load Memories**
   - You should see extracted facts like "Alex loves rock climbing"
3. Test retrieval:
   - Set Retrieval to **Preload (auto)**, Auto-Generate to **Off**
   - Click **Build Agent**, then **New Session** (fresh session with no history)
   - Ask: `"What do you know about me?"`
   - The agent should answer using the stored memories
   - Check **Last Turn Details** — the system instruction should contain the injected memories

##### Testing Tool-based (agent decides)
1. Ensure memories exist (follow step 1-2 above)
2. Set Retrieval to **Tool-based (agent decides)**, Auto-Generate to **Off**
3. Click **Build Agent**, then **New Session**
4. Ask: `"What do you know about me?"` — agent should call `LoadMemoryTool`
5. Ask: `"Hi!"` — agent may skip the tool (not needed for a greeting)
6. Check **Last Turn Details** for tool call logs

##### Testing Custom callback
1. Ensure memories exist (follow step 1-2 above)
2. Set Retrieval to **Custom callback**, Auto-Generate to **Off**
3. Click **Build Agent**, then **New Session**
4. Ask: `"What do you know about me?"`
5. Check **Last Turn Details** — should show all memories retrieved via scope (no similarity search) and the augmented system instruction

##### Testing None (baseline)
1. Set Retrieval to **None**, Auto-Generate to **Off**
2. Click **Build Agent**, then **New Session**
3. Ask: `"What do you know about me?"` — agent should have no memory context
4. Compare the response to the other strategies

##### Testing Auto-Generate modes
1. Set Auto-Generate to **After each turn (full session)** or **After each turn (last message only)**
2. Chat with personal information: `"Remember that I prefer window seats"`
3. Open **Existing Memories** and **Load Memories** — the new fact should appear after a few seconds
4. Click **New Session** and ask the agent about your preferences — it should recall the information

##### Comparing strategies
To compare how different strategies affect responses:
1. Generate memories once with a rich conversation
2. For each strategy: set it, **Build Agent**, **New Session**, ask the same question
3. Compare responses and **Last Turn Details** across strategies

## UI Features
- **Teal button theme**: All buttons styled with consistent teal color (`#0D9488`)
- **Dancing brain spinner**: Custom animated brain emoji replaces default Streamlit spinner during loading operations
- **All long operations have spinners**: Engine creation, session creation, memory generation, retrieval, deletion
