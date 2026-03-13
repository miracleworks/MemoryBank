# Memory Bank Feature Demo
A Streamlit app demonstrating the Vertex AI Memory Bank feature using the Agent Engine SDK.

## Reference
- [Memory Bank on ADK Colab](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/agents/agent_engine/memory_bank/get_started_with_memory_bank_on_adk.ipynb)
- [Memory Bank Colab](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/agents/agent_engine/memory_bank/get_started_with_memory_bank.ipynb)

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

### Main Page — Memory Bank Workflow

#### Step 1: Agent Engine
Two modes via radio toggle:
- **Select existing**: Lists all available Agent Engines in the project with a dropdown (display name + resource name). Includes a Refresh button to reload the list.
- **Create new**: Provide an Engine Name, select Embedding Model and Generation Model, then provision a new Agent Engine.
- **Disconnect**: Detach from the current engine without deleting it.
- API calls:
  - `client.agent_engines.list()`
  - `client.agent_engines.create(config={...})`

#### Existing Memories
- Expandable section to retrieve previously stored memories by **User ID**
- **Load Memories**: Fetches all memories scoped to the given user
- Displays TTL info (created, expires, remaining time) when TTL is configured
- **Delete All Memories**: Removes all memories for the user
- API calls:
  - `client.agent_engines.memories.retrieve(name=..., scope={...})`
  - `client.agent_engines.memories.get(name=...)` (for TTL timestamps)
  - `client.agent_engines.memories.delete(name=...)`

#### Memory Bank Customization
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

#### Step 2: Create Session
- **User ID**: Text input for identifying the user in the session
- **Session Display Name**: Text input for naming the session
- **New Session**: Reset the current session to start fresh without disconnecting from the engine
- API call: `client.agent_engines.sessions.create(name=..., user_id=..., config={...})`

#### Step 3: Chat Conversation
- **Memory-aware responses**: The model retrieves relevant memories (top 5 via similarity search) before generating each reply
- **Load Sample Conversation**: Pre-populates a hotel check-in scenario (Emma Chen) with 8 turns
- **Chat interface**: Scrollable message container with `st.chat_message` bubbles. Text input clears after sending via keyboard or button.
- Both user and model turns are appended to the Memory Bank session via `sessions.events.append()`
- New memories are not created automatically — use Step 4 to generate them
- API calls:
  - `client.agent_engines.sessions.events.append(name=..., author=..., ...)`
  - `client.agent_engines.memories.retrieve(name=..., scope={...}, similarity_search_params={...})`

#### Step 4: Generate Memories
- Triggers memory generation using `direct_contents_source` with explicit `scope={"user_id": ...}` to respect topic customization
- Displays extracted memories with NEW/UPDATED labels and TTL timestamps
- Performs both extraction (fact extraction from conversation) and consolidation (intelligent merge with existing memories)
- Only extracts facts matching the active memory topics configured in the customization section
- TTL configuration is applied automatically to generated memories
- API calls:
  - `client.agent_engines.memories.generate(name=..., scope={...}, direct_contents_source={...}, config={...})`
  - `client.agent_engines.memories.get(name=...)`

#### Step 5: Retrieve Memories
- **Scope-based**: Returns ALL memories for the user
- **Similarity search**: Returns TOP K most relevant memories for a query
  - **Search Query**: Text input
  - **Top K**: Number input (1–20, default 3)
  - Displays similarity scores and TTL info alongside results
- API call: `client.agent_engines.memories.retrieve(name=..., scope={...}, similarity_search_params={...})`

#### Cleanup
- Expandable section with **Delete Agent Engine** button
- Safely clears all session state after successful deletion
- API call: `client.agent_engines.delete(name=..., force=True)`

## UI Features
- **Teal button theme**: All buttons styled with consistent teal color (`#0D9488`)
- **Dancing brain spinner**: Custom animated brain emoji replaces default Streamlit spinner during loading operations
- **All long operations have spinners**: Engine creation, session creation, memory generation, retrieval, deletion
