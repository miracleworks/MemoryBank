# Memory Bank Feature Demo
A Streamlit app demonstrating the Vertex AI Memory Bank feature using the Agent Engine SDK.

## Reference
- [Colab notebook](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/agents/agent_engine/memory_bank/get_started_with_memory_bank.ipynb)

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

#### Step 1: Memory Bank Configuration
- **Embedding Model**: Select from `text-embedding-005`, `text-embedding-004`, `text-multilingual-embedding-002`
- **Generation Model**: Select from:
  - `gemini-3.1-pro`, `gemini-3-flash`, `gemini-3.1-flash-lite` (preview)
  - `gemini-2.5-pro`, `gemini-2.5-flash` (default), `gemini-2.5-flash-lite` (GA)
  - `gemini-2.0-flash`, `gemini-2.0-flash-lite` (GA)

#### Step 2: Agent Engine
Two modes via radio toggle:
- **Select existing**: Lists all available Agent Engines in the project with a dropdown (display name + resource name). Includes a Refresh button to reload the list.
- **Create new**: Provide an Engine Name (`display_name`) and provision a new Agent Engine with the Memory Bank config from Step 1.
- **Disconnect**: Detach from the current engine without deleting it.
- API calls:
  - `client.agent_engines.list()`
  - `client.agent_engines.create(config={...})`

#### Existing Memories
- Expandable section to retrieve previously stored memories by **User ID**
- **Load Memories**: Fetches all memories scoped to the given user
- **Delete All Memories**: Removes all memories for the user
- API calls:
  - `client.agent_engines.memories.retrieve(name=..., scope={...})`
  - `client.agent_engines.memories.delete(name=...)`

#### Step 3: Create Session
- **User ID**: Text input for identifying the user in the session
- **Session Display Name**: Text input for naming the session
- API call: `client.agent_engines.sessions.create(name=..., user_id=..., config={...})`

#### Step 4: Chat Conversation
- **Memory-aware responses**: The model retrieves relevant memories (top 5 via similarity search) before generating each reply
- **Load Sample Conversation**: Pre-populates a hotel check-in scenario (Emma Chen) with 8 turns
- **Chat interface**: Scrollable message container with `st.chat_message` bubbles. Inline text input + Send button that clears after sending.
- Both user and model turns are appended to the Memory Bank session via `sessions.events.append()`
- New memories are not created automatically — use Step 5 to generate them
- API calls:
  - `client.agent_engines.sessions.events.append(name=..., author=..., ...)`
  - `client.agent_engines.memories.retrieve(name=..., scope={...}, similarity_search_params={...})`

#### Step 5: Generate Memories
- Triggers memory generation with the session as source
- Displays extracted memories with NEW/UPDATED labels
- Performs both extraction (fact extraction from conversation) and consolidation (intelligent merge with existing memories)
- API calls:
  - `client.agent_engines.memories.generate(name=..., vertex_session_source={...}, config={...})`
  - `client.agent_engines.memories.get(name=...)`

#### Step 6: Retrieve Memories
- **Scope-based**: Returns ALL memories for the user
- **Similarity search**: Returns TOP K most relevant memories for a query
  - **Search Query**: Text input
  - **Top K**: Number input (1–20, default 3)
  - Displays similarity scores alongside results
- API call: `client.agent_engines.memories.retrieve(name=..., scope={...}, similarity_search_params={...})`

#### Cleanup
- Expandable section with **Delete Agent Engine** button
- Safely clears all session state after successful deletion
- API call: `client.agent_engines.delete(name=..., force=True)`
