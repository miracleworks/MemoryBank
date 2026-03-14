import streamlit as st
import uuid
import asyncio
import datetime
import vertexai
from vertexai import types
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv
import os

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import VertexAiMemoryBankService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.genai.types import Content, Part

load_dotenv()

# ADK requires Vertex AI mode — remove API key to avoid conflict
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ.pop("GOOGLE_API_KEY", None)


def _render_engine_section(key_prefix, engine_state_key, on_disconnect):
    """Render the Agent Engine select/create section. Used by both tabs."""
    with st.container(border=True):
        st.markdown("### 1. Agent Engine")

        if st.session_state[engine_state_key]:
            st.success(f"Agent Engine active: `{st.session_state[engine_state_key]}`")
            if st.button("Disconnect", key=f"{key_prefix}_disconnect_engine"):
                on_disconnect()
                st.rerun()
        else:
            engine_mode = st.radio(
                "Engine Source",
                ["Select existing", "Create new"],
                horizontal=True,
                key=f"{key_prefix}_engine_mode",
            )

            if engine_mode == "Select existing":
                if st.button("Refresh list", key=f"{key_prefix}_refresh_engines"):
                    st.session_state.pop("mb_engine_list", None)
                    st.rerun()

                if "mb_engine_list" not in st.session_state:
                    with st.spinner("Loading agent engines..."):
                        try:
                            engines = list(client.agent_engines.list())
                            st.session_state.mb_engine_list = engines
                        except Exception as e:
                            st.error(f"Failed to list engines: {e}")
                            st.session_state.mb_engine_list = []

                engines = st.session_state.get("mb_engine_list", [])
                if engines:
                    engine_options = {}
                    for eng in engines:
                        res = eng.api_resource
                        label = f"{res.display_name or 'Unnamed'} — {res.name}"
                        engine_options[label] = res.name
                    col_sel, col_use = st.columns([3, 1])
                    with col_sel:
                        selected_label = st.selectbox(
                            "Available Engines", list(engine_options.keys()),
                            key=f"{key_prefix}_engine_select",
                        )
                    with col_use:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Use this Engine", key=f"{key_prefix}_use_engine", use_container_width=True):
                            st.session_state[engine_state_key] = engine_options[selected_label]
                            st.session_state.pop("mb_engine_list", None)
                            st.rerun()
                else:
                    st.info("No agent engines found. Create a new one.")

            else:
                engine_display_name = st.text_input(
                    "Engine Name", value="memory-bank-engine",
                    key=f"{key_prefix}_engine_name",
                )
                col_em, col_gm = st.columns(2)
                with col_em:
                    create_embedding = st.selectbox(
                        "Embedding Model", EMBEDDING_MODELS, index=0,
                        key=f"{key_prefix}_create_embedding",
                    )
                with col_gm:
                    create_generation = st.selectbox(
                        "Generation Model", GENERATION_MODELS, index=4,
                        key=f"{key_prefix}_create_generation",
                    )
                if st.button("Create Agent Engine", key=f"{key_prefix}_create_engine"):
                    with st.spinner("Provisioning Agent Engine (this may take a few minutes)..."):
                        try:
                            mb_config = MemoryBankConfig(
                                similarity_search_config=SimilaritySearchConfig(
                                    embedding_model=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{create_embedding}"
                                ),
                                generation_config=GenerationConfig(
                                    model=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{create_generation}"
                                ),
                            )
                            ae = client.agent_engines.create(
                                config={
                                    "display_name": engine_display_name,
                                    "context_spec": {"memory_bank_config": mb_config},
                                }
                            )
                            st.session_state[engine_state_key] = ae.api_resource.name
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create Agent Engine: {e}")


def _run_async(coro):
    """Run an async coroutine from synchronous Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _format_memory_ttl(memory):
    """Return a TTL info string if the memory has expiration data."""
    expire = getattr(memory, "expire_time", None)
    create = getattr(memory, "create_time", None)
    if not expire:
        return ""
    parts = []
    if create:
        parts.append(f"Created: {create}")
    parts.append(f"Expires: {expire}")
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if hasattr(expire, "timestamp"):
        remaining = expire - now
        if remaining.total_seconds() > 0:
            mins, secs = divmod(int(remaining.total_seconds()), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                parts.append(f"Remaining: {hours}h {mins}m {secs}s")
            elif mins > 0:
                parts.append(f"Remaining: {mins}m {secs}s")
            else:
                parts.append(f"Remaining: {secs}s")
        else:
            parts.append("**Expired**")
    return " | ".join(parts)

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

st.set_page_config(page_title="Memory Bank Playground", layout="wide")

if not PROJECT_ID or not LOCATION:
    st.title("Configuration Required")
    st.error("Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in your `.env` file.")
    st.stop()

# Initialize Vertex AI
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

# Ensure ADK env vars are set
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

# --- SESSION STATE INITIALIZATION ---
# Tab 1 state
if "mb_agent_engine_name" not in st.session_state:
    st.session_state.mb_agent_engine_name = None
if "mb_session_name" not in st.session_state:
    st.session_state.mb_session_name = None
if "mb_guest_id" not in st.session_state:
    st.session_state.mb_guest_id = None
if "mb_conversation" not in st.session_state:
    st.session_state.mb_conversation = []
if "mb_event_count" not in st.session_state:
    st.session_state.mb_event_count = 0

# Tab 2 (ADK) state
if "mb_adk_engine_name" not in st.session_state:
    st.session_state.mb_adk_engine_name = None
if "mb_adk_model" not in st.session_state:
    st.session_state.mb_adk_model = "gemini-2.5-flash"
if "mb_adk_agent_name" not in st.session_state:
    st.session_state.mb_adk_agent_name = "memory_agent"
if "mb_adk_instruction" not in st.session_state:
    st.session_state.mb_adk_instruction = "Answer the user's questions"
if "mb_adk_retrieval" not in st.session_state:
    st.session_state.mb_adk_retrieval = "Preload — `PreloadMemoryTool`"
if "mb_adk_auto_gen" not in st.session_state:
    st.session_state.mb_adk_auto_gen = "Off"
if "mb_adk_scope_keys" not in st.session_state:
    st.session_state.mb_adk_scope_keys = ["user_id", "app_name"]
if "mb_adk_app_name" not in st.session_state:
    st.session_state.mb_adk_app_name = "memory_playground"
if "mb_adk_user_id" not in st.session_state:
    st.session_state.mb_adk_user_id = ""
if "mb_adk_session_id" not in st.session_state:
    st.session_state.mb_adk_session_id = None
if "mb_adk_conversation" not in st.session_state:
    st.session_state.mb_adk_conversation = []
if "mb_adk_runner" not in st.session_state:
    st.session_state.mb_adk_runner = None
if "mb_adk_session_service" not in st.session_state:
    st.session_state.mb_adk_session_service = None
if "mb_adk_memory_service" not in st.session_state:
    st.session_state.mb_adk_memory_service = None
if "mb_adk_turn_debug" not in st.session_state:
    st.session_state.mb_adk_turn_debug = []
if "mb_adk_config_hash" not in st.session_state:
    st.session_state.mb_adk_config_hash = None


# --- CUSTOM STYLING ---
st.markdown("""
<style>
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(240, 242, 246, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(200, 205, 215, 0.4);
    padding: 4px;
}

/* --- Modern button theme (Teal) --- */
div[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    background-color: #0D9488 !important;
    color: white !important;
    border: none !important;
}
div[data-testid="stButton"] button:hover {
    background-color: #0F766E !important;
}

/* --- Custom spinner with dancing brain --- */
div[data-testid="stSpinner"] > div {
    display: flex;
    align-items: center;
    gap: 8px;
}
div[data-testid="stSpinner"] > div::before {
    content: "🧠";
    font-size: 1.4rem;
    display: inline-block;
    animation: brain-dance 1s ease-in-out infinite;
}
@keyframes brain-dance {
    0%, 100% { transform: translateY(0) rotate(0deg) scale(1); }
    25% { transform: translateY(-6px) rotate(-10deg) scale(1.1); }
    50% { transform: translateY(0) rotate(0deg) scale(1); }
    75% { transform: translateY(-6px) rotate(10deg) scale(1.1); }
}

/* Hide default Streamlit spinner icon */
div[data-testid="stSpinner"] svg {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Type aliases
MemoryBankConfig = types.ReasoningEngineContextSpecMemoryBankConfig
SimilaritySearchConfig = types.ReasoningEngineContextSpecMemoryBankConfigSimilaritySearchConfig
GenerationConfig = types.ReasoningEngineContextSpecMemoryBankConfigGenerationConfig

# ── MODEL SELECTION (used by engine creation and customization) ──
EMBEDDING_MODELS = [
    "text-embedding-005",
    "text-embedding-004",
    "text-multilingual-embedding-002",
]
GENERATION_MODELS = [
    "gemini-3.1-pro",
    "gemini-3-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

SAMPLE_CONVERSATION = [
    {"role": "user", "message": "Hi! I'm Emma Chen. I have a reservation for a deluxe room for three nights."},
    {"role": "model", "message": "Welcome to our hotel, Ms. Chen! I have your reservation right here."},
    {"role": "user", "message": "I'm vegetarian and lactose intolerant, so please note that for room service."},
    {"role": "model", "message": "Absolutely, I've noted your dietary preferences. We have excellent vegetarian options."},
    {"role": "user", "message": "I prefer a high floor room, away from the elevator. I'm a light sleeper."},
    {"role": "model", "message": "I'll assign you a corner room on the 12th floor. It's one of our quietest locations."},
    {"role": "user", "message": "Can I get extra pillows and a white noise machine? I prefer the room at 68F."},
    {"role": "model", "message": "I'll have housekeeping bring extra pillows and a white noise machine, and set the thermostat to 68F."},
]

# --- MAIN UI ---
st.subheader("🧠 Memory Bank Playground")

tab1, tab2 = st.tabs(["Vertex AI Agent Engine", "Agent Development Kit"])

# ============================================================================
# TAB 1: Vertex AI Agent Engine (existing code, unchanged)
# ============================================================================
with tab1:
    st.caption("Demonstrates how to make API calls directly to Vertex AI Agent Engine Sessions and Memory Bank using the Vertex AI Agent Engine SDK.")

    # ── AGENT ENGINE: SELECT OR CREATE ──
    def _tab1_disconnect():
        st.session_state.mb_agent_engine_name = None
        st.session_state.mb_session_name = None
        st.session_state.mb_guest_id = None
        st.session_state.mb_conversation = []
        st.session_state.mb_event_count = 0
        st.session_state.pop("mb_engine_list", None)
        st.session_state.pop("mb_selected_topics", None)
        st.session_state.pop("mb_existing_memories", None)

    _render_engine_section("mb", "mb_agent_engine_name", _tab1_disconnect)

    # Only show remaining steps when engine exists
    if st.session_state.mb_agent_engine_name:
        ae_name = st.session_state.mb_agent_engine_name

        # ── EXISTING MEMORIES ──
        with st.expander("Existing Memories", expanded=False):
            col_uid, col_load = st.columns([3, 1])
            with col_uid:
                user_id = st.text_input("User ID", placeholder="Enter a user ID...", key="mb_user_id")
            with col_load:
                st.markdown("<br>", unsafe_allow_html=True)
                load_clicked = st.button("Load Memories", key="mb_load_existing", use_container_width=True)

            if load_clicked:
                st.session_state.pop("mb_existing_memories", None)

            if load_clicked and not user_id.strip():
                st.warning("Please enter a User ID to retrieve memories.")
            elif "mb_existing_memories" not in st.session_state and user_id.strip():
                with st.spinner("Retrieving existing memories..."):
                    try:
                        results = client.agent_engines.memories.retrieve(
                            name=ae_name,
                            scope={"user_id": user_id.strip()},
                        )
                        raw = list(results)
                        st.session_state.mb_existing_memories = [
                            item for item in raw
                            if (getattr(item.memory if hasattr(item, "memory") else item, "fact", None))
                        ]
                    except Exception as e:
                        st.error(f"Failed to retrieve memories: {e}")
                        st.session_state.mb_existing_memories = []

            existing = st.session_state.get("mb_existing_memories", [])
            if existing:
                st.info(f"Found {len(existing)} memories for user `{user_id}`")
                for i, mem_item in enumerate(existing, 1):
                    m = mem_item.memory if hasattr(mem_item, "memory") else mem_item
                    try:
                        full = client.agent_engines.memories.get(name=m.name)
                    except Exception:
                        full = m
                    ttl_info = _format_memory_ttl(full)
                    if ttl_info:
                        st.markdown(f"**{i}.** {full.fact}  \n`{ttl_info}`")
                    else:
                        st.markdown(f"**{i}.** {full.fact}")
                if st.button("Delete All Memories", key="mb_delete_all_memories"):
                    with st.spinner("Deleting all memories..."):
                        errors = 0
                        for mem_item in existing:
                            m = mem_item.memory if hasattr(mem_item, "memory") else mem_item
                            try:
                                client.agent_engines.memories.delete(name=m.name)
                            except Exception:
                                errors += 1
                        st.session_state.pop("mb_existing_memories", None)
                        if errors:
                            st.warning(f"Deleted with {errors} error(s). Reload to verify.")
                        else:
                            st.success("All memories deleted.")
                        st.rerun()
            elif "mb_existing_memories" in st.session_state and not existing:
                st.info(f"No memories found for user `{user_id}`. Start a conversation and generate memories to see them here.")

        # ── MEMORY CUSTOMIZATION ──
        with st.container(border=True):
            st.markdown("### 2. Memory Bank Customization")

            # --- Models ---
            st.markdown("**Models**")
            st.caption("Change the generation or embedding model for this engine.")
            col_em2, col_gm2 = st.columns(2)
            with col_em2:
                embedding_model = st.selectbox("Embedding Model", EMBEDDING_MODELS, index=0, key="mb_cfg_embedding")
            with col_gm2:
                generation_model = st.selectbox("Generation Model", GENERATION_MODELS, index=4, key="mb_cfg_generation")

            st.divider()

            # --- Topics ---
            st.markdown("**Memory Topics**")
            st.caption(
                "Choose which types of information Memory Bank should extract and persist. "
                "By default, all topics are active. Selecting a subset restricts extraction to only those topics."
            )

            MANAGED_TOPICS = {
                "USER_PERSONAL_INFO": "Personal information (names, relationships, hobbies, important dates)",
                "USER_PREFERENCES": "Preferences (likes, dislikes, preferred styles, patterns)",
                "KEY_CONVERSATION_DETAILS": "Key conversation details (milestones, conclusions, task outcomes)",
                "EXPLICIT_INSTRUCTIONS": "Explicit remember/forget instructions from the user",
            }

            if "mb_selected_topics" not in st.session_state:
                st.session_state.mb_selected_topics = list(MANAGED_TOPICS.keys())

            selected = []
            cols = st.columns(2)
            for i, (topic_key, topic_desc) in enumerate(MANAGED_TOPICS.items()):
                with cols[i % 2]:
                    if st.checkbox(
                        topic_desc,
                        value=topic_key in st.session_state.mb_selected_topics,
                        key=f"mb_topic_{topic_key}",
                    ):
                        selected.append(topic_key)

            st.divider()

            # --- TTL ---
            st.markdown("**Memory TTL**")
            st.caption("Set expiration times for memories. Memories are automatically deleted after the TTL expires.")

            ttl_mode = st.radio(
                "TTL Mode",
                ["None", "Default TTL", "Granular TTL"],
                horizontal=True,
                key="mb_ttl_mode",
            )

            TTL_UNITS = {"Seconds": 1, "Minutes": 60, "Hours": 3600}
            ttl_config = None

            if ttl_mode == "Default TTL":
                col_val, col_unit = st.columns([2, 1])
                with col_val:
                    default_val = st.number_input(
                        "TTL", min_value=1, value=60, key="mb_default_ttl_val",
                    )
                with col_unit:
                    default_unit = st.selectbox("Unit", list(TTL_UNITS.keys()), index=1, key="mb_default_ttl_unit")
                ttl_config = {"default_ttl": f"{default_val * TTL_UNITS[default_unit]}s"}

            elif ttl_mode == "Granular TTL":
                st.caption(
                    "Set TTL per operation type. Leave at 0 to skip (won't update expiration for that operation)."
                )
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    create_val = st.number_input("CreateMemory TTL", min_value=0, value=0, key="mb_create_ttl_val")
                    create_unit = st.selectbox("Unit", list(TTL_UNITS.keys()), index=1, key="mb_create_ttl_unit")
                with col_t2:
                    gen_created_val = st.number_input("GenerateMemories (new) TTL", min_value=0, value=5, key="mb_gen_created_ttl_val")
                    gen_created_unit = st.selectbox("Unit", list(TTL_UNITS.keys()), index=1, key="mb_gen_created_ttl_unit")
                with col_t3:
                    gen_updated_val = st.number_input("GenerateMemories (updated) TTL", min_value=0, value=0, key="mb_gen_updated_ttl_val")
                    gen_updated_unit = st.selectbox("Unit", list(TTL_UNITS.keys()), index=1, key="mb_gen_updated_ttl_unit")
                granular = {}
                if create_val > 0:
                    granular["create_ttl"] = f"{create_val * TTL_UNITS[create_unit]}s"
                if gen_created_val > 0:
                    granular["generate_created_ttl"] = f"{gen_created_val * TTL_UNITS[gen_created_unit]}s"
                if gen_updated_val > 0:
                    granular["generate_updated_ttl"] = f"{gen_updated_val * TTL_UNITS[gen_updated_unit]}s"
                if granular:
                    ttl_config = {"granular_ttl_config": granular}

            st.divider()

            # --- Single Apply button for both topics + TTL ---
            if st.button("Apply Configuration", key="mb_apply_config"):
                if not selected:
                    st.warning("Select at least one topic.")
                else:
                    with st.spinner("Updating Memory Bank configuration..."):
                        try:
                            memory_topics = [
                                {"managed_memory_topic": {"managed_topic_enum": t}}
                                for t in selected
                            ]
                            mb_config = {
                                "similarity_search_config": {
                                    "embedding_model": f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{embedding_model}"
                                },
                                "generation_config": {
                                    "model": f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{generation_model}"
                                },
                                "customization_configs": [
                                    {
                                        "scope_keys": ["user_id"],
                                        "memory_topics": memory_topics,
                                    }
                                ],
                            }
                            if ttl_config:
                                mb_config["ttl_config"] = ttl_config

                            client.agent_engines.update(
                                name=ae_name,
                                config={
                                    "context_spec": {
                                        "memory_bank_config": mb_config
                                    }
                                },
                            )
                            st.session_state.mb_selected_topics = selected
                            summary = f"Models: {generation_model}, {embedding_model}"
                            summary += f" | Topics: {len(selected)}"
                            if ttl_config:
                                summary += f" | TTL: {ttl_mode}"
                            st.success(f"Configuration applied — {summary}")
                        except Exception as e:
                            st.error(f"Failed to update configuration: {e}")

        # ── CREATE SESSION ──
        with st.container(border=True):
            st.markdown("### 3. Session")

            if st.session_state.mb_session_name:
                col_sess, col_reset = st.columns([4, 1])
                with col_sess:
                    st.success(f"Session active: `{st.session_state.mb_session_name}` (user: `{st.session_state.mb_guest_id}`)")
                with col_reset:
                    if st.button("New Session", key="mb_reset_session", use_container_width=True):
                        st.session_state.mb_session_name = None
                        st.session_state.mb_guest_id = None
                        st.session_state.mb_conversation = []
                        st.session_state.mb_event_count = 0
                        st.rerun()
            else:
                col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
                with col_s1:
                    guest_id_input = st.text_input(
                        "User ID", placeholder="Enter a user ID...", key="mb_guest_input"
                    )
                with col_s2:
                    session_display_name = st.text_input("Session Display Name", placeholder="Enter a session name...", key="mb_session_display")
                with col_s3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    create_session_clicked = st.button("Create Session", key="mb_create_session", use_container_width=True)

                if create_session_clicked:
                    with st.spinner("Creating session..."):
                        try:
                            session = client.agent_engines.sessions.create(
                                name=ae_name,
                                user_id=guest_id_input,
                                config={"display_name": f"{session_display_name} for {guest_id_input}"},
                            )
                            st.session_state.mb_session_name = session.response.name
                            st.session_state.mb_guest_id = guest_id_input
                            st.session_state.mb_conversation = []
                            st.session_state.mb_event_count = 0
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create session: {e}")

        # ── CHAT CONVERSATION ──
        if st.session_state.mb_session_name:

            def _append_event(role, text):
                """Append a single event to the Memory Bank session."""
                client.agent_engines.sessions.events.append(
                    name=st.session_state.mb_session_name,
                    author=st.session_state.mb_guest_id,
                    invocation_id=str(st.session_state.mb_event_count),
                    timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
                    config={
                        "content": {
                            "role": role,
                            "parts": [{"text": text}],
                        }
                    },
                )
                st.session_state.mb_event_count += 1

            with st.container(border=True):
                st.markdown(f"### 4. Chat — `{st.session_state.mb_guest_id}`")
                st.caption("Responses use existing memories for context. New memories are not created automatically — use Step 5 to generate them.")

                if st.button("Load Sample Conversation", key="mb_load_sample"):
                    with st.spinner("Appending sample conversation..."):
                        try:
                            for turn in SAMPLE_CONVERSATION:
                                _append_event(turn["role"], turn["message"])
                            st.session_state.mb_conversation.extend(SAMPLE_CONVERSATION)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load sample: {e}")

                # Chat message history
                chat_container = st.container(height=450)
                with chat_container:
                    for turn in st.session_state.mb_conversation:
                        role = "user" if turn["role"] == "user" else "assistant"
                        with st.chat_message(role):
                            st.markdown(turn["message"])

                # Chat input — using text_input + button so it stays inline
                def _submit_message():
                    st.session_state.mb_pending_message = st.session_state.mb_chat_input
                    st.session_state.mb_chat_input = ""

                col_input, col_send = st.columns([5, 1])
                with col_input:
                    st.text_input("Message", placeholder="Say something to the agent...", key="mb_chat_input", label_visibility="collapsed", on_change=_submit_message)
                with col_send:
                    send_clicked = st.button("Send", key="mb_send", use_container_width=True)

                if send_clicked and st.session_state.get("mb_chat_input", "").strip():
                    st.session_state.mb_pending_message = st.session_state.mb_chat_input
                    st.session_state.mb_chat_input = ""
                    st.rerun()

                prompt = st.session_state.pop("mb_pending_message", "")
                if prompt.strip():
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)

                    try:
                        _append_event("user", prompt)
                        st.session_state.mb_conversation.append({"role": "user", "message": prompt})
                    except Exception as e:
                        st.error(f"Failed to send user message: {e}")

                    # Generate model response with memory context
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    # Retrieve relevant memories for context
                                    memory_context = ""
                                    try:
                                        mem_results = client.agent_engines.memories.retrieve(
                                            name=ae_name,
                                            scope={"user_id": st.session_state.mb_guest_id},
                                            similarity_search_params={
                                                "search_query": prompt,
                                                "top_k": 5,
                                            },
                                        )
                                        memories = list(mem_results)
                                        if memories:
                                            facts = [
                                                (m.memory.fact if hasattr(m, "memory") else m.fact)
                                                for m in memories
                                            ]
                                            memory_context = "\n".join(f"- {f}" for f in facts)
                                    except Exception:
                                        pass

                                    model = GenerativeModel(generation_model)
                                    history_text = "\n".join(
                                        f"{t['role']}: {t['message']}" for t in st.session_state.mb_conversation
                                    )

                                    if memory_context:
                                        full_prompt = (
                                            f"You have the following memories about this user:\n{memory_context}\n\n"
                                            f"Conversation:\n{history_text}"
                                        )
                                    else:
                                        full_prompt = history_text

                                    response = model.generate_content(full_prompt)
                                    reply = response.text
                                    st.markdown(reply)

                                    _append_event("model", reply)
                                    st.session_state.mb_conversation.append({"role": "model", "message": reply})
                                except Exception as e:
                                    st.error(f"Model response failed: {e}")

            # ── GENERATE MEMORIES ──
            if st.session_state.mb_conversation:
                with st.container(border=True):
                    st.markdown(f"### 5. Generate — `{st.session_state.mb_guest_id}`")
                    st.caption("Extracts facts from the conversation (extract + consolidation).")

                    if st.button("Generate Memories", key="mb_generate"):
                        with st.spinner("Generating memories from conversation..."):
                            try:
                                events = [
                                    {
                                        "content": {
                                            "role": turn["role"],
                                            "parts": [{"text": turn["message"]}],
                                        }
                                    }
                                    for turn in st.session_state.mb_conversation
                                ]
                                operation = client.agent_engines.memories.generate(
                                    name=ae_name,
                                    scope={"user_id": st.session_state.mb_guest_id},
                                    direct_contents_source={"events": events},
                                    config={"wait_for_completion": True},
                                )
                                if operation.response and operation.response.generated_memories:
                                    st.success(f"Generated {len(operation.response.generated_memories)} memories")
                                    for i, gen_mem in enumerate(operation.response.generated_memories, 1):
                                        if gen_mem.action != "DELETED" and gen_mem.memory:
                                            try:
                                                full = client.agent_engines.memories.get(name=gen_mem.memory.name)
                                                action_label = "NEW" if gen_mem.action == "CREATED" else "UPDATED"
                                                ttl_info = _format_memory_ttl(full)
                                                if ttl_info:
                                                    st.markdown(f"**{i}. [{action_label}]** {full.fact}  \n`{ttl_info}`")
                                                else:
                                                    st.markdown(f"**{i}. [{action_label}]** {full.fact}")
                                            except Exception:
                                                st.markdown(f"**{i}.** _(could not retrieve)_")
                                else:
                                    st.info("No memories generated from this conversation.")
                            except Exception as e:
                                st.error(f"Generation failed: {e}")

                # ── RETRIEVE MEMORIES ──
                with st.container(border=True):
                    st.markdown(f"### 6. Retrieve — `{st.session_state.mb_guest_id}`")
                    retrieval_method = st.radio(
                        "Retrieval Method",
                        ["Scope-based (all memories)", "Similarity search"],
                        key="mb_retrieval_method",
                        horizontal=True,
                    )

                    col_r1, col_r2 = st.columns(2)
                    search_query = ""
                    top_k = 3
                    if retrieval_method == "Similarity search":
                        with col_r1:
                            search_query = st.text_input("Search Query", placeholder="Enter a search query...", key="mb_search_query")
                        with col_r2:
                            top_k = st.number_input("Top K", min_value=1, max_value=20, value=3, key="mb_top_k")

                    if st.button("Retrieve", key="mb_retrieve"):
                        with st.spinner("Retrieving memories..."):
                            try:
                                retrieve_kwargs = {
                                    "name": ae_name,
                                    "scope": {"user_id": st.session_state.mb_guest_id},
                                }
                                if retrieval_method == "Similarity search":
                                    retrieve_kwargs["similarity_search_params"] = {
                                        "search_query": search_query,
                                        "top_k": top_k,
                                    }
                                results = client.agent_engines.memories.retrieve(**retrieve_kwargs)
                                memories = list(results)
                                if memories:
                                    st.success(f"Retrieved {len(memories)} memories")
                                    is_similarity = retrieval_method == "Similarity search"
                                    for i, mem_item in enumerate(memories, 1):
                                        m = mem_item.memory if hasattr(mem_item, "memory") else mem_item
                                        score = getattr(mem_item, "similarity", None) or getattr(mem_item, "score", None) or getattr(mem_item, "distance", None)
                                        score_label = f" (score: {score:.4f})" if score is not None else ""
                                        header = f"{i}. {m.fact[:80]}...{score_label}" if is_similarity else f"{i}. {m.fact[:80]}..."
                                        with st.expander(header):
                                            st.write(f"**Fact:** {m.fact}")
                                            if score is not None:
                                                st.write(f"**Similarity Score:** {score:.4f}")
                                            ttl_info = _format_memory_ttl(m)
                                            if ttl_info:
                                                st.write(f"**TTL:** {ttl_info}")
                                            st.code(f"ID: {m.name}")
                                else:
                                    st.info("No memories found.")
                            except Exception as e:
                                st.error(f"Retrieval failed: {e}")

        # ── CLEANUP ──
        st.divider()
        with st.expander("Cleanup"):
            if st.button("Delete Agent Engine", key="mb_delete_engine"):
                with st.spinner("Deleting Agent Engine..."):
                    try:
                        client.agent_engines.delete(name=ae_name, force=True)
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                    else:
                        st.session_state.mb_agent_engine_name = None
                        st.session_state.mb_session_name = None
                        st.session_state.mb_guest_id = None
                        st.session_state.mb_conversation = []
                        st.session_state.mb_event_count = 0
                        st.session_state.pop("mb_engine_list", None)
                        st.session_state.pop("mb_selected_topics", None)
                        st.session_state.pop("mb_existing_memories", None)
                        st.success("Agent Engine deleted.")
                        st.rerun()


# ============================================================================
# TAB 2: Agent Development Kit
# ============================================================================
with tab2:
    st.caption("Demonstrates how you can use Memory Bank with ADK to manage long-term memories. After you configure your Agent Development Kit (ADK) agent to use Memory Bank, your agent orchestrates calls to Memory Bank to manage long-term memories for you.")

    # ── 1. AGENT ENGINE ──
    def _tab2_disconnect():
        st.session_state.mb_adk_engine_name = None
        st.session_state.mb_adk_runner = None
        st.session_state.mb_adk_config_hash = None
        st.session_state.mb_adk_session_id = None
        st.session_state.mb_adk_conversation = []
        st.session_state.mb_adk_turn_debug = []
        st.session_state.mb_adk_session_service = None
        st.session_state.mb_adk_memory_service = None
        st.session_state.pop("mb_engine_list", None)
        st.session_state.pop("mb_adk_existing_memories", None)

    _render_engine_section("mb_adk", "mb_adk_engine_name", _tab2_disconnect)

    # Only show remaining sections when engine is connected
    adk_engine_name = st.session_state.mb_adk_engine_name
    if adk_engine_name:
        adk_engine_id = adk_engine_name.split("/")[-1]

        # ── EXISTING MEMORIES ──
        with st.expander("Existing Memories", expanded=False):
            st.caption(
                "Check what memories exist for a given scope before chatting.")
            col_uid_em, col_app_em = st.columns(2)
            with col_uid_em:
                em_user_id = st.text_input(
                    "User ID", placeholder="Enter a user ID...", key="mb_adk_em_user_id")
            with col_app_em:
                em_app_name = st.text_input(
                    "App Name", value=st.session_state.mb_adk_app_name, key="mb_adk_em_app_name")

            col_load_em, col_scope_label = st.columns([1, 3])
            with col_load_em:
                load_em_clicked = st.button(
                    "Load Memories", key="mb_adk_load_existing", use_container_width=True)
            with col_scope_label:
                scope_preview = {"user_id": em_user_id or "...",
                                 "app_name": em_app_name or "..."}
                st.caption(f"Scope: `{scope_preview}`")

            if load_em_clicked:
                st.session_state.pop("mb_adk_existing_memories", None)

            if load_em_clicked and (not em_user_id.strip() or not em_app_name.strip()):
                st.warning("Please enter both User ID and App Name.")
            elif "mb_adk_existing_memories" not in st.session_state and em_user_id.strip() and em_app_name.strip() and load_em_clicked:
                scope = {"user_id": em_user_id.strip(
                ), "app_name": em_app_name.strip()}
                with st.spinner("Retrieving existing memories..."):
                    try:
                        results = client.agent_engines.memories.retrieve(
                            name=adk_engine_name, scope=scope,
                        )
                        raw = list(results)
                        st.session_state.mb_adk_existing_memories = [
                            item for item in raw
                            if (getattr(item.memory if hasattr(item, "memory") else item, "fact", None))
                        ]
                        st.session_state._adk_em_scope = scope
                    except Exception as e:
                        st.error(f"Failed to retrieve memories: {e}")
                        st.session_state.mb_adk_existing_memories = []

            existing = st.session_state.get("mb_adk_existing_memories", [])
            if existing:
                em_scope = st.session_state.get("_adk_em_scope", {})
                st.info(
                    f"Found {len(existing)} memories for scope `{em_scope}`")
                for i, mem_item in enumerate(existing, 1):
                    m = mem_item.memory if hasattr(
                        mem_item, "memory") else mem_item
                    try:
                        full = client.agent_engines.memories.get(name=m.name)
                    except Exception:
                        full = m
                    ttl_info = _format_memory_ttl(full)
                    if ttl_info:
                        st.markdown(f"**{i}.** {full.fact}  \n`{ttl_info}`")
                    else:
                        st.markdown(f"**{i}.** {full.fact}")
                if st.button("Delete All Memories", key="mb_adk_delete_all_memories"):
                    with st.spinner("Deleting all memories..."):
                        errors = 0
                        for mem_item in existing:
                            m = mem_item.memory if hasattr(
                                mem_item, "memory") else mem_item
                            try:
                                client.agent_engines.memories.delete(
                                    name=m.name)
                            except Exception:
                                errors += 1
                        st.session_state.pop("mb_adk_existing_memories", None)
                        if errors:
                            st.warning(
                                f"Deleted with {errors} error(s). Reload to verify.")
                        else:
                            st.success("All memories deleted.")
                        st.rerun()
            elif "mb_adk_existing_memories" in st.session_state and not existing:
                em_scope = st.session_state.get("_adk_em_scope", {})
                st.info(f"No memories found for scope `{em_scope}`.")

        # ── 2. AGENT CONFIGURATION ──
        with st.container(border=True):
            st.markdown("### 2. Agent Configuration")

            col_model, col_name = st.columns(2)
            with col_model:
                adk_model = st.selectbox(
                    "Model", GENERATION_MODELS,
                    index=GENERATION_MODELS.index(st.session_state.mb_adk_model)
                    if st.session_state.mb_adk_model in GENERATION_MODELS else 4,
                    key="mb_adk_model_select",
                )
            with col_name:
                adk_agent_name = st.text_input(
                    "Agent Name", value=st.session_state.mb_adk_agent_name,
                    key="mb_adk_agent_name_input",
                )

            adk_instruction = st.text_area(
                "System Instruction",
                value=st.session_state.mb_adk_instruction,
                height=100,
                key="mb_adk_instruction_input",
            )

        # ── 3. MEMORY CONFIGURATION ──
        with st.container(border=True):
            st.markdown("### 3. Memory Configuration")

            col_ret, col_gen = st.columns(2)
            with col_ret:
                st.markdown("**Retrieval Strategy**")
                adk_retrieval = st.radio(
                    "How the agent retrieves memories",
                    [
                        "Preload — `PreloadMemoryTool`",
                        "Tool-based — `LoadMemoryTool`",
                        "Custom callback",
                        "None",
                    ],
                    index=["Preload — `PreloadMemoryTool`", "Tool-based — `LoadMemoryTool`", "Custom callback", "None"].index(
                        st.session_state.mb_adk_retrieval
                    ),
                    key="mb_adk_retrieval_radio",
                    label_visibility="collapsed",
                )

            with col_gen:
                st.markdown("**Auto-Generate Memories**")
                adk_auto_gen = st.radio(
                    "When to auto-generate memories",
                    [
                        "Off",
                        "After each turn (full session)",
                        "After each turn (last message only)",
                    ],
                    index=["Off", "After each turn (full session)", "After each turn (last message only)"].index(
                        st.session_state.mb_adk_auto_gen
                    ),
                    key="mb_adk_auto_gen_radio",
                    label_visibility="collapsed",
                )

            st.divider()
            st.markdown("**Scope Configuration**")
            col_scope1, col_scope2 = st.columns(2)
            with col_scope1:
                st.checkbox("user_id (always on)", value=True,
                            disabled=True, key="mb_adk_scope_uid")
                st.checkbox("app_name (always on)", value=True,
                            disabled=True, key="mb_adk_scope_app")
            with col_scope2:
                adk_app_name = st.text_input(
                    "App Name value",
                    value=st.session_state.mb_adk_app_name,
                    key="mb_adk_app_name_input",
                )

        # ── 4. CALLBACKS (expandable summary) ──
        with st.expander("4. Active Callbacks"):
            cb_info = []
            if adk_retrieval == "Custom callback":
                cb_info.append("`before_model_callback` — retrieves memories via Agent Engine SDK and injects into system instruction")
            if adk_retrieval in ("Preload — `PreloadMemoryTool`", "Tool-based — `LoadMemoryTool`"):
                cb_info.append(
                    "`memory_service` — `VertexAiMemoryBankService`")
            if adk_auto_gen == "After each turn (full session)":
                cb_info.append("`after_agent_callback` — sends full session to `add_session_to_memory`")
            elif adk_auto_gen == "After each turn (last message only)":
                cb_info.append("`before_agent_callback` — sends latest user message via `direct_contents_source`")
            cb_info.append("`after_tool_callback` — logs all tool calls for transparency (always active)")

            if adk_retrieval == "None" and adk_auto_gen == "Off":
                st.info("Baseline mode — no memory retrieval or generation. Only tool logging is active.")
            for line in cb_info:
                st.markdown(f"- {line}")

        # ── 5. SESSION & CHAT ──
        with st.container(border=True):
            st.markdown("### 5. Session & Chat")

            col_uid, col_actions = st.columns([3, 2])
            with col_uid:
                adk_user_id = st.text_input(
                    "User ID", value=st.session_state.mb_adk_user_id,
                    placeholder="Enter a user ID...",
                    key="mb_adk_user_id_input",
                )
            with col_actions:
                st.markdown("<br>", unsafe_allow_html=True)
                col_build, col_new = st.columns(2)
                with col_new:
                    new_session_clicked = st.button("New Session", key="mb_adk_new_session", use_container_width=True)
                with col_build:
                    build_clicked = st.button("Build Agent", key="mb_adk_build", use_container_width=True)

            if new_session_clicked:
                if st.session_state.mb_adk_session_service and adk_user_id.strip():
                    try:
                        app = st.session_state.mb_adk_app_name
                        session = _run_async(
                            st.session_state.mb_adk_session_service.create_session(
                                app_name=app, user_id=adk_user_id,
                            )
                        )
                        st.session_state.mb_adk_session_id = session.id
                    except Exception as e:
                        st.error(f"Failed to create new session: {e}")
                else:
                    st.session_state.mb_adk_session_id = None
                st.session_state.mb_adk_conversation = []
                st.session_state.mb_adk_turn_debug = []
                st.rerun()

            # Compute config hash to detect changes
            current_config = (
                adk_engine_name, adk_model, adk_agent_name, adk_instruction,
                adk_retrieval, adk_auto_gen, adk_app_name, adk_user_id,
            )
            config_hash = hash(current_config)

            if build_clicked or (st.session_state.mb_adk_runner is not None and config_hash != st.session_state.mb_adk_config_hash):
                if not adk_user_id.strip():
                    st.warning("Please enter a User ID before building the agent.")
                else:
                    with st.spinner("Building ADK agent..."):
                        try:
                            # Save current config values
                            st.session_state.mb_adk_model = adk_model
                            st.session_state.mb_adk_agent_name = adk_agent_name
                            st.session_state.mb_adk_instruction = adk_instruction
                            st.session_state.mb_adk_retrieval = adk_retrieval
                            st.session_state.mb_adk_auto_gen = adk_auto_gen
                            st.session_state.mb_adk_scope_keys = [
                                "user_id", "app_name"]
                            st.session_state.mb_adk_app_name = adk_app_name
                            st.session_state.mb_adk_user_id = adk_user_id

                            # Build scope
                            scope_keys = ["user_id", "app_name"]

                            # Memory service (for Preload/Tool-based)
                            memory_service = None
                            if adk_retrieval in ("Preload — `PreloadMemoryTool`", "Tool-based — `LoadMemoryTool`"):
                                memory_service = VertexAiMemoryBankService(
                                    agent_engine_id=adk_engine_id,
                                    project=PROJECT_ID,
                                    location=LOCATION,
                                )
                                st.session_state.mb_adk_memory_service = memory_service

                            # Tools
                            tools = []
                            if adk_retrieval == "Preload — `PreloadMemoryTool`":
                                tools.append(PreloadMemoryTool())
                            elif adk_retrieval == "Tool-based — `LoadMemoryTool`":
                                tools.append(LoadMemoryTool())

                            # Mutable debug container for callbacks
                            turn_debug = {"system_instruction": "", "memories": [], "tool_calls": [], "auto_gen": False}
                            st.session_state._adk_turn_debug_ref = turn_debug

                            # Callbacks
                            before_model_cb = None
                            before_agent_cb = None
                            after_agent_cb = None

                            # Custom retrieval callback
                            if adk_retrieval == "Custom callback":
                                _engine_name = adk_engine_name
                                _scope_keys = scope_keys
                                _app_name = adk_app_name

                                def retrieve_memories_callback(callback_context, llm_request):
                                    uid = callback_context._invocation_context.user_id
                                    scope = {"user_id": uid}
                                    if "app_name" in _scope_keys:
                                        scope["app_name"] = _app_name
                                    response = client.agent_engines.memories.retrieve(
                                        name=_engine_name, scope=scope,
                                    )
                                    mem_list = list(response)
                                    facts = [f"* {m.memory.fact}" for m in mem_list if hasattr(m, "memory")]
                                    debug = st.session_state.get("_adk_turn_debug_ref", {})
                                    debug["memories"] = facts
                                    debug["system_instruction"] = str(llm_request.config.system_instruction or "")
                                    if facts:
                                        llm_request.config.system_instruction = (
                                            (llm_request.config.system_instruction or "")
                                            + "\nHere is information that you have about the user:\n"
                                            + "\n".join(facts)
                                        )
                                        debug["system_instruction"] = str(llm_request.config.system_instruction)

                                before_model_cb = retrieve_memories_callback
                            else:
                                # Logging-only before_model callback to capture system instruction
                                def log_system_instruction(callback_context, llm_request):
                                    debug = st.session_state.get("_adk_turn_debug_ref", {})
                                    debug["system_instruction"] = str(llm_request.config.system_instruction or "")

                                before_model_cb = log_system_instruction

                            # Auto-generate: full session
                            if adk_auto_gen == "After each turn (full session)":
                                _mem_svc = memory_service or VertexAiMemoryBankService(
                                    agent_engine_id=adk_engine_id,
                                    project=PROJECT_ID,
                                    location=LOCATION,
                                )

                                async def auto_save_full_session(callback_context):
                                    debug = st.session_state.get("_adk_turn_debug_ref", {})
                                    try:
                                        await _mem_svc.add_session_to_memory(
                                            callback_context._invocation_context.session
                                        )
                                        debug["auto_gen"] = True
                                    except Exception:
                                        debug["auto_gen"] = False

                                after_agent_cb = auto_save_full_session

                            # Auto-generate: last message only
                            elif adk_auto_gen == "After each turn (last message only)":
                                _engine_name_gen = adk_engine_name
                                _scope_keys_gen = scope_keys
                                _app_name_gen = adk_app_name

                                async def auto_save_last_message(callback_context):
                                    debug = st.session_state.get("_adk_turn_debug_ref", {})
                                    try:
                                        last_turn = callback_context._invocation_context.user_content
                                        uid = callback_context._invocation_context.user_id
                                        scope = {"user_id": uid}
                                        if "app_name" in _scope_keys_gen:
                                            scope["app_name"] = _app_name_gen
                                        client.agent_engines.memories.generate(
                                            name=_engine_name_gen,
                                            scope=scope,
                                            direct_contents_source={"events": [{"content": last_turn}]},
                                            config={"wait_for_completion": False},
                                        )
                                        debug["auto_gen"] = True
                                    except Exception:
                                        debug["auto_gen"] = False

                                before_agent_cb = auto_save_last_message

                            # Tool logging callback (always active)
                            def log_tool_call(tool, args, tool_response, **kwargs):
                                debug = st.session_state.get("_adk_turn_debug_ref", {})
                                debug.setdefault("tool_calls", []).append({
                                    "tool": str(tool),
                                    "args": str(args),
                                    "response": str(tool_response)[:500],
                                })

                            # Create agent
                            agent = LlmAgent(
                                model=adk_model,
                                name=adk_agent_name,
                                instruction=adk_instruction,
                                tools=tools,
                                before_model_callback=before_model_cb,
                                before_agent_callback=before_agent_cb,
                                after_agent_callback=after_agent_cb,
                                after_tool_callback=log_tool_call,
                            )

                            # Session service
                            session_service = InMemorySessionService()
                            st.session_state.mb_adk_session_service = session_service

                            # Runner
                            runner = Runner(
                                agent=agent,
                                app_name=adk_app_name,
                                session_service=session_service,
                                memory_service=memory_service,
                            )
                            st.session_state.mb_adk_runner = runner
                            st.session_state.mb_adk_config_hash = config_hash

                            # Create initial session
                            session = _run_async(
                                session_service.create_session(
                                    app_name=adk_app_name,
                                    user_id=adk_user_id,
                                )
                            )
                            st.session_state.mb_adk_session_id = session.id
                            st.session_state.mb_adk_conversation = []
                            st.session_state.mb_adk_turn_debug = []

                            st.success(f"Agent built: **{adk_agent_name}** ({adk_model}) | Session: `{session.id}`")
                        except Exception as e:
                            st.error(f"Failed to build agent: {e}")
                            st.session_state.mb_adk_runner = None

            # Show current agent status
            if st.session_state.mb_adk_runner and st.session_state.mb_adk_session_id:
                st.info(
                    f"Agent: **{st.session_state.mb_adk_agent_name}** | "
                    f"Model: {st.session_state.mb_adk_model} | "
                    f"Retrieval: {st.session_state.mb_adk_retrieval} | "
                    f"Auto-gen: {st.session_state.mb_adk_auto_gen}"
                )

                if st.button("Load Sample Conversation", key="mb_adk_load_sample"):
                    user_turns = [t for t in SAMPLE_CONVERSATION if t["role"] == "user"]
                    total = len(user_turns)
                    progress = st.progress(0, text=f"Processing turn 1/{total} through agent...")
                    try:
                        runner = st.session_state.mb_adk_runner
                        user_idx = 0
                        for turn in SAMPLE_CONVERSATION:
                            if turn["role"] == "user":
                                user_idx += 1
                                progress.progress(
                                    user_idx / total,
                                    text=f"Processing turn {user_idx}/{total}: \"{turn['message'][:50]}...\"",
                                )
                                content = Content(
                                    role="user",
                                    parts=[Part(text=turn["message"])],
                                )
                                async def _run_sample_turn():
                                    async for event in runner.run_async(
                                        user_id=adk_user_id,
                                        session_id=st.session_state.mb_adk_session_id,
                                        new_message=content,
                                    ):
                                        pass
                                _run_async(_run_sample_turn())
                            st.session_state.mb_adk_conversation.append(turn)
                        progress.progress(1.0, text="Sample conversation loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load sample: {e}")

                # Chat interface
                adk_chat_container = st.container(height=450)
                with adk_chat_container:
                    for i, turn in enumerate(st.session_state.mb_adk_conversation):
                        role = "user" if turn["role"] == "user" else "assistant"
                        with st.chat_message(role):
                            st.markdown(turn["message"])

                # Chat input
                def _submit_adk_message():
                    st.session_state.mb_adk_pending_message = st.session_state.mb_adk_chat_input
                    st.session_state.mb_adk_chat_input = ""

                col_input, col_send = st.columns([5, 1])
                with col_input:
                    st.text_input(
                        "Message", placeholder="Say something to the ADK agent...",
                        key="mb_adk_chat_input", label_visibility="collapsed",
                        on_change=_submit_adk_message,
                    )
                with col_send:
                    adk_send_clicked = st.button("Send", key="mb_adk_send", use_container_width=True)

                if adk_send_clicked and st.session_state.get("mb_adk_chat_input", "").strip():
                    st.session_state.mb_adk_pending_message = st.session_state.mb_adk_chat_input
                    st.session_state.mb_adk_chat_input = ""
                    st.rerun()

                adk_prompt = st.session_state.pop("mb_adk_pending_message", "")
                if adk_prompt.strip():
                    # Show user message immediately
                    with adk_chat_container:
                        with st.chat_message("user"):
                            st.markdown(adk_prompt)

                    st.session_state.mb_adk_conversation.append({"role": "user", "message": adk_prompt})

                    # Reset turn debug for this turn
                    turn_debug = {"system_instruction": "", "memories": [], "tool_calls": [], "auto_gen": False}
                    st.session_state._adk_turn_debug_ref = turn_debug

                    # Run agent
                    with adk_chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    runner = st.session_state.mb_adk_runner
                                    content = Content(role="user", parts=[Part(text=adk_prompt)])

                                    result = {"response": ""}

                                    async def _collect_events():
                                        async for event in runner.run_async(
                                            user_id=adk_user_id,
                                            session_id=st.session_state.mb_adk_session_id,
                                            new_message=content,
                                        ):
                                            if hasattr(event, "actions") and event.actions:
                                                if hasattr(event.actions, "tool_calls") and event.actions.tool_calls:
                                                    for tc in event.actions.tool_calls:
                                                        turn_debug.setdefault("tool_calls", []).append({
                                                            "tool": str(getattr(tc, "name", getattr(tc, "tool_name", tc))),
                                                            "args": str(getattr(tc, "args", getattr(tc, "arguments", ""))),
                                                            "response": "",
                                                        })
                                            if hasattr(event, "tool_response") and event.tool_response:
                                                existing_calls = turn_debug.get("tool_calls", [])
                                                if existing_calls:
                                                    existing_calls[-1]["response"] = str(event.tool_response)[:500]
                                            if event.is_final_response() and event.content and event.content.parts:
                                                for part in event.content.parts:
                                                    if hasattr(part, "text") and part.text:
                                                        result["response"] = part.text
                                                        break

                                    _run_async(_collect_events())
                                    final_response = result["response"]

                                    if final_response:
                                        st.markdown(final_response)
                                        st.session_state.mb_adk_conversation.append(
                                            {"role": "assistant", "message": final_response}
                                        )
                                        # Store debug info aligned with this assistant message
                                        # Find the index of this assistant message in conversation
                                        assistant_idx = len(st.session_state.mb_adk_conversation) - 1
                                        # Pad debug list if needed
                                        while len(st.session_state.mb_adk_turn_debug) < assistant_idx:
                                            st.session_state.mb_adk_turn_debug.append(None)
                                        st.session_state.mb_adk_turn_debug.append(dict(turn_debug))
                                    else:
                                        st.warning("No response from agent.")
                                except Exception as e:
                                    st.error(f"Agent error: {e}")

                # Transparency panel — show last turn's debug info below chat
                if st.session_state.mb_adk_turn_debug:
                    last_debug = st.session_state.mb_adk_turn_debug[-1]
                    if last_debug and any(last_debug.get(k) for k in ("system_instruction", "memories", "tool_calls", "auto_gen")):
                        with st.expander("Last Turn Details", expanded=False):
                            if last_debug.get("system_instruction"):
                                st.markdown(
                                    "**System Instruction Sent to Model:**")
                                st.code(
                                    last_debug["system_instruction"], language=None)
                            if last_debug.get("memories"):
                                st.markdown("**Memories Retrieved:**")
                                for mem in last_debug["memories"]:
                                    st.markdown(mem)
                            if last_debug.get("tool_calls"):
                                st.markdown("**Tool Calls:**")
                                for tc in last_debug["tool_calls"]:
                                    st.markdown(f"- **{tc['tool']}**({tc['args']})")
                                    if tc.get("response"):
                                        st.code(tc["response"], language=None)
                            if last_debug.get("auto_gen"):
                                st.markdown("**Auto-generate:** Triggered")

            elif not st.session_state.mb_adk_runner:
                st.info("Configure the agent above and click **Build Agent** to start chatting.")
