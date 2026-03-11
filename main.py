import streamlit as st
import uuid
import datetime
import vertexai
from vertexai import types
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv
import os

load_dotenv()

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

st.set_page_config(page_title="Vertex AI Memory Bank", layout="wide")

if not PROJECT_ID or not LOCATION:
    st.title("Configuration Required")
    st.error("Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in your `.env` file.")
    st.stop()

# Initialize Vertex AI
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

# --- SESSION STATE INITIALIZATION ---
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



# --- CUSTOM STYLING ---
st.markdown("""
<style>
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(240, 242, 246, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(200, 205, 215, 0.4);
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)

# --- MAIN UI: MEMORY BANK ---
st.subheader("Memory Bank Feature Tests")
st.caption("Create a Memory Bank config, session, add conversation, generate & retrieve memories.")

# Type aliases
MemoryBankConfig = types.ReasoningEngineContextSpecMemoryBankConfig
SimilaritySearchConfig = types.ReasoningEngineContextSpecMemoryBankConfigSimilaritySearchConfig
GenerationConfig = types.ReasoningEngineContextSpecMemoryBankConfigGenerationConfig

# ── CONFIG SECTION ──
with st.container(border=True):
    st.markdown("### 1. Memory Bank Configuration")
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "text-embedding-005",
                "text-embedding-004",
                "text-multilingual-embedding-002",
            ],
            index=0,
        )
    with col_cfg2:
        generation_model = st.selectbox(
            "Generation Model",
            [
                "gemini-3.1-pro",
                "gemini-3-flash",
                "gemini-3.1-flash-lite",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
            ],
            index=4,  # default to gemini-2.5-flash
        )

# ── AGENT ENGINE: SELECT OR CREATE ──
step2 = st.container(border=True)
with step2:
    st.markdown("### 2. Agent Engine")

    if st.session_state.mb_agent_engine_name:
        st.success(f"Agent Engine active: `{st.session_state.mb_agent_engine_name}`")
        if st.button("Disconnect", key="mb_disconnect_engine"):
            st.session_state.mb_agent_engine_name = None
            st.session_state.mb_session_name = None
            st.session_state.mb_guest_id = None
            st.session_state.mb_conversation = []
            st.session_state.mb_event_count = 0
            st.session_state.pop("mb_engine_list", None)
            st.rerun()
    else:
        engine_mode = st.radio(
            "Engine Source",
            ["Select existing", "Create new"],
            horizontal=True,
            key="mb_engine_mode",
        )

        if engine_mode == "Select existing":
            if st.button("Refresh list", key="mb_refresh_engines"):
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
                    selected_label = st.selectbox("Available Engines", list(engine_options.keys()), key="mb_engine_select")
                with col_use:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Use this Engine", key="mb_use_engine", use_container_width=True):
                        st.session_state.mb_agent_engine_name = engine_options[selected_label]
                        st.session_state.pop("mb_engine_list", None)
                        st.rerun()
            else:
                st.info("No agent engines found. Create a new one.")

        else:
            engine_display_name = st.text_input("Engine Name", value="memory-bank-engine", key="mb_engine_name")
            if st.button("Create Agent Engine", key="mb_create_engine"):
                with st.spinner("Provisioning Agent Engine (this may take a few minutes)..."):
                    try:
                        mb_config = MemoryBankConfig(
                            similarity_search_config=SimilaritySearchConfig(
                                embedding_model=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{embedding_model}"
                            ),
                            generation_config=GenerationConfig(
                                model=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{generation_model}"
                            ),
                        )
                        ae = client.agent_engines.create(
                            config={
                                "display_name": engine_display_name,
                                "context_spec": {"memory_bank_config": mb_config},
                            }
                        )
                        st.session_state.mb_agent_engine_name = ae.api_resource.name
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create Agent Engine: {e}")

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
                    st.session_state.mb_existing_memories = list(results)
                except Exception as e:
                    st.error(f"Failed to retrieve memories: {e}")
                    st.session_state.mb_existing_memories = []

        existing = st.session_state.get("mb_existing_memories", [])
        if existing:
            st.info(f"Found {len(existing)} memories for user `{user_id}`")
            for i, mem_item in enumerate(existing, 1):
                m = mem_item.memory if hasattr(mem_item, "memory") else mem_item
                st.markdown(f"**{i}.** {m.fact}")
            if st.button("Delete All Memories", key="mb_delete_all_memories", type="secondary"):
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
        elif existing is not None and user_id.strip() and not load_clicked:
            st.info(f"No existing memories found for user `{user_id}`.")

    # ── CREATE SESSION ──
    with st.container(border=True):
        st.markdown("### 3. Create Session")

        if st.session_state.mb_session_name:
            st.success(f"Session active: `{st.session_state.mb_session_name}` (user: `{st.session_state.mb_guest_id}`)")
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
            st.markdown(f"### 4. Chat Conversation — `{st.session_state.mb_guest_id}`")
            st.caption("Responses use existing memories for context. New memories are not created automatically — use Step 5 to generate them.")

            if st.button("Load Sample Conversation", key="mb_load_sample"):
                sample = [
                    {"role": "user", "message": "Hi! I'm Emma Chen. I have a reservation for a deluxe room for three nights."},
                    {"role": "model", "message": "Welcome to our hotel, Ms. Chen! I have your reservation right here."},
                    {"role": "user", "message": "I'm vegetarian and lactose intolerant, so please note that for room service."},
                    {"role": "model", "message": "Absolutely, I've noted your dietary preferences. We have excellent vegetarian options."},
                    {"role": "user", "message": "I prefer a high floor room, away from the elevator. I'm a light sleeper."},
                    {"role": "model", "message": "I'll assign you a corner room on the 12th floor. It's one of our quietest locations."},
                    {"role": "user", "message": "Can I get extra pillows and a white noise machine? I prefer the room at 68F."},
                    {"role": "model", "message": "I'll have housekeeping bring extra pillows and a white noise machine, and set the thermostat to 68F."},
                ]
                with st.spinner("Appending sample conversation..."):
                    try:
                        for turn in sample:
                            _append_event(turn["role"], turn["message"])
                        st.session_state.mb_conversation.extend(sample)
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
                if st.button("Send", key="mb_send", use_container_width=True):
                    _submit_message()

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
                st.markdown(f"### 5. Generate Memories — `{st.session_state.mb_guest_id}`")
                st.caption("Extracts facts from the conversation (extract + consolidation).")

                if st.button("Generate Memories", key="mb_generate"):
                    with st.spinner("Generating memories from conversation..."):
                        try:
                            operation = client.agent_engines.memories.generate(
                                name=ae_name,
                                vertex_session_source={"session": st.session_state.mb_session_name},
                                config={"wait_for_completion": True},
                            )
                            if operation.response and operation.response.generated_memories:
                                st.success(f"Generated {len(operation.response.generated_memories)} memories")
                                for i, gen_mem in enumerate(operation.response.generated_memories, 1):
                                    if gen_mem.action != "DELETED" and gen_mem.memory:
                                        try:
                                            full = client.agent_engines.memories.get(name=gen_mem.memory.name)
                                            action_label = "NEW" if gen_mem.action == "CREATED" else "UPDATED"
                                            st.markdown(f"**{i}. [{action_label}]** {full.fact}")
                                        except Exception:
                                            st.markdown(f"**{i}.** _(could not retrieve)_")
                            else:
                                st.info("No memories generated from this conversation.")
                        except Exception as e:
                            st.error(f"Generation failed: {e}")

            # ── RETRIEVE MEMORIES ──
            with st.container(border=True):
                st.markdown(f"### 6. Retrieve Memories — `{st.session_state.mb_guest_id}`")
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
                                        st.code(f"ID: {m.name}")
                            else:
                                st.info("No memories found.")
                        except Exception as e:
                            st.error(f"Retrieval failed: {e}")

    # ── CLEANUP ──
    st.divider()
    with st.expander("Cleanup"):
        if st.button("Delete Agent Engine", key="mb_delete_engine", type="secondary"):
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
                st.success("Agent Engine deleted.")
                st.rerun()
