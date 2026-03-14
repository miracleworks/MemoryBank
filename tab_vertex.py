import streamlit as st
import datetime
import uuid
from vertexai.generative_models import GenerativeModel

from shared import (
    PROJECT_ID, LOCATION, EMBEDDING_MODELS, GENERATION_MODELS,
    SAMPLE_CONVERSATION, MemoryBankConfig, SimilaritySearchConfig,
    GenerationConfig, _render_engine_section, _format_memory_ttl,
    _populate_engine_config,
)


def render(client):
    """Render Tab 1: Vertex AI Agent Engine."""

    # ── SESSION STATE INITIALIZATION ──
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
        st.session_state.pop("mb_config_loaded_for", None)

    _render_engine_section(client, "mb", "mb_agent_engine_name", _tab1_disconnect)

    # Only show remaining steps when engine exists
    if st.session_state.mb_agent_engine_name:
        ae_name = st.session_state.mb_agent_engine_name

        # Populate customization UI with current engine config
        _populate_engine_config(client, ae_name)

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
