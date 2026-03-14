import streamlit as st

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import VertexAiMemoryBankService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.genai.types import Content, Part

from shared import (
    PROJECT_ID, LOCATION, GENERATION_MODELS, SAMPLE_CONVERSATION,
    _render_engine_section, _run_async, _format_memory_ttl,
)


def render(client):
    """Render Tab 2: Agent Development Kit."""

    # ── SESSION STATE INITIALIZATION ──
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

    _render_engine_section(client, "mb_adk", "mb_adk_engine_name", _tab2_disconnect)

    # Only show remaining sections when engine is connected
    adk_engine_name = st.session_state.mb_adk_engine_name
    if adk_engine_name:
        adk_engine_id = adk_engine_name.split("/")[-1]

        # ── MEMORY BANK SETTINGS (read-only) ──
        with st.expander("Memory Bank Settings", expanded=False):
            st.caption(
                "These settings are configured on the engine. "
                "To change them, go to the **Vertex AI Agent Engine** tab → Memory Bank Customization."
            )
            try:
                _engine_info = client.agent_engines.get(name=adk_engine_name)
                _res = getattr(_engine_info, "api_resource", _engine_info)
                _ctx = getattr(_res, "context_spec", None)
                _mb = getattr(_ctx, "memory_bank_config", None) if _ctx else None

                if _mb:
                    # Models
                    _gen_model = getattr(getattr(_mb, "generation_config", None), "model", None)
                    _emb_model = getattr(getattr(_mb, "similarity_search_config", None), "embedding_model", None)
                    if _gen_model:
                        st.markdown(f"**Generation model:** `{_gen_model.split('/')[-1]}`")
                    if _emb_model:
                        st.markdown(f"**Embedding model:** `{_emb_model.split('/')[-1]}`")

                    # Topics
                    _custom = getattr(_mb, "customization_configs", None) or []
                    if _custom:
                        _topics = []
                        for cc in _custom:
                            for mt in (getattr(cc, "memory_topics", None) or []):
                                managed = getattr(mt, "managed_memory_topic", None)
                                if managed:
                                    _topics.append(getattr(managed, "managed_topic_enum", str(managed)))
                        if _topics:
                            st.markdown(f"**Topics:** {', '.join(str(t) for t in _topics)}")
                        else:
                            st.markdown("**Topics:** All (default)")
                    else:
                        st.markdown("**Topics:** All (default)")

                    # TTL
                    _ttl = getattr(_mb, "ttl_config", None)
                    if _ttl:
                        _default_ttl = getattr(_ttl, "default_ttl", None)
                        _granular = getattr(_ttl, "granular_ttl_config", None)
                        if _default_ttl:
                            st.markdown(f"**TTL:** Default — `{_default_ttl}`")
                        elif _granular:
                            st.markdown("**TTL:** Granular")
                            parts = []
                            for attr in ("create_ttl", "generate_created_ttl", "generate_updated_ttl"):
                                val = getattr(_granular, attr, None)
                                if val:
                                    parts.append(f"{attr}: `{val}`")
                            if parts:
                                st.markdown(" | ".join(parts))
                        else:
                            st.markdown("**TTL:** None")
                    else:
                        st.markdown("**TTL:** None")
                else:
                    st.info("Could not read Memory Bank config from engine.")
            except Exception as e:
                st.warning(f"Could not fetch engine config: {e}")

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

            # Active callbacks summary
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
            else:
                for line in cb_info:
                    st.markdown(f"- {line}")

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

        # ── 4. SESSION & CHAT ──
        with st.container(border=True):
            st.markdown("### 4. Session & Chat")

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
