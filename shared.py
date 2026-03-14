import streamlit as st
import asyncio
import datetime
import os
import vertexai
from vertexai import types

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

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


@st.cache_resource
def get_client():
    """Cached Vertex AI client — survives Streamlit reruns."""
    return vertexai.Client(project=PROJECT_ID, location=LOCATION)


def _render_engine_section(client, key_prefix, engine_state_key, on_disconnect):
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


def _populate_engine_config(client, engine_name):
    """Fetch engine config and populate session state for the customization UI.

    Only runs once per engine (tracks via mb_config_loaded_for).
    """
    if st.session_state.get("mb_config_loaded_for") == engine_name:
        return

    # Clear stale widget keys from previous engine so Streamlit doesn't
    # reuse cached widget state that overrides the values we set below.
    for key in ("mb_cfg_generation", "mb_cfg_embedding", "mb_ttl_mode",
                "mb_default_ttl_val", "mb_default_ttl_unit",
                "mb_create_ttl_val", "mb_create_ttl_unit",
                "mb_gen_created_ttl_val", "mb_gen_created_ttl_unit",
                "mb_gen_updated_ttl_val", "mb_gen_updated_ttl_unit"):
        st.session_state.pop(key, None)
    for key in ("USER_PERSONAL_INFO", "USER_PREFERENCES",
                "KEY_CONVERSATION_DETAILS", "EXPLICIT_INSTRUCTIONS"):
        st.session_state.pop(f"mb_topic_{key}", None)
    st.session_state.pop("mb_selected_topics", None)

    try:
        engine_info = client.agent_engines.get(name=engine_name)
        res = getattr(engine_info, "api_resource", engine_info)
        ctx = getattr(res, "context_spec", None)
        if not ctx:
            st.session_state.mb_config_loaded_for = engine_name
            return
        mb = getattr(ctx, "memory_bank_config", None)
        if not mb:
            st.session_state.mb_config_loaded_for = engine_name
            return

        # --- Models ---
        gen_cfg = getattr(mb, "generation_config", None)
        emb_cfg = getattr(mb, "similarity_search_config", None)
        if gen_cfg:
            model_full = getattr(gen_cfg, "model", "") or ""
            model_short = model_full.split("/")[-1] if "/" in model_full else model_full
            if model_short in GENERATION_MODELS:
                st.session_state.mb_cfg_generation = model_short
        if emb_cfg:
            model_full = getattr(emb_cfg, "embedding_model", "") or ""
            model_short = model_full.split("/")[-1] if "/" in model_full else model_full
            if model_short in EMBEDDING_MODELS:
                st.session_state.mb_cfg_embedding = model_short

        # --- Topics ---
        ALL_TOPIC_KEYS = ("USER_PERSONAL_INFO", "USER_PREFERENCES",
                          "KEY_CONVERSATION_DETAILS", "EXPLICIT_INSTRUCTIONS")
        custom_configs = getattr(mb, "customization_configs", None) or []
        topics = []
        for cc in custom_configs:
            for mt in (getattr(cc, "memory_topics", None) or []):
                managed = getattr(mt, "managed_memory_topic", None)
                if managed:
                    enum_val = getattr(managed, "managed_topic_enum", None)
                    if enum_val is not None:
                        # The API may return a string, enum object, or int.
                        # Try .name (protobuf enum), then .value, then str().
                        resolved = (
                            getattr(enum_val, "name", None)
                            or str(getattr(enum_val, "value", enum_val))
                        )
                        # Match against known keys
                        for known in ALL_TOPIC_KEYS:
                            if known == resolved or known in str(resolved):
                                topics.append(known)
                                break
        if not topics:
            # No custom topics configured — default is all active
            topics = list(ALL_TOPIC_KEYS)
        st.session_state.mb_selected_topics = topics
        for key in ALL_TOPIC_KEYS:
            st.session_state[f"mb_topic_{key}"] = key in topics

        # --- TTL ---
        ttl_cfg = getattr(mb, "ttl_config", None)
        if ttl_cfg:
            default_ttl = getattr(ttl_cfg, "default_ttl", None)
            granular = getattr(ttl_cfg, "granular_ttl_config", None)
            if default_ttl:
                st.session_state.mb_ttl_mode = "Default TTL"
                secs = _parse_ttl_seconds(str(default_ttl))
                val, unit = _best_ttl_unit(secs)
                st.session_state.mb_default_ttl_val = val
                st.session_state.mb_default_ttl_unit = unit
            elif granular:
                st.session_state.mb_ttl_mode = "Granular TTL"
                for attr, val_key, unit_key in (
                    ("create_ttl", "mb_create_ttl_val", "mb_create_ttl_unit"),
                    ("generate_created_ttl", "mb_gen_created_ttl_val", "mb_gen_created_ttl_unit"),
                    ("generate_updated_ttl", "mb_gen_updated_ttl_val", "mb_gen_updated_ttl_unit"),
                ):
                    raw = getattr(granular, attr, None)
                    if raw:
                        secs = _parse_ttl_seconds(str(raw))
                        val, unit = _best_ttl_unit(secs)
                        st.session_state[val_key] = val
                        st.session_state[unit_key] = unit
                    else:
                        st.session_state[val_key] = 0
            else:
                st.session_state.mb_ttl_mode = "None"
        else:
            st.session_state.mb_ttl_mode = "None"

        st.session_state.mb_config_loaded_for = engine_name
    except Exception:
        pass  # Silently fail — user can still set config manually


def _parse_ttl_seconds(ttl_str):
    """Parse a TTL string like '3600s' or '3600' into integer seconds."""
    s = ttl_str.strip().rstrip("s")
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _best_ttl_unit(total_seconds):
    """Convert seconds to the best human-readable (value, unit) pair."""
    if total_seconds <= 0:
        return (0, "Seconds")
    if total_seconds % 3600 == 0:
        return (total_seconds // 3600, "Hours")
    if total_seconds % 60 == 0:
        return (total_seconds // 60, "Minutes")
    return (total_seconds, "Seconds")


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
