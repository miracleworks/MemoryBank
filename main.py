import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# ADK requires Vertex AI mode — remove API key to avoid conflict
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ.pop("GOOGLE_API_KEY", None)

from shared import PROJECT_ID, LOCATION, get_client
import tab_vertex
import tab_adk

# --- PAGE CONFIG ---
st.set_page_config(page_title="Memory Bank Playground", layout="wide")

if not PROJECT_ID or not LOCATION:
    st.title("Configuration Required")
    st.error("Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in your `.env` file.")
    st.stop()

# Initialize Vertex AI (cached — survives reruns)
client = get_client()

# Ensure ADK env vars are set
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

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

# --- MAIN UI ---
st.subheader("🧠 Memory Bank Playground")

tab1, tab2 = st.tabs(["Vertex AI Agent Engine", "Agent Development Kit"])

with tab1:
    tab_vertex.render(client)

with tab2:
    tab_adk.render(client)
