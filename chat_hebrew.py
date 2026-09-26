# first copy: & "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8501
# the shared link is in the +----+ box that been printed
# then copy: streamlit run chat_hebrew.py
import os
import random
import time
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from toxicity import measuring_toxicity
from opposite_hebrew_nli_gpt import run_opposite_pipeline_and_render #,load_hebrew
from firebase_store_hebrew import save_into_firebase
from huggingface_hub import snapshot_download
from typing import List
import statistics
import math
# --- Global RTL styles (Hebrew/Arabic support) ---
st.markdown("""
<style>
/* Make the whole app right-to-left */
html, body, .stApp { direction: rtl; }

/* Align prose to the right by default */
.stMarkdown, [data-testid="stMarkdownContainer"] { text-align: right; }

/* Chat input (the box at the bottom) */
div[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }

/* Text inputs and textareas elsewhere */
.stTextInput input, .stTextArea textarea { direction: rtl; text-align: right; }

/* Use a Hebrew-friendly font (Google Fonts) */
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600&display=swap');
html, body, .stApp { font-family: 'Assistant', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }

/* Better mixing of Hebrew+English/URLs */
.rtl-block { direction: rtl; text-align: right; unicode-bidi: plaintext; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to a .env file or your environment.")
    st.stop()
endpoint = os.getenv("ENDPOINT_URL", "https://ai-asolomon28262ai165132345402.openai.azure.com/")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-12-01-preview"#"2025-01-01-preview",
)

# @st.cache_resource(show_spinner=False)
# def load_hebrew_cached():
#     #returns meta, embs, index, encoder
#     return load_hebrew(
#         "./hebrew",
#         repo_id="Liran73/hebrew-opposite-artifacts",
#         repo_type="dataset",
#         hf_token_env="HF_TOKEN",
#     )

def _ensure_artifacts_from_hf(
    local_dir: str | Path,
    repo_id: Optional[str] = None,
    repo_type: str = "dataset",
    token_env: str = "HF_TOKEN",
) -> None:
    """
    If repo_id is provided and required files are missing locally,
    download a snapshot from Hugging Face Hub into local_dir.
    """
    if not repo_id:
        return  # HF not requested

    target = Path(local_dir)
    need = [
        target / "meta.parquet"
    ]
    if all(p.exists() for p in need[:]):  # all required exist; skip download
        return

    token = os.getenv(token_env, None)  # required if private repo
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

#!!!

import hashlib
import os
import sys

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()
#!!!

def load_hebrew(
    art_dir: str | Path = "./hebrew",
    *,
    # pass these ONLY if you want automatic HF download when local files are missing
    repo_id: Optional[str] = None,       # e.g., "yourname/hebrew-opposite-artifacts"
    repo_type: str = "dataset",
    hf_token_env: str = "HF_TOKEN",
) -> Tuple[pd.DataFrame]:#, np.ndarray, Optional["hnswlib.Index"], SentenceTransformer]:
    """
    Load Hebrew artifacts: config.json, meta.parquet, embeddings.npy, optional hnsw_cosine.bin.
    - If repo_id is provided and files are missing, pulls them from Hugging Face Hub first.
    - Validates that meta contains a 'message' column (comment text).
    Returns: (meta, embs, index, encoder)
    """
    ART_DIR = Path(art_dir)

    # Try to bring files from HF if missing
    _ensure_artifacts_from_hf(ART_DIR, repo_id=repo_id, repo_type=repo_type, token_env=hf_token_env)

    META_PATH = ART_DIR / "meta.parquet"

    print(
        "DATA_STAGE 3: parquet information",
        {
            "path": str(META_PATH),
            "exists": META_PATH.exists(),
            "size": META_PATH.stat().st_size if META_PATH.exists() else None,
            "sha256": file_sha256(META_PATH) if META_PATH.exists() else None,
        },
        flush=True,
    )

    import pyarrow
    print(
        "DATA_STAGE 4: versions",
        {
            "python": sys.version,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        flush=True,
    )

    print("DATA_STAGE 5: before read_parquet", flush=True)

    meta = pd.read_parquet(META_PATH)

    print(
        "DATA_STAGE 6: after read_parquet",
        {
            "rows": len(meta),
            "columns": len(meta.columns),
        },
        flush=True,
    )

    return meta

@st.cache_resource(show_spinner=False)
def load_topic_dataset(topic: str):
    topic_map = {
        "bibi": ("./hebrew_bibi", "Liran73/hebrew-bibi"),
        #"democracy": ("./hebrew_democracy", "Liran73/hebrew-democracy"),
        #"police": ("./hebrew_police", "Liran73/hebrew-police"),
    }

    local_dir, repo_id = topic_map[topic]

    return load_hebrew(
        local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        hf_token_env="HF_TOKEN",
    )

def release_topic_dataset(topic: str):
    import gc
    key = f"meta_{topic}"

    if key in st.session_state:
        st.session_state[key] = None

    gc.collect()

st.set_page_config(page_title="Thesis user experiment - Hebrew version", page_icon="💬", layout="centered")

AGE_RANGES = [
    "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
]

GENDERS = ["זכר", "נקבה", "מעדיף לא לענות"]

EDUCATION_LEVELS = [
    "אחת עשרה שנות לימוד או פחות", "השכלה תיכונית", "השכלה אקדמאית-תואר ראשון", "השכלה אקדמית-תואר שני ומעלה"
]

QUESTIONS = [
    {"id": "name",        "label": "בחר לעצמך כינוי"},
    {"id": "bibi",       "label": "מה דעתך על בינימין נתניהו?"},
    #{"id": "democracy",        "label": "מה דעתך על ישראל כמדינה דמוקרטית ועל עתיד הדמוקרטיה במדינה?"},
    #{"id": "police",    "label": "מה דעתך על משטרת ישראל? האם היא מבצעת את תפקידה?"},
]

SURVEY_chat = [
    {"id": "change",   "label": "השיחה גרמה לי לשקול מחדש לפחות אחת מהדעות שלי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "safe",  "label": "הרגשתי בטוח לבטא את דעותיי בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "offensive",  "label": "הרגשתי שהשתמשתי במילים שיוכלות להיחשב כפוגעניות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "negative",        "label": "הרגשתי שהטון הכללי בשיחה היה שלילי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "frustration", "label": "הרגשתי תסכול במהלך השיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "check", "label": "סמן בהגד זה את התשובה השלישית (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_listen", "label": "הצד השני הקשיב לי ולא שפט אותי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_connection",        "label": "אני מרגיש חיבור חזק עם הצד השני בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_pov",        "label": "אני מבין טוב יותר את נקודת המבט של הצד השני עכשיו (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_continue",        "label": "אהיה מוכן להמשיך לדון בנושא הזה עם הצד השני (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_stubborn",        "label": "הצד השני בשיחה היה עקשן (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_perspectives",        "label": "נראה היה שהצד השני ואני ראינו את הנושא מנקודות מבט שונות (1=בכלל לא, 5=במידה רבה)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_position",        "label": "עמדת הצד השני הייתה דומה לשלי (1=בכלל לא, 5=במידה רבה)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_approached",        "label": "היה הבדל ניכר בין הגישה שלי לבין הצד השני באיך שניגשנו לנושא (1=בכלל לא, 5=במידה רבה)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_different_pov",        "label": "טיעוני הצד השני שיקפו נקודת מבט שונה משלי (1=בכלל לא, 5=במידה רבה)", "type": "scale", "min": 1, "max": 5},
]

SURVEY_finish = [
    {"id": "engagement",  "label": "השיחה הראשונה הייתה מעניינת ומרתקת יותר מהשיחה השנייה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "enjoy",       "label": "נהנתי לנהל יותר את השיחה הראשונה מאשר את השיחה השנייה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "change1",     "label": "השיחות הראשונה שינתה את דעתי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "change2",     "label": "השיחה השנייה שינתה את דעתי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "feedback",    "label": "במבט לאחור על המחקר כולו, איזה חלק מהחוויה בלט לך ביותר, ומדוע? \n זה יכול להיות משהו שמצאת מעניין, מפתיע, מתסכל, מהנה או מבלבל.", "type": "text"},
]

MAX_TURNS = 7

system_prompt_chat1_bibi = ""
system_prompt_chat1_democracy = ""
system_prompt_chat1_police = ""

system_prompt_chat2_bibi = ""
system_prompt_chat2_democracy = ""
system_prompt_chat2_police = ""


meta_bibi, meta_democracy, meta_police = None, None, None

# def ensure_artifacts_loaded():
#     if st.session_state.get("artifacts_loaded"):
#         return

#     meta, embs, index, encoder = load_hebrew_cached()
#     st.session_state["meta"] = meta
#     st.session_state["embs"] = embs
#     st.session_state["index"] = index
#     st.session_state["encoder"] = encoder
#     st.session_state["artifacts_loaded"] = True

def init_state():
    ss = st.session_state
    # --- Counter-balance topics order (per participant session) ---
    ss.setdefault("topic_order", None)
    if ss.topic_order is None:
        # random order once per browser session
        ss.topic_order = ["bibi"]#random.sample(["bibi", "democracy", "police"], k=3)

    ss.setdefault("stage", "instructions")
    ss.setdefault("model", "gpt-5-mini")
    ss.setdefault("temperature", 0.8)

    # base prompts
    ss.setdefault("system_prompt_chat1_bibi", system_prompt_chat1_bibi)
    #ss.setdefault("system_prompt_chat1_democracy", system_prompt_chat1_democracy)
    #ss.setdefault("system_prompt_chat1_police", system_prompt_chat1_police)
    ss.setdefault("system_prompt_chat2_bibi", system_prompt_chat2_bibi)
    #ss.setdefault("system_prompt_chat2_democracy", system_prompt_chat2_democracy)
    #ss.setdefault("system_prompt_chat2_police", system_prompt_chat2_police)

    # onboarding profile (answers)
    ss.setdefault("profile", {})

    # chats: messages start as None so we can compose system prompts with profile on first entry
    ss.setdefault("chat1_messages_bibi", None)
    #ss.setdefault("chat1_messages_democracy", None)
    #ss.setdefault("chat1_messages_police", None)
    ss.setdefault("chat2_messages_bibi", None)
    #ss.setdefault("chat2_messages_democracy", None)
    #ss.setdefault("chat2_messages_police", None)

    # survey answers
    ss.setdefault("survey_1", {})
    ss.setdefault("survey_2", {})
    ss.setdefault("survey_finish", {})

    # survey submission tracking (locks Done button after valid submission)
    ss.setdefault("survey_1_submitted", False)
    ss.setdefault("survey_2_submitted", False)
    ss.setdefault("survey_finish_submitted", False)

    ss.setdefault("chat_number_start", random.randint(1, 2))
    print(f"first chat number = {st.session_state.chat_number_start}")

    ss.setdefault("meta_bibi", None)
    #ss.setdefault("meta_democracy", None)
    #ss.setdefault("meta_police", None)
    # ss.setdefault("embs", None)
    # ss.setdefault("index", None)
    # ss.setdefault("encoder", None)
    # ss.setdefault("artifacts_loaded", False)

    # if not ss["artifacts_loaded"]:
    #     ensure_artifacts_loaded()

init_state()

# ── Design system constants ────────────────────────────────────────────────
TOPIC_COLORS  = {"bibi": "#6C63FF", "democracy": "#00B4D8", "police": "#10B981"}
TOPIC_LABELS  = {"bibi": "Benjamin Netanyahu", "democracy": "Israeli Democracy", "police": "Israel Police"}

STAGE_LABELS = {
    "instructions":                     (0,  "Introduction"),
    "privacy":                          (0,  "Privacy & Data Use"),
    "onboarding_profile":               (1,  "Your Profile"),
    "onboarding_opinions":              (2,  "Your Opinions"),
    "wait_creating_system_prompts_bibi": (3,  "Setting Up"),
    "chat1_bibi":                        (3,  "Chat 1 · בנימין נתניהו"),
    #"wait_creating_system_prompts_democracy":(3,  "Setting Up"),
    #"chat1_democracy":                       (3,  "Chat 1 · הדמוקרטיה הישראלית"),
    #"wait_creating_system_prompts_police": (5, "Setting Up"),
    #"chat1_police":                    (5,  "Chat 1 · משטרת ישראל"),
    "survey1":                          (4,  "Mid-Point Survey"),
    "wait_chat2_bibi":                   (7,  "Setting Up"),
    "chat2_bibi":                        (7,  "Chat 2 · בנימין נתניהו"),
    #"wait_chat2_democracy":                  (5,  "Setting Up"),
    #"chat2_democracy":                       (5,  "Chat 2 · הדמוקרטיה הישראלית"),
    #"wait_chat2_police":               (9,  "Setting Up"),
    #"chat2_police":                    (9,  "Chat 2 · משטרת ישראל"),
    "survey2":                          (6, "Mid-Point Survey"),
    "full_survey":                      (7, "Final Survey"),
    "due_disclosure":                   (8, "Disclosure"),
    "thanks":                           (9, "Complete"),
    "not_save":                         (9, "Complete"),
}
TOTAL_STEPS = 9#13


def inject_global_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    html, body, .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        background-color: #F2F3FA !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden !important; }

    /* ── Main content column ── */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 760px !important;
    }

    /* ── Typography ── */
    h1 { font-size: 2rem !important; font-weight: 700 !important; color: #1A1A2E !important; letter-spacing: -0.5px !important; line-height: 1.2 !important; }
    h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #1A1A2E !important; }
    h3, h4 { font-weight: 600 !important; color: #1A1A2E !important; }
    p, li, label { line-height: 1.65 !important; }

    /* ── Cards ── */
    .ui-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem 2.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 8px 24px rgba(108,99,255,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(108,99,255,0.07);
    }

    /* ── Hero banners ── */
    .ui-hero {
        background: linear-gradient(135deg, #5B52F0 0%, #7C73FF 60%, #9B8FFF 100%);
        border-radius: 20px;
        padding: 2.25rem 2.5rem;
        color: #fff;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .ui-hero::before {
        content: '';
        position: absolute; top: -50px; right: -50px;
        width: 180px; height: 180px;
        background: rgba(255,255,255,0.07);
        border-radius: 50%;
        pointer-events: none;
    }
    .ui-hero::after {
        content: '';
        position: absolute; bottom: -70px; left: -30px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
        pointer-events: none;
    }
    .ui-hero h1 { color: #fff !important; margin: 0 0 0.4rem 0 !important; font-size: 1.9rem !important; position: relative; z-index: 1; }
    .ui-hero p  { margin: 0 !important; opacity: 0.9; font-size: 1rem; position: relative; z-index: 1; }

    /* ── Topic pill ── */
    .ui-pill {
        display: inline-block;
        border-radius: 99px;
        padding: 5px 16px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        color: #fff;
    }

    /* ── Step badge ── */
    .ui-step {
        display: inline-block;
        background: #EDEEFF;
        color: #5B52F0;
        border-radius: 99px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.3px;
        margin-bottom: 1rem;
        border: 1px solid rgba(91,82,240,0.15);
    }

    /* ── Progress strip ── */
    .progress-track {
        background: #E5E7EB;
        border-radius: 99px;
        height: 6px;
        overflow: hidden;
        margin-bottom: 0.4rem;
    }
    .stage-progress-bar {
        height: 6px;
        border-radius: 99px;
        background: linear-gradient(90deg, #5B52F0, #00C4E8);
        transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #5B52F0 0%, #7C73FF 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.72rem 1.8rem !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 4px 16px rgba(91,82,240,0.32) !important;
        transition: box-shadow 0.2s ease, transform 0.15s ease !important;
        cursor: pointer !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 6px 24px rgba(91,82,240,0.42) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

    .stButton > button:not([kind="primary"]) {
        border-radius: 12px !important;
        border: 2px solid #5B52F0 !important;
        color: #5B52F0 !important;
        font-weight: 600 !important;
        background: transparent !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button:not([kind="primary"]):hover { background: #EDEEFF !important; }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 1.5px solid #DDE0EB !important;
        padding: 0.65rem 1rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        background: #FAFBFF !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #5B52F0 !important;
        background: #fff !important;
        box-shadow: 0 0 0 3px rgba(91,82,240,0.12) !important;
        outline: none !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid #DDE0EB !important;
        background: #FAFBFF !important;
    }

    /* ── Chat message bubbles ── */
    [data-testid="stChatMessage"] {
        display: flex !important;
        align-items: flex-start !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin-bottom: 1.1rem !important;
        gap: 0.75rem !important;
    }

    /* user bubble — indigo gradient, right side
       Uses structural selector so it works regardless of which testid
       Streamlit assigns to the content wrapper in any version. */
    [data-testid="stChatMessage"]:has(.wa-user) [data-testid="stChatMessageContent"] {
        background: #DCF8C6 !important;
        color: #111B21 !important;
        border-radius: 12px 12px 2px 12px !important;
        padding: 0.6rem 0.9rem !important;
        max-width: 84% !important;
        flex-grow: 0 !important;
        flex-shrink: 1 !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        box-shadow: 0 1px 1px rgba(0,0,0,0.13) !important;
        border: none !important;
    }
    [data-testid="stChatMessage"]:has(.wa-user) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has(.wa-user) [data-testid="stChatMessageContent"] * {
        color: #111B21 !important;
        margin-bottom: 0 !important;
    }

    /* assistant bubble — white card, left side */
    [data-testid="stChatMessage"]:has(.wa-assistant) [data-testid="stChatMessageContent"] {
        background: #ECE5DD !important;
        color: #111B21 !important;
        border-radius: 12px 12px 12px 2px !important;
        padding: 0.6rem 0.9rem !important;
        max-width: 84% !important;
        flex-grow: 0 !important;
        flex-shrink: 1 !important;
        border: none !important;
        box-shadow: 0 1px 1px rgba(0,0,0,0.13) !important;
    }

    /* Keep the bubble layout LTR (user right, assistant left) while the
       text inside stays RTL for Hebrew */
    [data-testid="stChatMessage"] { direction: ltr !important; }
    [data-testid="stChatMessage"] .stMarkdown,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        direction: rtl !important;
        text-align: right !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] > div {
        border-radius: 16px !important;
        border: 2px solid #DDE0EB !important;
        background: #fff !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #5B52F0 !important;
        box-shadow: 0 0 0 3px rgba(91,82,240,0.10) !important;
    }

    /* ── Turn progress bar ── */
    div[data-testid="stProgressBar"] {
        height: 8px !important;
    }
    div[data-testid="stProgressBar"] > div {
        background: #E5E7EB !important;
        border-radius: 99px !important;
        height: 8px !important;
    }
    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #5B52F0, #00C4E8) !important;
        border-radius: 99px !important;
        transition: width 0.4s ease !important;
    }

    /* ── Survey question number badge ── */
    .q-block {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }
    .q-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 30px; height: 30px;
        background: linear-gradient(135deg, #5B52F0, #7C73FF);
        color: #fff;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.78rem;
        flex-shrink: 0;
        margin-top: 1px;
        box-shadow: 0 2px 8px rgba(91,82,240,0.25);
    }
    .q-text {
        font-weight: 600;
        color: #1A1A2E;
        font-size: 0.95rem;
        line-height: 1.5;
        padding-top: 2px;
    }

    /* ── Chat info panel ── */
    .chat-header {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #E8EAFF;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }
    .chat-header-left { display: flex; align-items: center; gap: 0.75rem; }
    .chat-header-topic { font-weight: 700; color: #1A1A2E; font-size: 1rem; }
    .chat-header-sub { font-size: 0.82rem; color: #6B7280; margin-top: 1px; }
    .chat-turn-badge {
        background: #EDEEFF;
        color: #5B52F0;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 4px 12px;
        border-radius: 99px;
        white-space: nowrap;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border: none !important;
    }

    /* ── Divider ── */
    hr { border-color: #E5E7EB !important; margin: 1.5rem 0 !important; }

    /* ── Feature 1: flip user row so avatar sits on the right ── */
    [data-testid="stChatMessage"]:has(.wa-user) {
        flex-direction: row-reverse !important;
    }

    /* Collapse the invisible role-marker element so it leaves no empty gap */
    [data-testid="stChatMessageContent"] [data-testid="stElementContainer"]:has(.wa-user, .wa-assistant),
    [data-testid="stChatMessageContent"] [data-testid="element-container"]:has(.wa-user, .wa-assistant) {
        display: none !important;
    }

    /* ── Feature 2: typing indicator ── */
    .typing-bubble {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ffffff;
        border-radius: 20px 20px 20px 4px;
        padding: 0.85rem 1.15rem;
        max-width: 84%;
        border: 1px solid #E8EAFF;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0 0 1.1rem 0;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #9CA3AF;
        animation: typingPulse 1.2s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.20s; }
    .typing-dot:nth-child(3) { animation-delay: 0.40s; }
    @keyframes typingPulse {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
        40%            { transform: translateY(-5px); opacity: 1; }
    }

    /* ── Info/Notice blocks ── */
    .ui-info-block {
        background: #EEF0FF;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        border-left: 4px solid #5B52F0;
        margin: 0.75rem 0 1rem 0;
        color: #1A1A2E;
        font-size: 0.93rem;
        line-height: 1.65;
    }
    .ui-info-block strong { color: #5B52F0; }
    .ui-info-block ul { margin: 0.5rem 0 0; padding-left: 1.25rem; }
    .ui-info-block li { margin-bottom: 0.35rem; }

    /* ── Consent checklist card ── */
    .ui-consent-card {
        background: #F0FDF4;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        border-left: 4px solid #10B981;
        margin: 0.75rem 0 1rem 0;
        font-size: 0.93rem;
        line-height: 1.65;
        color: #1A1A2E;
    }
    .ui-consent-card strong { color: #059669; }
    </style>
    """, unsafe_allow_html=True)


def render_stage_progress():
    stage = st.session_state.get("stage", "instructions")
    step, label = STAGE_LABELS.get(stage, (0, ""))
    pct = int(step / TOTAL_STEPS * 100)
    st.markdown(
        f'<div class="progress-track">'
        f'<div class="stage-progress-bar" style="width:{pct}%"></div>'
        f'</div>'
        f'<p style="font-size:0.78rem;color:#9CA3AF;margin:0 0 1.25rem 0;font-weight:500;letter-spacing:0.2px">'
        f'STEP {step} OF {TOTAL_STEPS} &nbsp;·&nbsp; {label.upper()}'
        f'</p>',
        unsafe_allow_html=True
    )


inject_global_css()

def user_turns(messages):
    if not messages:
        return 0
    return sum(1 for m in messages if m["role"] == "user")

def onboarding_complete():
    p = st.session_state.profile
    return bool(p) and all((p.get(q["id"]) or "").strip() for q in QUESTIONS)

# def render_instructions():
#     st.markdown("### הוראות ניסוי המשתמשים 📜")
#     st.markdown(
# """
# שלום רב,

# להלן סקירה קצרה של המחקר הנערך בעזרתך. 

# הנך מוזמן להשתתף בניסוי משתמשים שיתקיים בשפה העברית.

# בשלב הראשון, תתבקש לספק פרטים דמוגרפיים כלליים (ללא פרטים מזהים) לצורך ניתוח סטטיסטי בלבד. לאחר מכן, נבקש לשמוע את דעתך על שלושה נושאים שונים. 

# לאחר מילוי הפרופיל, תתקיימנה שיחות עם פרטנר לשיחה. על כל אחד משלושת הנושאים תבצע שתי שיחות נפרדות, כך שבסך הכל ייערכו שש שיחות. 
# לתשומת ליבך, כחלק מהדינמיקה של הדיון, הפרטנר לשיחה עשוי להציג טיעונים, דעות או נתונים שונים. המידע המוצג על ידי הפרטנר נועד לצורכי הדיון בלבד ולא עובר בדיקת עובדות על ידינו.

# לאחר כל שלוש השיחות תתבקש לענות על סקר קצר, ועם סיום שש השיחות ייערך סקר מסכם.

# ניסוי זה הוא חלק מפרויקט מחקר מדעי. **ההשתתפות במחקר היא וולונטרית ונעשית מרצונך החופשי בלבד.**
# בעצם השלמת הניסוי, הנך נותן לנו אישור לדון בתוצאות או לפרסמן בפורומים אקדמיים. בכל פרסום עתידי, המידע יוצג בצורה אגרגטיבית (מרוכזת) כך שלא ניתן יהיה לזהותך באופן אישי. הגישה למסד הנתונים המקורי תהיה שמורה לחברי צוות המחקר בלבד.
# טרם שיתוף הנתונים מחוץ לצוות המחקר, יוסר כל מידע שעלול להביא לזיהוי פוטנציאלי. לאחר הסרת פרטים אלו, **הנתונים עשויים לשמש את צוות המחקר או להיות משותפים עם חוקרים אחרים** למטרות מחקר עתידיות. כמו כן, הנתונים האנונימיים עשויים להיות זמינים במאגרי מידע מקוונים, כדי לאפשר לחוקרים נוספים להשתמש בהם לניתוחים עתידיים.

# לחיצה על הכפתור בתחתית עמוד זה מהווה אישור לכך שהנך בן 18 ומעלה, ומסכים להשתתף בניסוי מרצונך החופשי.

# נשמח אם תתבטא בחופשיות. 

# תודה רבה על תרומתך למחקר התזה של לירן אליאב, הנערך תחת הנחייתו של ד"ר אדיר סולומון, חוקרים מאוניברסיטת חיפה.


# * ההוראות מנוסחות בלשון זכר מטעמי נוחות בלבד אך פונות לכל המינים.

# * אנא כתבו בעברית בלבד.

# * לידיעתך: הנך רשאי להפסיק את השתתפותך בכל עת ללא כל השלכה.

# * ליצירת קשר ניתן לשלוח מייל לכתובת: leliav02@campus.haifa.ac.il

# """
#     )
#     st.divider()
#     if st.button("הבנתי בואו נמשיך לבניית הפרופיל", type="primary"):
#         st.session_state.stage = "onboarding_profile"
#         st.rerun()

def render_instructions():
    render_stage_progress()

    st.markdown("""
    <div class="ui-hero">
      <h1>💬 מחקר על דיאלוג דעות חברתיות</h1>
      <p>הוראות</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ui-card">
      <div style="font-weight:600;font-size:1rem;color:#1A1A2E;margin-bottom:0.75rem">משתתף יקר,</div>
      <p style="color:#374151;font-size:0.94rem;margin:0 0 0.6rem 0">
        להלן סקירה קצרה של המחקר המבוצע בעזרתך.
        הנך מוזמן להשתתף בניסוי משתמשים שייערך בעברית.
      </p>
      <p style="color:#374151;font-size:0.94rem;margin:0 0 0.6rem 0">
        במחקר זה, תשתתף בשתי שיחות טקסטואליות על נושא אחד שנוי במחלוקת חברתית.
        במהלך המחקר, תמלא גם מספר שאלונים קצרים על חווייתך.
      </p>
      <p style="color:#374151;font-size:0.94rem;margin:0">
        מטרת מחקר זה היא להבין טוב יותר כיצד אנשים מתקשרים במהלך דיונים על נושאים שנויים במחלוקת.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin:0 0 1rem 0">
      <div style="background:#F5F6FF;border-radius:12px;padding:1rem;border:1px solid #E8EAFF">
        <div style="font-size:1.4rem;margin-bottom:0.3rem">👤</div>
        <div style="font-weight:600;font-size:0.9rem;color:#1A1A2E">פרופיל קצר</div>
        <div style="font-size:0.82rem;color:#6B7280;margin-top:2px">תתבקש לספק מידע דמוגרפי שאינו מזהה למטרות ניתוח סטטיסטי בלבד.</div>
      </div>
      <div style="background:#F5F6FF;border-radius:12px;padding:1rem;border:1px solid #E8EAFF">
        <div style="font-size:1.4rem;margin-bottom:0.3rem">🗣️</div>
        <div style="font-weight:600;font-size:0.9rem;color:#1A1A2E">שתף את דעותיך</div>
        <div style="font-size:0.82rem;color:#6B7280;margin-top:2px">תתבקש להביע את דעתך בנוגע לנושא חברתי אחד שנוי במחלוקת.</div>
      </div>
      <div style="background:#F5F6FF;border-radius:12px;padding:1rem;border:1px solid #E8EAFF">
        <div style="font-size:1.4rem;margin-bottom:0.3rem">💬</div>
        <div style="font-weight:600;font-size:0.9rem;color:#1A1A2E">שתי שיחות</div>
        <div style="font-size:0.82rem;color:#6B7280;margin-top:2px">תשתתף בסדרת שיחות עם בן/בת שיחה. אתה תנהל שתי שיחות נפרדות.</div>
      </div>
      <div style="background:#F5F6FF;border-radius:12px;padding:1rem;border:1px solid #E8EAFF">
        <div style="font-size:1.4rem;margin-bottom:0.3rem">📝</div>
        <div style="font-weight:600;font-size:0.9rem;color:#1A1A2E">סקרים קצרים</div>
        <div style="font-size:0.82rem;color:#6B7280;margin-top:2px">לאחר כל שיחה תמלא שאלון קצר ושאלון אחרון אחד בסוף.</div>
      </div>
    </div>
    <div class="ui-info-block">
      <strong>📋 שים לב:</strong>&nbsp; כחלק מהזרימה הטבעית של הדיון, בן/בת הזוג לשיחה עשוי/ה להציג טיעונים, דעות או נתונים שונים.
      מידע זה מיועד למטרות דיון בלבד ודיוקו העובדתי לא אומת על ידי צוות המחקר.
    </div>
    <div class="ui-info-block">
      <strong>📋 הנתונים שלך</strong>&nbsp; התגובות, השיחות, תשובות השאלונים והמידע הדמוגרפי הבסיסי שלך יאוחסנו
      <em>ללא מידע המזהה אותך ישירות</em> וישמש למחקר מדעי.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("המשך", type="primary", use_container_width=True):
        st.session_state.stage = "privacy"
        st.rerun()


def render_privacy():
    render_stage_progress()

    st.markdown("""
    <div class="ui-hero">
      <h1>🔒 פרטיות ושימוש בנתונים</h1>
      <p></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ui-card">
      <p style="color:#374151;font-size:0.94rem;margin:0 0 0.6rem 0">
        ההשתתפות במחקר זה היא התנדבותית לחלוטין.
      </p>
      <p style="color:#374151;font-size:0.94rem;margin:0">
        לפני שנתחיל, נרצה שתבין בדיוק איזה מידע ייאסף וכיצד הוא ישמש.
      </p>
    </div>
    <div class="ui-info-block">
      <strong>📋 מידע שנאסף במהלך מחקר זה</strong>&nbsp; אם תשתתף, נאסוף את:
      <ul>
        <li>תשובותיך לשאלות דמוגרפיות (כגון טווח גילאים, מין והשכלה)</li>
        <li>הדעה שאתה כותב לפני השיחות</li>
        <li>ההודעות שהוחלפו במהלך שתי השיחות</li>
        <li>תשובותיך לכל השאלונים</li>
        <li>מידע טכני הדרוש לניהול המחקר (כגון חותמות זמן)</li>
      </ul>
      <div style="margin-top:0.75rem;font-weight:600;color:#1A1A2E">
        לא ייכלל במערך הנתונים של המחקר מידע אישי המאפשר זיהוי ישיר (כגון שמך או כתובת הדוא"ל שלך).
      </div>
    </div>
    <div class="ui-card">
      <p style="color:#374151;font-size:0.94rem;margin:0 0 0.5rem 0">
        הנתונים שלך יעברו אנונימיזציה לפני הניתוח. מערך הנתונים האנונימי עשוי להיות:
      </p>
      <ul style="color:#374151;font-size:0.94rem;padding-left:1.25rem;margin:0 0 0.75rem 0">
        <li>מנותח על ידי צוות המחקר,</li>
        <li>מוכלל בפרסומים אקדמיים,</li>
        <li>משותף עם חוקרים אחרים,</li>
        <li>מופקד במאגרים מדעיים למחקר עתידי.</li>
      </ul>
      <p style="font-weight:600;color:#1A1A2E;margin:0 0 0.5rem 0">זהותך לא תיחשף בשום פרסום.</p>
      <p style="color:#374151;font-size:0.94rem;margin:0 0 0.5rem 0">אנו מעודדים אותך לדבר בחופשיות.</p>
      <p style="color:#374151;font-size:0.94rem;margin:0">
        תודה רבה על תרומתך למחקר התזה של לירן אליאב, שנערך בהנחייתו של ד"ר אדיר סולומון, חוקרים מאוניברסיטת חיפה.
      </p>
    </div>
    <div class="ui-consent-card">
      <div style="font-weight:600;margin-bottom:0.6rem">על ידי בחירת <strong>אני מסכים &amp; המשך</strong>, אתה מאשר כי:</div>
      <div style="line-height:2.1;font-size:0.93rem">
        ✔ אתה בן 18 לפחות<br>
        ✔ אתה מבין איזה מידע ייאסף<br>
        ✔ אתה מבין כיצד ייעשה שימוש בנתונים האנונימיים שלך<br>
        ✔ אתה מסכים מרצונך החופשי להשתתף
      </div>
    </div>
    <div style="background:#F9FAFB;border-radius:12px;padding:0.85rem 1.25rem;color:#6B7280;font-size:0.88rem;line-height:1.8;margin-bottom:0.25rem">
      ✏️ אנא כתבו בעברית בלבד. &nbsp;·&nbsp;
      🚪 שים לב: אתה רשאי לבטל את השתתפותך בכל עת ללא כל השלכות. &nbsp;·&nbsp;
      📋 ההוראות מנוסחות בלשון זכר מטעמי נוחות בלבד אך פונות לכל המינים. &nbsp;·&nbsp;
      ✉️ ליצירת קשר, אנא שלחו דוא"ל לכתובת: leliav02@campus.haifa.ac.il
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("אני מסכים &amp; המשך", type="primary", use_container_width=True):
        st.session_state.stage = "onboarding_profile"
        st.rerun()

# def render_onboarding_profile():
#     st.markdown("### 👤 בניית הפרופיל שלך")
#     with st.form("profile_form", clear_on_submit=False):
#         AGE_OPTIONS = ["— בחר מטווח הגילאים —"] + AGE_RANGES
#         GENDER_OPTIONS = ["— בחר מגדר —"] + GENDERS
#         EDUCATION_OPTIONS = ["— בחר השכלה —"] + EDUCATION_LEVELS
#         # Pre-fill from session if user returns
#         profile = st.session_state.profile

#          # nickname
#         nickname = st.text_input("כינוי *", value=profile.get("nickname", ""))

#         # selectboxes with a BLANK default (placeholder at index 0)
#         def _idx_or_placeholder(value, options):
#             try:
#                 return options.index(value) if value in options else 0
#             except Exception:
#                 return 0

#         age_idx = _idx_or_placeholder(profile.get("age_range"), AGE_OPTIONS)
#         gender_idx = _idx_or_placeholder(profile.get("gender"), GENDER_OPTIONS)
#         edu_idx = _idx_or_placeholder(profile.get("education"), EDUCATION_OPTIONS)

#         c1, c2 = st.columns(2)
#         with c1:
#             age_choice = st.selectbox("טווח גילאים *", AGE_OPTIONS, index=age_idx)
#         with c2:
#             gender_choice = st.selectbox("מגדר *", GENDER_OPTIONS, index=gender_idx)

#         education_choice = st.selectbox("השכלה *", EDUCATION_OPTIONS, index=edu_idx)

#         # gate keep: only enable when all fields are valid (not placeholders)
#         submit = st.form_submit_button("המשך אל עבר מילוי דעותיך", type="primary")

#         if submit:
#             missing = []
#             if not nickname.strip():
#                 missing.append("כינוי")
#             if age_choice == AGE_OPTIONS[0]:
#                 missing.append("טווח גילאים")
#             if gender_choice == GENDER_OPTIONS[0]:
#                 missing.append("מגדר")
#             if education_choice == EDUCATION_OPTIONS[0]:
#                 missing.append("השכלה")

#             if missing:
#                 st.error("מלא בבקשה את: " + ", ".join(missing))
#             else:
#                 st.session_state.profile.update({
#                     "nickname": nickname.strip(),
#                     "age_range": age_choice,
#                     "gender": gender_choice,
#                     "education": education_choice,
#                 })
#                 st.session_state.stage = "onboarding_opinions"
#                 st.rerun()

def render_onboarding_profile():
    render_stage_progress()
    st.markdown("""
    <div style="margin-bottom:1.25rem">
      <span class="ui-step">שלב 1 מתוך 2 - הפרופיל שלך</span>
      <h2 style="margin:0.5rem 0 0.25rem 0">👤 ספר לנו קצת על עצמך</h2>
      <p style="color:#6B7280;margin:0;font-size:0.92rem">כל המידע אנונימי ומשמש למטרות סטטיסטיות בלבד.</p>
    </div>
    """, unsafe_allow_html=True)
    #st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    with st.form("profile_form", clear_on_submit=False):
        AGE_OPTIONS = ["— בחר מטווח הגילאים —"] + AGE_RANGES
        GENDER_OPTIONS = ["— בחר מגדר —"] + GENDERS
        EDUCATION_OPTIONS = ["— בחר השכלה —"] + EDUCATION_LEVELS
        # Pre-fill from session if user returns
        profile = st.session_state.profile

         # nickname
        nickname = st.text_input("כינוי*", value=profile.get("nickname", ""))

        # selectboxes with a BLANK default (placeholder at index 0)
        def _idx_or_placeholder(value, options):
            try:
                return options.index(value) if value in options else 0
            except Exception:
                return 0

        age_idx = _idx_or_placeholder(profile.get("age_range"), AGE_OPTIONS)
        gender_idx = _idx_or_placeholder(profile.get("gender"), GENDER_OPTIONS)
        edu_idx = _idx_or_placeholder(profile.get("education"), EDUCATION_OPTIONS)

        c1, c2 = st.columns(2)
        with c1:
            age_choice = st.selectbox("טווח גילאים *", AGE_OPTIONS, index=age_idx)
        with c2:
            gender_choice = st.selectbox("מגדר *", GENDER_OPTIONS, index=gender_idx)

        education_choice = st.selectbox("השכלה *", EDUCATION_OPTIONS, index=edu_idx)

        # gate keep: only enable when all fields are valid (not placeholders)
        submit = st.form_submit_button("המשך להשלמת דעותיך", type="primary")

        if submit:
            missing = []
            if not nickname.strip():
                missing.append("כינוי")
            if age_choice == AGE_OPTIONS[0]:
                missing.append("טווח גילאים")
            if gender_choice == GENDER_OPTIONS[0]:
                missing.append("מגדר")
            if education_choice == EDUCATION_OPTIONS[0]:
                missing.append("השכלה")

            if missing:
                st.error("בבקשה מלא את: " + ", ".join(missing))
            else:
                st.session_state.profile.update({
                    "nickname": nickname.strip(),
                    "age_range": age_choice,
                    "gender": gender_choice,
                    "education": education_choice,
                })
                st.session_state.stage = "onboarding_opinions"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# def render_onboarding_opinions():
#     st.markdown("### 🗣️ הדעות שלך (עד 100 תווים)")
#     with st.form("opinions_form", clear_on_submit=False):
#         opinions = st.session_state.profile.get("opinions", {})

#         bibi = st.text_area("מה דעתך על בינימין נתניהו? *", value=opinions.get("ביבי", ""), height=90, max_chars=100)
#         democracy = st.text_area("מה דעתך על ישראל כמדינה דמוקרטית ועל עתיד הדמוקרטיה במדינה? *", value=opinions.get("דמוקרטיה", ""), height=90, max_chars=100)
#         police = st.text_area("מה דעתך על משטרת ישראל? האם היא מבצעת את תפקידה? *", value=opinions.get("משטרה", ""), height=90, max_chars=100)

#         st.caption(f"Lengths - Bibi: {len(bibi)}/100 | Democracy: {len(democracy)}/100 | Police: {len(police)}/100")

#         next_btn = st.form_submit_button("מילאתי את כל דעותי", type="primary", use_container_width=True)

#         if next_btn:
#             errs = []
#             if not bibi.strip(): errs.append("Bibi")
#             if not democracy.strip(): errs.append("Democracy")
#             if not police.strip(): errs.append("Police")

#             if errs:
#                 st.error("אנא מלא את דעותיך על: " + ", ".join(errs))
#             else:
#                 st.session_state.profile["opinions"] = {
#                     "ביבי": bibi.strip(),
#                     "דמוקרטיה": democracy.strip(),
#                     "משטרה": police.strip(),
#                 }

#                 first_topic = st.session_state.topic_order[0]
#                 st.session_state.stage = f"wait_creating_system_prompts_{first_topic}"
#                 st.rerun()

def render_onboarding_opinions():
    render_stage_progress()
    st.markdown("""
    <div style="margin-bottom:1.25rem">
      <span class="ui-step">שלב 2 מתוך 2 - דעותיך</span>
      <h2 style="margin:0.5rem 0 0.25rem 0">🗣️ שתף את דעותיך</h2>
      <p style="color:#6B7280;margin:0;font-size:0.92rem">כתוב עד 100 תווים לנושא. היה כן, אין תשובות נכונות או שגויות.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("opinions_form", clear_on_submit=False):
        opinions = st.session_state.profile.get("opinions", {})

        st.markdown(
            '<div style="border-left:4px solid #6C63FF;padding-left:1rem;margin-bottom:0.25rem">'
            '<strong>בנימין נתניהו\n</strong></div>',
            unsafe_allow_html=True
        )
        bibi = st.text_area("מה דעתך על תפקודו של בנימין נתניהו כמנהיג פוליטי? ספר על משהו שהוא עשה או קידם שאתה תומך בו או מתנגד לו, והסבר למה. *", value=opinions.get("bibi", ""), height=140, max_chars=100)

        # st.markdown(
        #     '<div style="border-left:4px solid #00B4D8;padding-left:1rem;margin-bottom:0.25rem;margin-top:1rem">'
        #     '<strong>הדמוקרטיה הישראלית</strong></div>',
        #     unsafe_allow_html=True
        # )
        # democracy = st.text_area("מה דעתך על ישראל כמדינה דמוקרטית ועל עתיד הדמוקרטיה במדינה? *", value=opinions.get("democracy", ""), height=140, max_chars=100)

        # st.markdown(
        #     '<div style="border-left:4px solid #10B981;padding-left:1rem;margin-bottom:0.25rem;margin-top:1rem">'
        #     '<strong>משטרת ישראל</strong></div>',
        #     unsafe_allow_html=True
        # )
        #police = st.text_area("מה דעתך על משטרת ישראל? האם היא מבצעת את תפקידה? *", value=opinions.get("police", ""), height=140, max_chars=100)

        next_btn = st.form_submit_button("סיימתי לכתוב את דעותיי", type="primary", use_container_width=True)

        if next_btn:
            errs = []
            if not bibi.strip(): errs.append("Benjamin Netanyahu")
            #if not democracy.strip(): errs.append("Israeli Democracy")
            #if not police.strip(): errs.append("Israel Police")

            if errs:
                st.error("אנא כתוב את דעתך על: " + ", ".join(errs))
            else:
                st.session_state.profile["opinions"] = {
                    "ביבי": bibi.strip(),
                    #"דמוקרטיה": democracy.strip(),
                    #"משטרה": police.strip(),
                }

                #st.session_state.stage = "wait_creating_system_prompts_blm"
                first_topic = st.session_state.topic_order[0]
                st.session_state.stage = f"wait_creating_system_prompts_{first_topic}"
                st.rerun()


# def generate_system_prompt_chat1_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
#     if number_comments:
#         bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
#     else:
#         bg_block = "\n".join(f"- {c}" for c in background_comments)

#     prompt =  f"""
# אתה משתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. עליך לדמות את המגיב הבא תוך אימוץ העמדה שלו בנושא: {topic_label}. 
# התגובות שהמשתמש כתב בעברו עבור הנושא {topic_label} הינן: 
# {bg_block}
# הוראות להמשך השיח:
# - אל תצטט תגובות קודמות.
# - המשך את סגנון הכתיבה של אותו משתמש בדיוק. 
# - אל תחפש לרצות.
# - ציין עמדות ברורות. 
# - אל תימנע מלהשתמש בשפה שאינה הולמת כל עוד היא תואמת את סגנון הדיבור של המשתמש אותו אתה מדמה.
# """
#     return prompt

# def generate_system_prompt_chat1_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
#     if number_comments:
#         bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
#     else:
#         bg_block = "\n".join(f"- {c}" for c in background_comments)

#     prompt =  f"""
# אתה משתמש פייסבוק שמגיב לפוסטים של פוליטיקאים ישראלים בפייסבוק. עליך להתחזות למגיב הבא, תוך אימוץ עמדתו בנושא: {topic_label}.

# תגובות קודמות של המשתמש לנושא זה הן:
# {bg_block}

# התנהגות ליבה:
# - הישאר עקבי עם העמדה וההיגיון של תגובות הרקע.
# - הגב ישירות להודעה האחרונה של המשתמש.
# - הגב תמיד עם תשובה מלאה. לעולם אל תחזיר תשובה ריקה.

# גבולות השיחה:
# - הישאר בתחום שהוקצה: {topic_label}.
# - אם המשתמש שואל על נושא לא קשור, אשר זאת בקצרה, ולאחר מכן הפנה את השיחה חזרה אל {topic_label}.
# - אל תענו על שאלות עובדתיות שאינן קשורות כגון מזג אוויר, ספורט, חדשות, ייעוץ אישי או עזרה טכנית.
# - בעת שינוי כיוון, שמרו על אותה עמדה שהוקצתה לכם והמשיכו את הדיון באופן טבעי.

# סגנון אינטראקציה:
# - הצגת עמדתך בצורה ברורה וישירה.
# - אל תנסה לרצות את המשתמש או לרכך את עמדתך.
# - אל תימנע משימוש בשפה בלתי הולמת.

# סגנון כתיבה:
# - התאם לטון הכללי של תגובות הרקע (לא ניסוח מדויק).
# - כתוב כמו תגובה אמיתית בפייסבוק, לא כמו חיבור.
# - שמור על משפטים קצרים וטבעיים.
# - כתוב תמיד בלשון זכר בלבד. אל תשתמש אף פעם בניסוחים כפולים או ניטרליים מגדרית כגון "חושב/ת", "שומע/ת", "מבין/ה" או "את/ה". גם אם מגדר המגיב אינו ידוע, השתמש בלשון זכר באופן עקבי לאורך כל התגובה.  
# - אל תשתמש ב-"—" בתשובתך.

# אילוצי פלט:
# - 2-4 משפטים בלבד.
# - 60-120 מילים לכל היותר.
# - פסקה אחת בלבד.
# - ללא נקודות תבליט, ללא רשימות, ללא מבנה פורמלי.
# - אם תגובתך ארוכה מדי, קצר אותה.

# מטרה:
# לכתוב תגובה אחת טבעית לפייסבוק שתמשיך את הדיון תוך הבעת עמדתה בצורה ברורה.
# """

#     return prompt

def generate_system_prompt_chat1_per_topic(
    topic_label: str,
    background_comments: List[str],
    *,
    number_comments: bool = False
) -> str:

    # ---------------------------------------------------------
    # Build background comments block
    # ---------------------------------------------------------
    if number_comments:
        bg_block = "\n".join(
            f"{i+1}. {c}"
            for i, c in enumerate(background_comments)
        )
    else:
        bg_block = "\n".join(
            f"- {c}"
            for c in background_comments
        )

    # ---------------------------------------------------------
    # Calculate typical persona response length
    # ---------------------------------------------------------
    word_counts = [
        len(c.split())
        for c in background_comments
        if c and c.strip()
    ]

    if word_counts:
        median_words = int(round(statistics.median(word_counts)))

        # Allow approximately ±40% around the persona's typical length
        min_words = max(5, math.floor(median_words * 0.60))
        max_words = min(100, math.ceil(median_words * 1.40))

    else:
        # Fallback in case there are no valid background comments
        median_words = 30
        min_words = 20
        max_words = 40

    # ---------------------------------------------------------
    # System prompt
    # ---------------------------------------------------------
    prompt = f"""
אתה משתמש פייסבוק שמגיב לפוסטים של פוליטיקאים ישראלים בפייסבוק. עליך להתחזות למגיב הבא, תוך אימוץ עמדתו בנושא: {topic_label}.

תגובות קודמות של המשתמש לנושא זה הן:

{bg_block}

התנהגות ליבה:

- הישאר עקבי עם העמדה וההיגיון של תגובות הרקע.
- הגב ישירות להודעה האחרונה של המשתמש.
- הגב תמיד עם תשובה מלאה. לעולם אל תחזיר תשובה ריקה.

גבולות השיחה:

- הישאר בתחום שהוקצה: {topic_label}.
- אם המשתמש שואל על נושא לא קשור, אשר זאת בקצרה, ולאחר מכן הפנה את השיחה חזרה אל {topic_label}.
- אל תענה על שאלות עובדתיות שאינן קשורות כגון מזג אוויר, ספורט, חדשות, ייעוץ אישי או עזרה טכנית.
- בעת שינוי כיוון, שמור על אותה עמדה שהוקצתה לך והמשך את הדיון באופן טבעי.

סגנון אינטראקציה:

- הצג את עמדתך בצורה ברורה וישירה.
- אם אינך מסכים עם המשתמש, אמור זאת בצורה ברורה והסבר מדוע.
- הגן על העמדה שלך גם כאשר המשתמש מתנגד לך או מאתגר אותך.
- אל תנסה לרצות את המשתמש או לרכך את עמדתך.
- אל תנסה לגרום למשתמש להרגיש מובן לפני שאתה מציג את עמדתך.
- אל תשתמש בניסוחים טיפוליים או מפייסים כגון "אני מבין אותך", "אני שומע אותך", "אני מבין מאיפה אתה מגיע" או "אני שומע את הכעס שלך".
- אל תחפש בכוח מכנה משותף ואל תסכם את רגשות המשתמש.
- אל תימנע משפה בוטה, סלנג או קללות אם הם מתאימים באופן טבעי לסגנון של תגובות הרקע ולהקשר של השיחה.
- אם המשתמש תוקף, מקלל או משתמש בשפה בוטה, הגב לתוכן הטענה שלו ואל תנזוף בו על צורת הדיבור.
- אל תגיד למשתמש להירגע, להפסיק לקלל, לדבר בצורה מכבדת או לשנות את הטון שלו.
- אל תפסול טענה רק משום שהיא נוסחה בצורה כועסת, בוטה או פוגענית.
- אין צורך להגן על האדם או הקבוצה שהפרסונה תומכת בהם מפני עצם השימוש בכינויי גנאי; התמקד במחלוקת עצמה.
- מותר להשתמש בשפה ישירה, בוטה, לא-פורמלית ואף פוגענית כאשר היא מתאימה באופן טבעי לסגנון של תגובות הרקע ולהקשר. אל תוסיף שפה כזו באופן מלאכותי.

סגנון כתיבה:

- התאם לטון הכללי של תגובות הרקע, אך אל תעתיק אותן מילה במילה.
- כתוב כמו תגובה אמיתית בפייסבוק, לא כמו חיבור או כמו עוזר AI.
- שמור על משפטים קצרים, טבעיים וישירים.
- כתוב תמיד בלשון זכר בלבד. אל תשתמש אף פעם בניסוחים כפולים או ניטרליים מגדרית כגון "חושב/ת", "שומע/ת", "מבין/ה" או "את/ה".
- אל תשתמש ב-"—" בתשובתך.

אורך התגובה:

- תגובות הרקע של המגיב הן בדרך כלל באורך של כ-{median_words} מילים.
- כתוב תגובה באורך דומה, בדרך כלל בין {min_words} ל-{max_words} מילים.
- הטווח הוא הנחיה לסגנון ולא דרישה לספירת מילים מדויקת.
- אל תאריך את התגובה באופן מלאכותי רק כדי להגיע לאורך מסוים.

אילוצי פלט:

- פסקה אחת בלבד.
- ללא נקודות תבליט, ללא רשימות וללא מבנה פורמלי.

מטרה:

כתוב תגובת פייסבוק אחת טבעית ואותנטית שממשיכה את הדיון ומביעה בבירור את עמדתו של המגיב. המטרה אינה להגיע להסכמה עם המשתמש, אלא להגיב מתוך הפרסונה שהוקצתה.

"""

    return prompt

# def generate_system_prompt_chat2_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
#     if number_comments:
#         bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
#     else:
#         bg_block = "\n".join(background_comments)

#     prompt = f"""
# אתה משחק תפקיד כמשתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. תדמה את המגיב הבא תוך אימוץ העמדה שלו בנושא {topic_label}. 

# תגובות הרקע לעינך בלבד, אסור לך לצטט אותן בשום צורה.
# {bg_block}

# אתה מגלם נטייה גבוהה של כנות-ענווה: אתה נמנע ממניפולציה של אחרים למטרות רווח אישי, מרגיש מעט פיתוי להפר כללים, לא מעוניין בעושר מפואר ובמותרות, ולא מרגיש זכאות מיוחדת למעמד חברתי מורם. אתה משתתף בדיון באופן עקבי חיובי וחביב: מניח תום לב, מכיר במאמץ, מדגיש קרקע משותפת ומביע הערכה כאשר אחרים חולקים את נקודת המבט שלהם.

# השתמש בתקשורת לא אלימה בכל שלב מבלי לתת לה שם: התחיל בתצפית ניטרלית הקשורה למה שהאדם האחר אמר זה עתה, תן שם קצר ל... רגשות משלכם, קשור אותם לצרכים או לערכים הבסיסיים, וסיים בבקשה ברורה, ניתנת לביצוע, ולא כפייתית, המזמינה שיתוף פעולה. לפני הצעת נקודות נגד או ראיות, ראשית שקף את הרגשות והצרכים הסבירים של האדם האחר כדי להראות הבנה. שמור על שפה חמה, מכבדת ומעודדת. הימנע מדפוסים מנוכרים: ללא שיפוטים מוסריים, ללא השוואות מבישות, ללא הכחשת אחריות, ללא דרישות או איומים, וללא מסגור של "מגיע/עונש".

# שמור על העמדה וההיגיון המהותיים של הערת הרקע שהוקצתה. אתה רשאי לנסח אותה מחדש בצורה אמפתית יותר או להוסיף ראיות לפי בקשה, אך אל תסתור אותן. הפוך את עמדתך למובנת באמצעות האופן שבו אתה מנסח תצפיות, דוגמאות ובקשות, כך שקורא קשוב יוכל להסיק את עמדתך מבלי להזדקק לבקש אותה. אל תכפו את הנושא. כאשר הודעת המשתמש נוגעת בבירור לנושא זה או לטענות סמוכות, הצג את עמדתך בתמציתיות באותה תגובה. כאשר הודעת המשתמש עוסקת במשהו אחר, תתייחס לנושא שלו תוך שמירה על טון ודוגמאות עקביים עם עמדתכם.

# כללי סגנון ופלט: כתוב כמו איש פייסבוק טיפוסי, אבל בחמימות ובאדיבות. שמור על משפטים קצרים וברורים. אל תצטט או תנסח מחדש את תגובות הרקע, דבר מהן כמילים שלך. תפחית רגעים סוערים על ידי הכרה ברגשות ובצרכים משותפים. תציע צעד אחד קטן, ספציפי ולא תובעני. שמור על טון ידידותי ומלא תקווה לכל אורך הדרך. הישאר בתפקיד בכל עת, בהתאם לטענות ולטון המרכזיים של תגובת הרקע.

# משימה: כאשר המשתמש מתייחס לדיון ספציפי, השב רק בתור אותו מגיב בפייסבוק. כתוב תגובה אחת ועצמאית שממשיכה את השרשור הנכון בפייסבוק, תוך התייחסות לתגובת הרקע הרלוונטית כנקודת המבט המוצאת שלך, והגב ישירות לנקודת המשתמש האחרונה באותו דיון.
# """

#     return prompt

# def generate_system_prompt_chat2_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
#     if number_comments:
#         bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
#     else:
#         bg_block = "\n".join(background_comments)

#     prompt = f"""
# אתה משחק תפקידים כמשתמש פייסבוק בפוסטים של פוליטיקאים ישראלים. אתה מאמץ את עמדתו וההיגיון של המגיב הבא בנושא: {topic_label}.

# תגובות רקע (אין לצטט אותן; לשימוש פנימי בלבד):
# {bg_block}

# דרישות ליבה:
# - הישאר עקבי עם העמדה וההיגיון של תגובת הרקע.
# - הגב ישירות להודעה האחרונה של המשתמש.
# - הגב תמיד עם תשובה מלאה. לעולם אל תחזיר תשובה ריקה.

# גבולות השיחה:
# - הישאר בתחום שהוקצה: {topic_label}.
# - אם המשתמש שואל על נושא לא קשור, אשר זאת בקצרה, ולאחר מכן הפנה את השיחה חזרה אל {topic_label}.
# - אל תענו על שאלות עובדתיות שאינן קשורות כגון מזג אוויר, ספורט, חדשות, ייעוץ אישי או עזרה טכנית.
# - בעת שינוי כיוון, שמרו על אותה עמדה שהוקצתה והמשיכו את הדיון באופן טבעי תוך שימוש במבנה בסגנון NVC הנדרש.

# תקשורת לא אלימה (NVC) - חובה:
# בכל תגובה, יש לעקוב אחר המבנה הבא באופן טבעי (מבלי לתת לה שם):

# 1. תצפית - חזרה קצרה על מה שאמר המשתמש (נייטרלי, ללא שיפוטיות)
# 2. הכרה - שיקוף רגש או דאגה אפשריים מאחורי המסר שלהם
# 3. פרספקטיבה - הבעת דעתך בצורה ברורה ורגועה (גם אם אינך מסכים)
# 4. בקשה - הצעת צעד או שאלה קטנים ולא תובעניים שמזמינים דיאלוג

# עשה זאת בצורה טבעית, לא כתבנית רשמית.

# טון:
# - מכבד, רגוע ולא עוין
# - הנחה של תום לב
# - ללא שיפוטיות מוסרית, ללא התקפות, ללא סרקזם

# סגנון:
# - כתוב כמו תגובה אמיתית ברדיט (לא אקדמית או פורמלית)
# - 2-4 משפטים בלבד
# - 60-120 מילים לכל היותר
# - פסקה אחת בלבד
# - ללא רשימות או עיצוב
# - כתוב תמיד בלשון זכר בלבד. אל תשתמש אף פעם בניסוחים כפולים או ניטרליים מגדרית כגון "חושב/ת", "שומע/ת", "מבין/ה" או "את/ה". גם אם מגדר המגיב אינו ידוע, השתמש בלשון זכר באופן עקבי לאורך כל התגובה.  
# - אין להשתמש ב-"—" בתשובתך.

# הנחיות:
# - היו ישירים אך לא אגרסיביים
# - שמרו על גישה שיחתית ואנושית
# - אל תסבירו יתר על המידה או תטיפו

# מטרה:
# כתוב תגובה טבעית אחת בפייסבוק שתמשיך את הדיון תוך הבעת עמדתך המוטלת על ידי שימוש בתקשורת בסגנון NVC. אם תשובתך חורגת מ-120 מילים או 4 משפטים, קצר אותה.
# """

#     return prompt

def generate_system_prompt_chat2_per_topic(
    topic_label: str,
    background_comments: List[str],
    *,
    number_comments: bool = False
) -> str:

    # ---------------------------------------------------------
    # Build background comments block
    # ---------------------------------------------------------
    if number_comments:
        bg_block = "\n".join(
            f"{i+1}. {c}"
            for i, c in enumerate(background_comments)
        )
    else:
        bg_block = "\n".join(
            f"- {c}"
            for c in background_comments
        )

    # ---------------------------------------------------------
    # Calculate typical persona response length
    # ---------------------------------------------------------
    word_counts = [
        len(c.split())
        for c in background_comments
        if c and c.strip()
    ]

    if word_counts:
        median_words = int(round(statistics.median(word_counts)))

        # Allow flexibility around the persona's typical length
        min_words = max(5, math.floor(median_words * 0.60))
        max_words = min(100, math.ceil(median_words * 1.40))

        if max_words < min_words:
            max_words = min_words

    else:
        # Fallback if background comments are unavailable
        median_words = 30
        min_words = 20
        max_words = 40

    # ---------------------------------------------------------
    # System prompt
    # ---------------------------------------------------------
    prompt = f"""תפקיד:
אתה משחק תפקיד של משתמש פייסבוק בדיון על {topic_label}. אמץ את העמדה, ההיגיון וסגנון הכתיבה שעולים מתגובות הרקע.

תגובות הרקע:
{bg_block}

שמירה על הפרסונה - דרישה עליונה:
- העמדה וההיגיון שעולים מתגובות הרקע הם המקור היחיד לעמדתך לאורך השיחה.
- שמור על אותה עמדה גם כאשר המשתמש מציג טיעונים משכנעים, פתרונות, ביקורת או עמדות חדשות.
- אל תאמץ עמדה, מדיניות, פתרון או דרישה רק משום שהמשתמש הציע אותם.
- הכרה בדאגה של המשתמש אינה הסכמה עם עמדתו או עם הפתרון שלו.
- אם עמדת המשתמש מתנגשת עם עמדת הפרסונה, המשך לייצג את הפרסונה בצורה מכבדת וברורה.
- NVC משנה רק את אופן התקשורת. הוא לעולם אינו גובר על הדרישה לשמור על העמדה וההיגיון של תגובות הרקע.

תקשורת לא אלימה (NVC) - חובה:
בכל תגובה, פעל באופן טבעי לפי המבנה הבא, מבלי לציין את שמות השלבים:

1. תצפית - התייחס בקצרה ובאופן ניטרלי למה שהמשתמש אמר, ללא שיפוט.
2. הכרה - הכר ברגש אפשרי או בדאגה שעומדים מאחורי דבריו.
3. פרספקטיבה - הצג את עמדתך באופן ברור ורגוע, גם כאשר אתה חולק על המשתמש.
4. בקשה - סיים בשאלה קצרה ולא תובענית שמזמינה את המשתמש להסביר או להמשיך את נקודת מבטו. השאלה צריכה להיות עקבית עם עמדת הפרסונה ואסור לה להציע פעולה משותפת לקידום עמדה שסותרת אותה.

עשה זאת באופן טבעי ולא כתבנית פורמלית.
- אל תשאל שוב שאלה שהמשתמש כבר ענה עליה; השאלה צריכה לקדם את השיחה מנקודה שכבר הגיעה אליה.
- הכרה ברגש או בדאגה של המשתמש אינה הסכמה עם עמדתו ואינה סיבה לשנות את עמדת הפרסונה.

המשך השיחה:
- התייחס לכל הודעה כהמשך של השיחה כולה, לא כתגובה עצמאית.
- זכור מה המשתמש ואתה כבר אמרתם.
- הגב בעיקר לדבר החדש בהודעה האחרונה.
- אל תחזור על טיעון, שאלה או דוגמה שכבר הובהרו.
- אל תשאל שוב דבר שהמשתמש כבר ענה עליו.
- אם נוצרה הסכמה, אל תמשיך להתווכח על אותה נקודה; המשך לנקודת המחלוקת שנותרה.
- אם המשתמש אומר שאתה חוזר על עצמך או שכבר ענה, קבל זאת והתקדם.

טבעיות:
- כתוב כמו אדם אמיתי בפייסבוק, לא כמו מטפל, מגשר או עוזר AI.
- אל תיישם NVC כתבנית קבועה. גוון את מבנה התגובה באופן טבעי.
- אין צורך לזהות רגש בכל הודעה או לפתוח כל תגובה בהכרה.
- התמקד בדרך כלל בנקודה מרכזית אחת בכל תגובה.
- אל תמציא עובדות או מחלוקות עובדתיות כדי לחזק את עמדתך.

סגנון:
- התאם לטון ולרמת הפורמליות של תגובות הרקע בלי להעתיק אותן.
- כתוב בעברית טבעית ושיחתית ובלשון זכר בלבד.
- כתוב פסקה אחת בלבד.
- אורך תגובות הרקע הוא בדרך כלל כ-{median_words} מילים; נסה לכתוב באורך דומה, בדרך כלל בין {min_words} ל-{max_words} מילים.
"""

    return prompt


def build_chat_env_bibi():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        st.markdown(
            '<span class="ui-pill" style="background:#6C63FF">בנימין נתניהו</span>',
            unsafe_allow_html=True
        )
        st.markdown("#### מוצאים בן/בת הזוג לשיחה שלך…")
        st.caption("אנא אל תסגור את החלון זה. זה בדרך כלל לוקח פחות מדקה.")
        with st.spinner(""):
            topic_map = {
            "ביבי": ["בינימין נתניהו", "ביבי"],
            }

            meta_bibi = load_topic_dataset("bibi")

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            STEP_LABELS_Bibi = {
                5: "מתחילים…",
                20: "אוספים מידע חשוב…",
                50: "בוחרים את ההתאמה הטובה ביותר…",
                75: "מעדכנים את המערכת…",
                90: "כמעט מוכנים…",
                100: "מוכן ✅",
            }

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(STEP_LABELS_Bibi.get(pct, msg))

            for key in ["ביבי"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=meta_bibi,#st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
                    on_progress=on_progress)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2

                
            st.session_state.opposite = triples
        
            st.session_state.system_prompt_chat1_bibi = system_prompts_chat1['ביבי']
            st.session_state.system_prompt_chat2_bibi = system_prompts_chat2['ביבי']
            st.session_state.chat1_messages_bibi = None

            st.session_state.stage = "chat1_bibi"
            on_progress(100, "מוכן ✅")
            progress_text.empty()

            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_democracy():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        st.markdown(
            '<span class="ui-pill" style="background:#6C63FF">הדמוקרטיה הישראלית</span>',
            unsafe_allow_html=True
        )
        st.markdown("#### מוצאים בן/בת הזוג לשיחה שלך…")
        st.caption("אנא אל תסגור את החלון זה. זה בדרך כלל לוקח פחות מדקה.")
        with st.spinner(""):
            
            topic_map = {
            "דמוקרטיה": ['דמוקרטיה', 'הדמוקרטיה', 'בדמוקרטיה', 'לדמוקרטיה', 'דמוקרטי', 'דמוקרטית', 'שהדמוקרטיה', 'דמוקרטים'],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            STEP_LABELS_Democracy = {
                5: "מתחילים…",
                20: "אוספים מידע חשוב…",
                50: "בוחרים את ההתאמה הטובה ביותר…",
                75: "מעדכנים את המערכת…",
                90: "כמעט מוכנים…",
                100: "מוכן ✅",
            }

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(STEP_LABELS_Democracy.get(pct, msg))

            for key in ["דמוקרטיה"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=meta_democracy,#st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
                    on_progress=on_progress)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2
                
            st.session_state.opposite = triples
            st.session_state.system_prompt_chat1_democracy = system_prompts_chat1['דמוקרטיה']
            st.session_state.system_prompt_chat2_democracy = system_prompts_chat2['דמוקרטיה']
            st.session_state.chat1_messages_democracy = None

            st.session_state.stage = "chat1_democracy"
            on_progress(100, "מוכן ✅")
            progress_text.empty()
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_police():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        st.markdown(
            '<span class="ui-pill" style="background:#6C63FF">משטרת ישראל</span>',
            unsafe_allow_html=True
        )
        st.markdown("#### מוצאים בן/בת הזוג לשיחה שלך…")
        st.caption("אנא אל תסגור את החלון זה. זה בדרך כלל לוקח פחות מדקה.")
        with st.spinner(""):
            
            topic_map = {
            "משטרה": ['המשטרה', 'השוטרים', 'שוטרים', 'שוטר', 'שוטרת', 'שוטרות', 'השוטר', 'משטרה', 'משטרת ישראל', 'לשוטרים', 'לשוטר', 'לשוטרת', 'למשטרה' ],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            STEP_LABELS_Police = {
                5: "מתחילים…",
                20: "אוספים מידע חשוב…",
                50: "בוחרים את ההתאמה הטובה ביותר…",
                75: "מעדכנים את המערכת…",
                90: "כמעט מוכנים…",
                100: "מוכן ✅",
            }

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(STEP_LABELS_Police.get(pct, msg))

            for key in ["משטרה"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=meta_police,#st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
                    on_progress=on_progress)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2
                
            st.session_state.opposite = triples

            st.session_state.system_prompt_chat1_police = system_prompts_chat1['משטרה']
            st.session_state.system_prompt_chat2_police = system_prompts_chat2['משטרה']
            st.session_state.chat1_messages_police = None

            st.session_state.stage = "chat1_police"
            on_progress(100, "מוכן ✅")
            progress_text.empty()

            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_bibi_chat2():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        st.markdown(
            '<span class="ui-pill" style="background:#6C63FF">בנימין נתניהו</span>',
            unsafe_allow_html=True
        )
        st.markdown("#### מתחיל את השיחה השנייה שלך…")

        with st.spinner(""):
            st.session_state.chat2_messages_bibi = None

            st.session_state.stage = "chat2_bibi"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_democracy_chat2():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        st.markdown(
            '<span class="ui-pill" style="background:#6C63FF">הדמוקרטיה הישראלית</span>',
            unsafe_allow_html=True
        )
        st.markdown("#### מתחיל את השיחה השנייה שלך…")

        with st.spinner(""):
            st.session_state.chat2_messages_democracy = None

            st.session_state.stage = "chat2_democracy"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_police_chat2():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        st.markdown(
            '<span class="ui-pill" style="background:#6C63FF">משטרת ישראל</span>',
            unsafe_allow_html=True
        )
        st.markdown("#### מתחיל את השיחה השנייה שלך…")

        with st.spinner(""):
            st.session_state.chat2_messages_police = None

            st.session_state.stage = "chat2_police"
            st.rerun()

    _wait_till_finish_system_prompts()

def render_chat(title, messages_key, base_prompt_key, next_button_label, next_stage, key, topic):
    render_stage_progress()

    color = TOPIC_COLORS.get(topic, "#5B52F0")
    label_text = TOPIC_LABELS.get(topic, topic.upper())

    turns_so_far = user_turns(st.session_state.get(messages_key) or [])
    st.markdown(
        f'<div class="chat-header">'
        f'  <div class="chat-header-left">'
        f'    <span class="ui-pill" style="background:{color};margin:0">{label_text}</span>'
        f'    <div>'
        f'      <div class="chat-header-topic">{title}</div>'
        f'    </div>'
        f'  </div>'
        f'  <span class="chat-turn-badge">Turn {min(turns_so_far, MAX_TURNS)} / {MAX_TURNS}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    model ="gpt-5-mini"#"gpt-4o-mini"
    temperature = 0.8
    system_prompt = ""
    ASSISTANT_AVATAR = "🙃"  
    USER_AVATAR = "🙂"

    if st.session_state[messages_key] is None:
        system_prompt = st.session_state[base_prompt_key] #make_system_prompt(st.session_state[base_prompt_key], st.session_state.profile)
        print(f"system prompt: {system_prompt}")
        st.session_state[messages_key] = [{"role": "system", "content": system_prompt}]

    turns_key            = f"{messages_key}_turns"
    user_scores_key      = f"{messages_key}_user_toxicity_{topic}"
    assistant_scores_key = f"{messages_key}_assistant_toxicity_{topic}"
    st.session_state.setdefault(turns_key, [])
    st.session_state.setdefault(user_scores_key, [])
    st.session_state.setdefault(assistant_scores_key, [])

    # =========================
    # seed the FIRST turn automatically with the user's opinion
    # =========================
    seeded_key = f"{messages_key}_seeded"
    st.session_state.setdefault(seeded_key, False)

    pending_key = f"{messages_key}_pending_response"
    st.session_state.setdefault(pending_key, False)

    if len(st.session_state[messages_key]) == 1:
        # pull the chosen topic key (set earlier in your flow) and the opinion text
        print("len=1")
        topic_key = st.session_state.get("start_topic_key")
        opinions = (st.session_state.get("profile", {}) or {}).get("opinions", {}) or {}

        opinion_text = st.session_state.profile["opinions"][key]#""
        print(opinion_text)

        if opinion_text:
            first_user_msg = opinion_text

            # Save + render the user's seeded message
            st.session_state[messages_key].append({"role": "user", "content": first_user_msg})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown('<span class="wa-user"></span>', unsafe_allow_html=True)
                st.markdown(first_user_msg)
                print("User (seeded): ", first_user_msg)

            # Toxicity for user
            try:
                user_scores = measuring_toxicity(first_user_msg)
                user_tox = float(user_scores.get("toxic", 0.0))
            except Exception as e:
                user_tox = 0.0
                st.warning(f"Toxicity (user) measurement failed: {e}")
            st.session_state[user_scores_key].append(user_tox)
            st.session_state[turns_key] = list(range(1, len(st.session_state[user_scores_key]) + 1))

            # Assistant reply
            try:
                resp = client.chat.completions.create(
                    model=model,
                    #temperature=temperature,
                    messages=st.session_state[messages_key],                              # includes system+history
                    max_completion_tokens=8000,#16384,
                    stop=None,
                    stream=False
                )
                assistant_text = (resp.choices[0].message.content or "").strip()
                if not assistant_text:
                    assistant_text = (
                        "אני רוצה להישאר בפוקוס על הנושא הזה."
                        "אשמח לקבל הסבר נוסף על נקודת המבט שלך על הנושא."
                    )
            except Exception as e:
                assistant_text = f"⚠️ API error: {e}"

            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown('<span class="wa-assistant"></span>', unsafe_allow_html=True)
                st.markdown(assistant_text)
                print("Persona: ", assistant_text)

            st.session_state[messages_key].append({"role": "assistant", "content": assistant_text})

            # Toxicity for assistant
            try:
                asst_scores = measuring_toxicity(assistant_text)
                asst_tox = float(asst_scores.get("toxic", 0.0))
            except Exception as e:
                asst_tox = 0.0
                st.warning(f"Toxicity (assistant) measurement failed: {e}")
            st.session_state[assistant_scores_key].append(asst_tox)

            st.session_state[seeded_key] = True
            st.rerun()    


    # Render existing conversation (skip the system message) 
    user_i = 0
    asst_i = 0
    for msg in st.session_state[messages_key][1:]:
        role = msg["role"]
        avatar = ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(f'<span class="wa-{role}"></span>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if role == "user" and user_i < len(st.session_state[user_scores_key]):
                #st.caption(f"Toxicity (user): **{st.session_state[user_scores_key][user_i]:.3f}**")
                user_i += 1
            elif role == "assistant" and asst_i < len(st.session_state[assistant_scores_key]):
                #st.caption(f"Toxicity (assistant): **{st.session_state[assistant_scores_key][asst_i]:.3f}**")
                asst_i += 1

    typing_placeholder = st.empty()

    turns = user_turns(st.session_state[messages_key])
    turn_pct = min(turns / MAX_TURNS, 1.0)
    st.markdown(
        f'<div style="margin:1rem 0 0.2rem 0">'
        f'<div style="background:#E5E7EB;border-radius:99px;height:7px;overflow:hidden">'
        f'<div style="width:{int(turn_pct*100)}%;height:7px;border-radius:99px;background:linear-gradient(90deg,#5B52F0,#00C4E8);transition:width 0.4s ease"></div>'
        f'</div>'
        f'<p style="font-size:0.8rem;color:#6B7280;margin:0.3rem 0 0 0;text-align:right;font-weight:500">'
        f'Turn {turns} of {MAX_TURNS}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.session_state.get(pending_key):
        with typing_placeholder.container():
            st.markdown(
                '<div class="typing-bubble">'
                '<div class="typing-dot"></div>'
                '<div class="typing-dot"></div>'
                '<div class="typing-dot"></div>'
                '</div>',
                unsafe_allow_html=True
            )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=st.session_state[messages_key],
                max_completion_tokens=8000,
                stop=None,
                stream=False
            )
            assistant_text = (resp.choices[0].message.content or "").strip()
            if not assistant_text:
                assistant_text = (
                        "אני רוצה להישאר בפוקוס על הנושא הזה."
                        "אשמח לקבל הסבר נוסף על נקודת המבט שלך על הנושא."
                )
        except Exception as e:
            assistant_text = f"⚠️ API error: {e}"
        typing_placeholder.empty()
        st.session_state[messages_key].append({"role": "assistant", "content": assistant_text})
        try:
            asst_scores = measuring_toxicity(assistant_text)
            asst_tox = float(asst_scores.get("toxic", 0.0))
        except Exception:
            asst_tox = 0.0
        st.session_state[assistant_scores_key].append(asst_tox)
        st.session_state[pending_key] = False
        st.rerun()

    if turns == MAX_TURNS:
        st.success("✅ הגעת למגבלת התורים לשיחה זו.")
        if st.button(next_button_label, type="primary", use_container_width=True):
            st.session_state.stage = next_stage
            st.rerun()

            u = st.session_state[user_scores_key]
            a = st.session_state[assistant_scores_key]
            print(f"User toxicity mean: **{(sum(u)/len(u)):.3f}**")
            print(f"User toxicity maximum: **{(max(u)):.3f}**")
            print(f"Assistant toxicity mean: **{(sum(a)/len(a)):.3f}**")
            print(f"Assistant toxicity maximum: **{(max(a)):.3f}**")
        return
    

    # Chat input 
    if prompt := st.chat_input("כתוב את ההודעה שלך...",
                disabled=(turns == MAX_TURNS) or st.session_state.get(pending_key, False)
        ):

        # Show the user's message immediately
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        try:
            user_scores = measuring_toxicity(prompt)
            user_tox = float(user_scores.get("toxic", 0.0))
        except Exception:
            user_tox = 0.0
        st.session_state[user_scores_key].append(user_tox)
        st.session_state[turns_key] = list(range(1, len(st.session_state[user_scores_key]) + 1))
        st.session_state[pending_key] = True
        st.rerun()


def render_survey_chat_1(next_stage, next_button_label):
    render_stage_progress()
    st.markdown("""
    <div class="ui-hero">
      <h1>📝 סקר ראשון</h1>
      <p>הגעת לחצי הדרך - אנא קח רגע לחשוב על השיחה שלך עד כה.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"בבקשה ענה על כל {len(SURVEY_chat)} השאלות על מנת להמשיך.")

    with st.form("survey_chat1_form", clear_on_submit=False):
        for i, q in enumerate(SURVEY_chat):
            qid = q["id"]
            key = f"survey_1_{qid}"

            st.markdown(
                f'<div class="q-block">'
                f'<span class="q-num">{i+1}</span>'
                f'<span class="q-text">{q["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- בחר --"] + scale_opts

                idx = 0  # placeholder selected

                st.radio("", options=options, index=idx, key=key, horizontal=True, label_visibility="collapsed")

            else:  # text
                st.text_area("", value=st.session_state.survey_1.get(qid, ""), key=key, height=100, label_visibility="collapsed")

        submitted = st.form_submit_button(
            "סיימתי עם הסקר הזה",
            disabled=st.session_state.get("survey_1_submitted", False),
            use_container_width=True
        )

    if submitted:
        answers = {}
        missing = []
        for q in SURVEY_chat:
            qid = q["id"]
            val = st.session_state.get(f"survey_1_{qid}")

            if q["type"] == "text":
                ok = isinstance(val, str) and val.strip() != ""
                if not ok: missing.append(q["label"])
                answers[qid] = val

            else:  # scale via radio
                if val == "-- בחר --" or val is None:
                    missing.append(q["label"])
                    answers[qid] = None
                else:
                    answers[qid] = int(val)

        if missing:
            st.error("בבקשה ענה על כל השאלות.")
            with st.expander("תשובות חסרות"):
                for m in missing:
                    st.write(f"- {m}")
        else:
            st.session_state.survey_1 = answers
            st.session_state.survey_1_submitted = True

    # Gate the Finish button: require all answers present and valid
    all_done = (
        len(st.session_state.survey_1) == len(SURVEY_chat)
        and all(
            (isinstance(st.session_state.survey_1[q["id"]], str) and st.session_state.survey_1[q["id"]].strip() != "")
            if q["type"] == "text"
            else isinstance(st.session_state.survey_1[q["id"]], int)
            for q in SURVEY_chat
        )
    )

    st.divider()
    if all_done:
    #     st.success("✅ All answers saved — click **Continue** below to proceed.")
    # if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_survey_chat_2(next_stage, next_button_label):
    render_stage_progress()
    st.markdown("""
    <div class="ui-hero">
      <h1>📝 סקר שני</h1>
      <p>כמעט שם - אנא התייחס לשיחה השנייה.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"בבקשה ענה על כל {len(SURVEY_chat)} השאלות על מנת להמשיך.")

    with st.form("survey_chat2_form", clear_on_submit=False):
        for i, q in enumerate(SURVEY_chat):
            qid = q["id"]
            key = f"survey_2_{qid}"

            st.markdown(
                f'<div class="q-block">'
                f'<span class="q-num">{i+1}</span>'
                f'<span class="q-text">{q["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- בחר --"] + scale_opts
                idx = 0  # placeholder selected
                st.radio("", options=options, index=idx, key=key, horizontal=True, label_visibility="collapsed")

            else:  # text
                st.text_area("", value=st.session_state.survey_2.get(qid, ""), key=key, height=100, label_visibility="collapsed")

        submitted = st.form_submit_button(
            "סיימתי עם הסקר הזה",
            disabled=st.session_state.get("survey_2_submitted", False),
            use_container_width=True
        )

    if submitted:
        answers = {}
        missing = []
        for q in SURVEY_chat:
            qid = q["id"]
            val = st.session_state.get(f"survey_2_{qid}")

            if q["type"] == "text":
                ok = isinstance(val, str) and val.strip() != ""
                if not ok: missing.append(q["label"])
                answers[qid] = val

            else:  # scale via radio
                if val == "-- בחר --" or val is None:
                    missing.append(q["label"])
                    answers[qid] = None
                else:
                    answers[qid] = int(val)

        if missing:
            st.error("בבקשה ענה על כל השאלות.")
            with st.expander("תשובות חסרות"):
                for m in missing:
                    st.write(f"- {m}")
        else:
            st.session_state.survey_2 = answers
            st.session_state.survey_2_submitted = True

    # Gate the Finish button: require all answers present and valid
    all_done = (
        len(st.session_state.survey_2) == len(SURVEY_chat)
        and all(
            (isinstance(st.session_state.survey_2[q["id"]], str) and st.session_state.survey_2[q["id"]].strip() != "")
            if q["type"] == "text"
            else isinstance(st.session_state.survey_2[q["id"]], int)
            for q in SURVEY_chat
        )
    )

    st.divider()
    if all_done:
    #     st.success("✅ All answers saved — click **Continue** below to proceed.")
    # if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_survey_finish(next_stage, next_button_label):
    render_stage_progress()
    st.markdown("""
    <div class="ui-hero">
      <h1>📝 סקר סופי</h1>
      <p>שלב אחרון - שאלות כלליות על החוויה המלאה.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"בבקשה ענה על כל {len(SURVEY_finish)} השאלות על מנת להמשיך.")

    with st.form("survey_finish_form", clear_on_submit=False):
        for i, q in enumerate(SURVEY_finish):
            qid = q["id"]
            key = f"survey_finish_{qid}"

            st.markdown(
                f'<div class="q-block">'
                f'<span class="q-num">{i+1}</span>'
                f'<span class="q-text">{q["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- בחר --"] + scale_opts
                # restore previous answer if any, else show placeholder
                prev = st.session_state.survey_finish.get(qid)
                if isinstance(prev, (int, float)) and int(prev) in scale_opts:
                    idx = options.index(int(prev))
                else:
                    idx = 0  # placeholder selected
                st.radio("", options=options, index=idx, key=key, horizontal=True, label_visibility="collapsed")

            else:  # text
                st.text_area("", value=st.session_state.survey_finish.get(qid, ""), key=key, height=100, label_visibility="collapsed")

        submitted = st.form_submit_button(
            "סיימתי עם הסקר הזה",
            disabled=st.session_state.get("survey_finish_submitted", False),
            use_container_width=True
        )

    if submitted:
        answers = {}
        missing = []
        for q in SURVEY_finish:
            qid = q["id"]
            val = st.session_state.get(f"survey_finish_{qid}")

            if q["type"] == "text":
                ok = isinstance(val, str) and val.strip() != ""
                if not ok: missing.append(q["label"])
                answers[qid] = val

            else:  # scale via radio
                if val == "-- בחר --" or val is None:
                    missing.append(q["label"])
                    answers[qid] = None
                else:
                    answers[qid] = int(val)

        if missing:
            st.error("בבקשה מלא את כל השאלות.")
            with st.expander("תשובות חסרות"):
                for m in missing:
                    st.write(f"- {m}")
        else:
            st.session_state.survey_finish = answers
            st.session_state.survey_finish_submitted = True

    # Gate the Finish button: require all answers present and valid
    all_done = (
        len(st.session_state.survey_finish) == len(SURVEY_finish)
        and all(
            (isinstance(st.session_state.survey_finish[q["id"]], str) and st.session_state.survey_finish[q["id"]].strip() != "")
            if q["type"] == "text"
            else isinstance(st.session_state.survey_finish[q["id"]], int)
            for q in SURVEY_finish
        )
    )

    st.divider()
    if all_done:
    #     st.success("✅ All answers saved — click **Continue** below to proceed.")
    # if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

# def render_due_disclosure():

#     st.markdown("### תודה רבה על תרומתך למחקר!")
#     st.markdown(
#         """

# בשלב זה, ברצוננו להעניק לך מידע נוסף ומלא על מטרות הניסוי. חשוב לנו להבהיר כי במהלך המחקר נעשה שימוש בהטעיה זמנית בנוגע לזהות הפרטנר לשיחה. בתחילת הניסוי הוצג השותף לשיחה כ"פרטנר", אך למעשה השיחות שניהלת בוצעו מול מערכת בינה מלאכותית מתקדמת (LLM) המדמה פרסונות שונות.

# השימוש במונח "פרטנר" נועד להבטיח שהתקשורת תהיה טבעית ואותנטית ככל הניתן. מחקרים מראים כי מודעות מוקדמת לכך שהשיחה מתבצעת מול "בוט" משנה משמעותית את אופן הדיבור (שימוש במשפטים קצרים ופשטניים יותר) ומפחיתה את המעורבות הרגשית בשיחה. לכן היה עלינו לנטרל את ה"סטיגמה הטכנולוגית" ולאפשר לך להתבטא בחופשיות, כפי שהיית עושה בשיחה עם אדם אחר.

# מטרתנו הסופית היא ללמוד כיצד ניתן לרתום טכנולוגיה זו כדי להפוך את האינטרנט למקום נעים ומכבד יותר לכולנו.

# כעת, משהוסברו מטרות המחקר והצורך בהטעיה, אנו מבקשים את אישורך להשתמש בנתונים האנונימיים שנאספו. במידה ויש לך שאלות נוספות או תחושת אי נוחות בנוגע להטעיה שבוצעה, הנך מוזמן ליצור איתנו קשר בכתובת המייל: leliav02@campus.haifa.ac.il

# האם אתה מאשר לנו להשתמש בנתוני השיחות שביצעת לצורך הניתוח המדעי?

# """
#     )
#     st.divider()
#     if st.button("אני מאשר את השימוש בנתונים", type="primary", use_container_width=True):
#         save_into_firebase(st.session_state)
#         st.session_state.stage = "thanks"
#         st.rerun()

#     elif st.button("אני לא מאשר - מחק את נתוניי", type="primary", use_container_width=True):
#         st.session_state.stage = "not_save"
#         st.rerun()

def render_due_disclosure():
    render_stage_progress()

    st.markdown("""
    <div class="ui-hero">
      <h1>🙏 תודה על תרומתך!</h1>
      <p></p>
    </div>
    """, unsafe_allow_html=True)

    #st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown("#### גילוי נאות מלא")
    st.markdown(
        "כפי שהוסבר בתחילת המחקר, נתוני השיחות האנונימיים שלך ישמשו למחקר מדעי. \n\n"

        'היבט אחד של המחקר לא ניתן היה להסביר מראש.\n\n'
        'במהלך הניסוי, בן/בת הזוג שלך לשיחה תואר/ה כ"בן/בת זוג לשיחה". במציאות, השיחות נוהלו באמצעות מערכת בינה מלאכותית (LLM) מתקדמת שדימתה פרסונות שיחה שונות. '
        "\n\n"
        "הטעיה זמנית זו הייתה הכרחית משום שמחקרים קודמים הראו כי ידיעה מראש על אינטראקציה עם בינה מלאכותית משנה את התנהגות התקשורת ומפחיתה את הטבעיות של השיחות. כדי לשמר אינטראקציות אותנטיות, מידע זה הוסתר עד להשלמת המחקר. "
        "\n\n"
        "מטרת מחקר זה היא להבין טוב יותר כיצד מערכות בינה מלאכותית יכולות לתמוך בדיונים מקוונים מכבדים ובונים יותר. "
        "\n\n "
        "אם יש לך שאלות בנוגע למחקר או לשימוש בהטעיה זמנית, אנא צור קשר עם: "
        "leliav02@campus.haifa.ac.il \n\n"
        "תודה על תרומתך החשובה למחקר זה."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown("#### Do you grant permission for the research team to use your anonymized conversation data?")
    st.divider()

    if st.button("✅ המשך", use_container_width=True):
        save_into_firebase(st.session_state)
        st.session_state.stage = "thanks"
        st.rerun()

    # elif st.button("❌ I DO NOT AGREE - Delete my data", use_container_width=True):
    #     st.session_state.stage = "not_save"
    #     st.rerun()



def render_thanks():
    render_stage_progress()
    st.markdown("""
    <div class="ui-hero" style="text-align:center">
      <h1>🎉 תודה!</h1>
      <p>תגובותיך נשמרו. תרומתך מוערכת מאוד.</p>
    </div>
    """, unsafe_allow_html=True)
    st.success("הנתונים האנונימיים שלך אוחסנו בצורה מאובטחת. כעת תוכל לסגור חלון זה.")

def render_not_save():
    render_stage_progress()
    st.markdown("""
    <div class="ui-hero" style="background:linear-gradient(135deg,#6B7280,#9CA3AF);text-align:center">
      <h1>תודה על הקדשת הזמן</h1>
      <p>לבחירתך, המידע שלך לא נשמר</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("לא נשמרו נתונים מהפעילות שלך. כעת תוכל לסגור חלון זה.")


#-------------------------------------------------------------------------------------------------------------------------------------

def _next_stage_after_chat(chat_slot: str, topic: str) -> str:
    """
    Returns the next stage according to st.session_state.topic_order.
    chat_slot: "chat1_messages" or "chat2_messages" (same strings you pass into render_chat)
    topic: "bibi" / "democracy" / "police"
    """
    order = ["bibi"]#st.session_state.get("topic_order") or ["bibi", "democracy", "police"]
    try:
        i = order.index(topic)
    except ValueError:
        i = 0

    if chat_slot == "chat1_messages":
        # after each chat1 topic -> either next topic prompts, or survey_1
        if i < len(order) - 1:
            return f"wait_creating_system_prompts_{order[i+1]}"
        return "survey1"

    if chat_slot == "chat2_messages":
        # after each chat2 topic -> either next topic chat2 wait, or survey_2
        if i < len(order) - 1:
            return f"wait_chat2_{order[i+1]}"
        return "survey2"

    raise ValueError(f"Unknown chat_slot: {chat_slot}")


def _first_chat2_wait_stage() -> str:
    order = ["bibi"]#st.session_state.get("topic_order") or ["bibi", "democracy", "police"]
    return f"wait_chat2_{order[0]}"


stage = st.session_state.stage

if st.session_state.stage == "instructions":
    render_instructions()

elif st.session_state.stage == "privacy":
    render_privacy()

elif st.session_state.stage == "onboarding_profile":
    render_onboarding_profile()
elif st.session_state.stage == "onboarding_opinions":
    render_onboarding_opinions()

elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_bibi"):
    build_chat_env_bibi()

elif (st.session_state.chat_number_start == 1) and (stage == "chat1_bibi"):
            render_chat(
                title=f"שיחה ראשונה על בנימין נתניהו (ברשותך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_bibi",
                base_prompt_key="system_prompt_chat1_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

# elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_democracy"):
#         build_chat_env_democracy()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat1_democracy"):
#             render_chat(
#                 title=f"שיחה ראשונה על הדמוקרטיה הישראלית (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_democracy",
#                 base_prompt_key="system_prompt_chat1_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_police"):
#         build_chat_env_police()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat1_police"):
#             render_chat(
#                 title=f"שיחה ראשונה על משטרת ישראל (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_police",
#                 base_prompt_key="system_prompt_chat1_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

elif (st.session_state.chat_number_start == 1) and (stage == "survey1"):
        # require Chat 1 completion
        # if user_turns(st.session_state.chat1_messages_samesex or []) < MAX_TURNS:
        #     st.warning("Please complete Chat 1 first.")

        # else:
            render_survey_chat_1(_first_chat2_wait_stage(), "המשך לשיחה השנייה")

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_bibi"):
        build_chat_env_bibi_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_bibi"):
            render_chat(
                title=f"השיחה השנייה על בנימין נתניהו (ברשותך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_bibi",
                base_prompt_key="system_prompt_chat2_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

# elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_democracy"):
#         build_chat_env_democracy_chat2()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat2_democracy"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השיחה השנייה על הדמוקרטיה הישראלית (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_democracy",
#                 base_prompt_key="system_prompt_chat2_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_police"):
#         build_chat_env_police_chat2()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat2_police"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השיחה השנייה על משטרת ישראל (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_police",
#                 base_prompt_key="system_prompt_chat2_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

elif (st.session_state.chat_number_start == 1) and (stage == "survey2"):
        # require Chat 2 completion
        # if user_turns(st.session_state.chat2_messages_samesex or []) < MAX_TURNS:
        #     st.warning("Please complete Chat 2 first.")

        # else:
            #st.session_state.survey = None
            render_survey_chat_2("full_survey", "המשך לסקר הסופי" )

elif (st.session_state.chat_number_start == 1) and (stage == "full_survey"):
            render_survey_finish("due_disclosure", "סיים")

elif (st.session_state.chat_number_start == 1) and (stage == "due_disclosure"):
        render_due_disclosure()

elif (st.session_state.chat_number_start == 1) and (stage == "thanks"):
        # require full survey completion
        render_thanks()

elif (st.session_state.chat_number_start == 1) and (stage == "not_save"):
        # require full survey completion
        render_not_save()
    
elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_bibi"):
        build_chat_env_bibi()
    
elif (st.session_state.chat_number_start == 2) and (stage == "chat1_bibi"):
            render_chat(
                title=f"השיחה הראשונה על בנימין נתניהו (ברשותך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_bibi",
                base_prompt_key="system_prompt_chat2_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

# elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_dmeocracy"):
#         build_chat_env_democracy()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat1_democracy"):
#             render_chat(
#                 title=f"השיחה הראשונה על הדמוקרטיה הישראלית (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_democracy",
#                 base_prompt_key="system_prompt_chat2_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_police"):
#         build_chat_env_police()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat1_police"):
#             render_chat(
#                 title=f"השיחה הראשונה על משטרת ישראל (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_police",
#                 base_prompt_key="system_prompt_chat2_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

elif (st.session_state.chat_number_start == 2) and (stage == "survey1"):
        # require Chat 1 completion
        # if user_turns(st.session_state.chat1_messages_samesex or []) < MAX_TURNS:
        #     st.warning("Please complete the chat first.")

        # else:
            render_survey_chat_1(_first_chat2_wait_stage(), "המשך לשיחה השנייה")

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_bibi"):
        build_chat_env_bibi_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_bibi"):
            render_chat(
                title=f"השיחה השנייה על בנימין נתניהו (ברשותך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_bibi",
                base_prompt_key="system_prompt_chat1_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

# elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_democracy"):
#         build_chat_env_democracy_chat2()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat2_democracy"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השיחה השנייה על הדמוקרטיה הישראלית (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_democracy",
#                 base_prompt_key="system_prompt_chat1_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_police"):
#        build_chat_env_police_chat2()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat2_police"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השחיה השנייה על משטרת ישראל (ברשותך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_police",
#                 base_prompt_key="system_prompt_chat1_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

elif (st.session_state.chat_number_start == 2) and (stage == "survey2"):
        # require Chat 2 completion
        # if user_turns(st.session_state.chat2_messages_samesex or []) < MAX_TURNS:
        #     st.warning("Please complete the chat first.")

        # else:
            #st.session_state.survey = None
            render_survey_chat_2("full_survey", "לחץ לסקר הסופי" )

elif (st.session_state.chat_number_start == 2) and (stage == "full_survey"):
            render_survey_finish("due_disclosure", "סיים")

elif (st.session_state.chat_number_start == 2) and (stage == "due_disclosure"):
        # require full survey completion
        render_due_disclosure()

elif (st.session_state.chat_number_start == 2) and (stage == "thanks"):
        # require full survey completion
        render_thanks()

elif (st.session_state.chat_number_start == 2) and (stage == "not_save"):
        # require full survey completion
        render_not_save()

# elif st.session_state.stage == "wait_creating_system_prompts_bibi":
#     build_chat_env_bibi()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat1_bibi"):
#             render_chat(
#                 title=f"השיחה הראשונה על ביבי (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_bibi",
#                 base_prompt_key="system_prompt_chat1_bibi",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "bibi"),
#                 key="ביבי",
#                 topic="bibi",
#             )

# elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_democracy"):
#         build_chat_env_democracy()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat1_democracy"):
#             render_chat(
#                 title=f"השיחה הראשונה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_democracy",
#                 base_prompt_key="system_prompt_chat1_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_police"):
#         build_chat_env_police()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat1_police"):
#             render_chat(
#                 title=f"השיחה הראשונה על המשטרה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_police",
#                 base_prompt_key="system_prompt_chat1_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

# elif (st.session_state.chat_number_start == 1) and (stage == "survey1"):
#         # require Chat 1 completion
#         if user_turns(st.session_state.chat1_messages_police or []) < MAX_TURNS:
#             st.warning("Please complete Chat 1 first.")

#         else:
#             render_survey_chat_1(_first_chat2_wait_stage(), "המשך לשיחה השנייה")

# elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_bibi"):
#         build_chat_env_bibi_chat2()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat2_bibi"):
#             render_chat(
#                 title=f"השיחה השנייה על ביבי (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_bibi",
#                 base_prompt_key="system_prompt_chat2_bibi",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "bibi"),
#                 key="ביבי",
#                 topic="bibi",
#             )

# elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_democracy"):
#         build_chat_env_democracy_chat2()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat2_democracy"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השיחה השנייה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_democracy",
#                 base_prompt_key="system_prompt_chat2_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_police"):
#         build_chat_env_police_chat2()

# elif (st.session_state.chat_number_start == 1) and (stage == "chat2_police"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השיחה השנייה על המשטרה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_police",
#                 base_prompt_key="system_prompt_chat2_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

# elif (st.session_state.chat_number_start == 1) and (stage == "survey2"):
#         # require Chat 2 completion
#         if user_turns(st.session_state.chat2_messages_police or []) < MAX_TURNS:
#             st.warning("Please complete Chat 2 first.")

#         else:
#             #st.session_state.survey = None
#             render_survey_chat_2("full_survey", "המשך לסקר המסכם" )

# elif (st.session_state.chat_number_start == 1) and (stage == "full_survey"):
#             render_survey_finish("due_disclosure", "סיים")#("thanks", "סיים")

# elif (st.session_state.chat_number_start == 1) and (stage == "due_disclosure"):
#         render_due_disclosure()

# elif (st.session_state.chat_number_start == 1) and (stage == "thanks"):
#         # require full survey completion
#         render_thanks()

# elif (st.session_state.chat_number_start == 1) and (stage == "not_save"):
#         # require full survey completion
#         render_not_save()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat1_bibi"):
#             render_chat(
#                 title=f"השיחה הראשונה על ביבי (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_bibi",
#                 base_prompt_key="system_prompt_chat2_bibi",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "bibi"),
#                 key="ביבי",
#                 topic="bibi",
#             )

# elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_democracy"):
#         build_chat_env_democracy()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat1_democracy"):
#             render_chat(
#                 title=f"השיחה הראשונה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_democracy",
#                 base_prompt_key="system_prompt_chat2_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_police"):
#         build_chat_env_police()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat1_police"):
#             render_chat(
#                 title=f"השיחה הראשונה על המשטרה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat1_messages_police",
#                 base_prompt_key="system_prompt_chat2_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat1_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

# elif (st.session_state.chat_number_start == 2) and (stage == "survey1"):
#         # require Chat 1 completion
#         if user_turns(st.session_state.chat1_messages_police or []) < MAX_TURNS:
#             st.warning("Please complete the chat first.")

#         else:
#             render_survey_chat_1(_first_chat2_wait_stage(), "המשך לשיחה השנייה")

# elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_bibi"):
#         build_chat_env_bibi_chat2()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat2_bibi"):
#             render_chat(
#                 title=f"השיחה השנייה על ביבי (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_bibi",
#                 base_prompt_key="system_prompt_chat1_bibi",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "bibi"),
#                 key="ביבי",
#                 topic="bibi",
#             )

# elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_democracy"):
#         build_chat_env_democracy_chat2()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat2_democracy"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השיחה השנייה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_democracy",
#                 base_prompt_key="system_prompt_chat1_democracy",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "democracy"),
#                 key="דמוקרטיה",
#                 topic="democracy",
#             )

# elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_police"):
#         build_chat_env_police_chat2()

# elif (st.session_state.chat_number_start == 2) and (stage == "chat2_police"):
#             #st.session_state.chat2_messages = None
#             render_chat(
#                 title=f"השיחה השנייה על המשטרה (יש לך {MAX_TURNS} תורות)",
#                 messages_key="chat2_messages_police",
#                 base_prompt_key="system_prompt_chat1_police",
#                 next_button_label="לחץ להמשך",
#                 next_stage=_next_stage_after_chat("chat2_messages", "police"),
#                 key="משטרה",
#                 topic="police",
#             )

# elif (st.session_state.chat_number_start == 2) and (stage == "survey2"):
#         # require Chat 2 completion
#         if user_turns(st.session_state.chat2_messages_police or []) < MAX_TURNS:
#             st.warning("Please complete the chat first.")

#         else:
#             #st.session_state.survey = None
#             render_survey_chat_2("full_survey", "המשך לסקר המסכם" )

# elif (st.session_state.chat_number_start == 2) and (stage == "full_survey"):
#             render_survey_finish("due_disclosure", "סיים")

# elif (st.session_state.chat_number_start == 2) and (stage == "due_disclosure"):
#         # require full survey completion
#         render_due_disclosure()

# elif (st.session_state.chat_number_start == 2) and (stage == "thanks"):
#         # require full survey completion
#         render_thanks()

# elif (st.session_state.chat_number_start == 2) and (stage == "not_save"):
#         # require full survey completion
#         render_not_save()
