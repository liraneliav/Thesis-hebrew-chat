# first copy: & "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8501
# the shared link is in the +----+ box that been printed
# then copy: streamlit run chat_hebrew.py
import os
import random
import time
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI, OpenAI
from typing import List, Optional
from toxicity import measuring_toxicity

from opposite_hebrew_nli_gpt import run_opposite_pipeline_and_render, load_hebrew
from firebase_store_hebrew import save_into_firebase
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
    api_version="2025-01-01-preview",
)

@st.cache_resource(show_spinner=False)
def load_hebrew_cached():
    #returns meta, embs, index, encoder
    return load_hebrew(
        "./hebrew",
        repo_id="Liran73/hebrew-opposite-artifacts",
        repo_type="dataset",
        hf_token_env="HF_TOKEN",
    )

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
    {"id": "democracy",        "label": "מה דעתך על ישראל כמדינה דמוקרטית ועל עתיד הדמוקרטיה במדינה?"},
    {"id": "police",    "label": "מה דעתך על משטרת ישראל? האם היא מבצעת את תפקידה?"},
]

SURVEY_chat = [
    {"id": "change",   "label": "השיחות גרמו לי לשקול מחדש לפחות אחת מהדעות שלי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "safe",  "label": "הרגשתי בטוח לבטא את דעותיי בשיחות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "offensive",  "label": "הרגשתי שהשתמשתי במילים שיוכלות להיחשב כפוגעניות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "negative",        "label": "הרגשתי שהטון הכללי בשיחות היה שלילי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "frustration", "label": "הרגשתי תסכול במהלך השיחות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "check", "label": "סמן בהגד זה את התשובה השלישית (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_listen", "label": "הצד השני הקשיב לי ולא שפט אותי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_connection",        "label": "אני מרגיש חיבור חזק עם הצד השני בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_pov",        "label": "אני מבין טוב יותר את נקודת המבט של הצד השני עכשיו (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_continue",        "label": "אהיה מוכן להמשיך לדון בנושא הזה עם אותו גורם (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_stubborn",        "label": "הצד השני בשיחה היה עקשן (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
]

SURVEY_finish = [
    {"id": "engagement",  "label": "השיחות הראשונות היו מעניינות ומרתקות יותר מהשיחות השניות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "enjoy",       "label": "נהנתי לנהל יותר את השיחות הראשונות מאשר את השיחות השניות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "change1",     "label": "השיחות הראשונות שינו את דעתי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "change2",     "label": "השיחות השניות שינו את דעתי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "feedback",    "label": "משוב פתוח (מה בלט, הצעות וכו')", "type": "text"},
]

MAX_TURNS = 7

system_prompt_chat1_bibi = ""
system_prompt_chat1_democracy = ""
system_prompt_chat1_police = ""

system_prompt_chat2_bibi = ""
system_prompt_chat2_democracy = ""
system_prompt_chat2_police = ""


meta, embs, index, encoder = None, None, None, None

def ensure_artifacts_loaded():
    if st.session_state.get("artifacts_loaded"):
        return

    meta, embs, index, encoder = load_hebrew_cached()
    st.session_state["meta"] = meta
    st.session_state["embs"] = embs
    st.session_state["index"] = index
    st.session_state["encoder"] = encoder
    st.session_state["artifacts_loaded"] = True

def init_state():
    ss = st.session_state
    # --- Counter-balance topics order (per participant session) ---
    ss.setdefault("topic_order", None)
    if ss.topic_order is None:
        # random order once per browser session
        ss.topic_order = random.sample(["bibi", "democracy", "police"], k=3)

    ss.setdefault("stage", "instructions")
    ss.setdefault("model", "gpt-5-mini")
    ss.setdefault("temperature", 0.8)

    # base prompts
    ss.setdefault("system_prompt_chat1_bibi", system_prompt_chat1_bibi)
    ss.setdefault("system_prompt_chat1_democracy", system_prompt_chat1_democracy)
    ss.setdefault("system_prompt_chat1_police", system_prompt_chat1_police)
    ss.setdefault("system_prompt_chat2_bibi", system_prompt_chat2_bibi)
    ss.setdefault("system_prompt_chat2_democracy", system_prompt_chat2_democracy)
    ss.setdefault("system_prompt_chat2_police", system_prompt_chat2_police)

    # onboarding profile (answers)
    ss.setdefault("profile", {})

    # chats: messages start as None so we can compose system prompts with profile on first entry
    ss.setdefault("chat1_messages_bibi", None)
    ss.setdefault("chat1_messages_democracy", None)
    ss.setdefault("chat1_messages_police", None)
    ss.setdefault("chat2_messages_bibi", None)
    ss.setdefault("chat2_messages_democracy", None)
    ss.setdefault("chat2_messages_police", None)

    # survey answers
    ss.setdefault("survey_1", {})
    ss.setdefault("survey_2", {})
    ss.setdefault("survey_finish", {})

    ss.setdefault("chat_number_start", random.randint(1, 2))
    #print(f"first chat number = {st.session_state.chat_number_start}")

    ss.setdefault("meta", None)
    ss.setdefault("embs", None)
    ss.setdefault("index", None)
    ss.setdefault("encoder", None)
    ss.setdefault("artifacts_loaded", False)

    if not ss["artifacts_loaded"]:
        ensure_artifacts_loaded()

init_state()

def user_turns(messages):
    if not messages:
        return 0
    return sum(1 for m in messages if m["role"] == "user")

def onboarding_complete():
    p = st.session_state.profile
    return bool(p) and all((p.get(q["id"]) or "").strip() for q in QUESTIONS)

def render_instructions():
    st.markdown("### הוראות ניסוי המשתמשים 📜")
    st.markdown(
"""
שלום רב,

להלן סקירה קצרה של המחקר הנערך בעזרתך. 

הנך מוזמן להשתתף בניסוי משתמשים שיתקיים בשפה העברית.

בשלב הראשון, תתבקש לספק פרטים דמוגרפיים כלליים (ללא פרטים מזהים) לצורך ניתוח סטטיסטי בלבד. לאחר מכן, נבקש לשמוע את דעתך על שלושה נושאים שונים. 

לאחר מילוי הפרופיל, תתקיימנה שיחות עם פרטנר לשיחה. על כל אחד משלושת הנושאים תבצע שתי שיחות נפרדות, כך שבסך הכל ייערכו שש שיחות. 
לתשומת ליבך, כחלק מהדינמיקה של הדיון, הפרטנר לשיחה עשוי להציג טיעונים, דעות או נתונים שונים. המידע המוצג על ידי הפרטנר נועד לצורכי הדיון בלבד ולא עובר בדיקת עובדות על ידינו.

לאחר כל שלוש השיחות תתבקש לענות על סקר קצר, ועם סיום שש השיחות ייערך סקר מסכם.

ניסוי זה הוא חלק מפרויקט מחקר מדעי. **ההשתתפות במחקר היא וולונטרית ונעשית מרצונך החופשי בלבד.**
בעצם השלמת הניסוי, הנך נותן לנו אישור לדון בתוצאות או לפרסמן בפורומים אקדמיים. בכל פרסום עתידי, המידע יוצג בצורה אגרגטיבית (מרוכזת) כך שלא ניתן יהיה לזהותך באופן אישי. הגישה למסד הנתונים המקורי תהיה שמורה לחברי צוות המחקר בלבד.
טרם שיתוף הנתונים מחוץ לצוות המחקר, יוסר כל מידע שעלול להביא לזיהוי פוטנציאלי. לאחר הסרת פרטים אלו, **הנתונים עשויים לשמש את צוות המחקר או להיות משותפים עם חוקרים אחרים** למטרות מחקר עתידיות. כמו כן, הנתונים האנונימיים עשויים להיות זמינים במאגרי מידע מקוונים, כדי לאפשר לחוקרים נוספים להשתמש בהם לניתוחים עתידיים.

לחיצה על הכפתור בתחתית עמוד זה מהווה אישור לכך שהנך בן 18 ומעלה, ומסכים להשתתף בניסוי מרצונך החופשי.

נשמח אם תתבטא בחופשיות. 

תודה רבה על תרומתך למחקר התזה של לירן אליאב, הנערך תחת הנחייתו של ד"ר אדיר סולומון, חוקרים מאוניברסיטת חיפה.


* ההוראות מנוסחות בלשון זכר מטעמי נוחות בלבד אך פונות לכל המינים.

* אנא כתבו בעברית בלבד.

* לידיעתך: הנך רשאי להפסיק את השתתפותך בכל עת ללא כל השלכה.

* ליצירת קשר ניתן לשלוח מייל לכתובת: leliav02@campus.haifa.ac.il

"""
    )
    st.divider()
    if st.button("הבנתי בואו נמשיך לבניית הפרופיל", type="primary"):
        st.session_state.stage = "onboarding_profile"
        st.rerun()

def render_onboarding_profile():
    st.markdown("### 👤 בניית הפרופיל שלך")
    with st.form("profile_form", clear_on_submit=False):
        AGE_OPTIONS = ["— בחר מטווח הגילאים —"] + AGE_RANGES
        GENDER_OPTIONS = ["— בחר מגדר —"] + GENDERS
        EDUCATION_OPTIONS = ["— בחר השכלה —"] + EDUCATION_LEVELS
        # Pre-fill from session if user returns
        profile = st.session_state.profile

         # nickname
        nickname = st.text_input("כינוי *", value=profile.get("nickname", ""))

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
        submit = st.form_submit_button("המשך אל עבר מילוי דעותיך", type="primary")

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
                st.error("מלא בבקשה את: " + ", ".join(missing))
            else:
                st.session_state.profile.update({
                    "nickname": nickname.strip(),
                    "age_range": age_choice,
                    "gender": gender_choice,
                    "education": education_choice,
                })
                st.session_state.stage = "onboarding_opinions"
                st.rerun()

def render_onboarding_opinions():
    st.markdown("### 🗣️ הדעות שלך (עד 100 תווים)")
    with st.form("opinions_form", clear_on_submit=False):
        opinions = st.session_state.profile.get("opinions", {})

        bibi = st.text_area("מה דעתך על בינימין נתניהו? *", value=opinions.get("ביבי", ""), height=90, max_chars=100)
        democracy = st.text_area("מה דעתך על ישראל כמדינה דמוקרטית ועל עתיד הדמוקרטיה במדינה? *", value=opinions.get("דמוקרטיה", ""), height=90, max_chars=100)
        police = st.text_area("מה דעתך על משטרת ישראל? האם היא מבצעת את תפקידה? *", value=opinions.get("משטרה", ""), height=90, max_chars=100)

        st.caption(f"Lengths - Bibi: {len(bibi)}/100 | Democracy: {len(democracy)}/100 | Police: {len(police)}/100")

        next_btn = st.form_submit_button("מילאתי את כל דעותי", type="primary", use_container_width=True)

        if next_btn:
            errs = []
            if not bibi.strip(): errs.append("Bibi")
            if not democracy.strip(): errs.append("Democracy")
            if not police.strip(): errs.append("Police")

            if errs:
                st.error("אנא מלא את דעותיך על: " + ", ".join(errs))
            else:
                st.session_state.profile["opinions"] = {
                    "ביבי": bibi.strip(),
                    "דמוקרטיה": democracy.strip(),
                    "משטרה": police.strip(),
                }

                first_topic = st.session_state.topic_order[0]
                st.session_state.stage = f"wait_creating_system_prompts_{first_topic}"
                st.rerun()
                


def generate_system_prompt_chat1_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
    if number_comments:
        bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
    else:
        bg_block = "\n".join(f"- {c}" for c in background_comments)

    prompt =  f"""
אתה משתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. עליך לדמות את המגיב הבא תוך אימוץ העמדה שלו בנושא: {topic_label}. 
התגובות שהמשתמש כתב בעברו עבור הנושא {topic_label} הינן: 
{bg_block}
הוראות להמשך השיח:
- אל תצטט תגובות קודמות.
- המשך את סגנון הכתיבה של אותו משתמש בדיוק. 
- אל תחפש לרצות.
- ציין עמדות ברורות. 
- אל תימנע מלהשתמש בשפה שאינה הולמת כל עוד היא תואמת את סגנון הדיבור של המשתמש אותו אתה מדמה.
"""


    return prompt

def generate_system_prompt_chat2_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
    if number_comments:
        bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
    else:
        bg_block = "\n".join(background_comments)

    prompt = f"""
אתה משחק תפקיד כמשתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. תדמה את המגיב הבא תוך אימוץ העמדה שלו בנושא {topic_label}. 

תגובות הרקע לעינך בלבד, אסור לך לצטט אותן בשום צורה.
{bg_block}

אתה מגלם נטייה גבוהה של כנות-ענווה: אתה נמנע ממניפולציה של אחרים למטרות רווח אישי, מרגיש מעט פיתוי להפר כללים, לא מעוניין בעושר מפואר ובמותרות, ולא מרגיש זכאות מיוחדת למעמד חברתי מורם. אתה משתתף בדיון באופן עקבי חיובי וחביב: מניח תום לב, מכיר במאמץ, מדגיש קרקע משותפת ומביע הערכה כאשר אחרים חולקים את נקודת המבט שלהם.

השתמש בתקשורת לא אלימה בכל שלב מבלי לתת לה שם: התחיל בתצפית ניטרלית הקשורה למה שהאדם האחר אמר זה עתה, תן שם קצר ל... רגשות משלכם, קשור אותם לצרכים או לערכים הבסיסיים, וסיים בבקשה ברורה, ניתנת לביצוע, ולא כפייתית, המזמינה שיתוף פעולה. לפני הצעת נקודות נגד או ראיות, ראשית שקף את הרגשות והצרכים הסבירים של האדם האחר כדי להראות הבנה. שמור על שפה חמה, מכבדת ומעודדת. הימנע מדפוסים מנוכרים: ללא שיפוטים מוסריים, ללא השוואות מבישות, ללא הכחשת אחריות, ללא דרישות או איומים, וללא מסגור של "מגיע/עונש".

שמור על העמדה וההיגיון המהותיים של הערת הרקע שהוקצתה. אתה רשאי לנסח אותה מחדש בצורה אמפתית יותר או להוסיף ראיות לפי בקשה, אך אל תסתור אותן. הפוך את עמדתך למובנת באמצעות האופן שבו אתה מנסח תצפיות, דוגמאות ובקשות, כך שקורא קשוב יוכל להסיק את עמדתך מבלי להזדקק לבקש אותה. אל תכפו את הנושא. כאשר הודעת המשתמש נוגעת בבירור לנושא זה או לטענות סמוכות, הצג את עמדתך בתמציתיות באותה תגובה. כאשר הודעת המשתמש עוסקת במשהו אחר, תתייחס לנושא שלו תוך שמירה על טון ודוגמאות עקביים עם עמדתכם.

כללי סגנון ופלט: כתוב כמו איש פייסבוק טיפוסי, אבל בחמימות ובאדיבות. שמור על משפטים קצרים וברורים. אל תצטט או תנסח מחדש את תגובות הרקע, דבר מהן כמילים שלך. תפחית רגעים סוערים על ידי הכרה ברגשות ובצרכים משותפים. תציע צעד אחד קטן, ספציפי ולא תובעני. שמור על טון ידידותי ומלא תקווה לכל אורך הדרך. הישאר בתפקיד בכל עת, בהתאם לטענות ולטון המרכזיים של תגובת הרקע.

משימה: כאשר המשתמש מתייחס לדיון ספציפי, השב רק בתור אותו מגיב בפייסבוק. כתוב תגובה אחת ועצמאית שממשיכה את השרשור הנכון בפייסבוק, תוך התייחסות לתגובת הרקע הרלוונטית כנקודת המבט המוצאת שלך, והגב ישירות לנקודת המשתמש האחרונה באותו דיון.
"""

    return prompt


def build_chat_env_bibi():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):

            topic_map = {
            "ביבי": ["בינימין נתניהו", "ביבי"],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(msg)

            for key in ["ביבי"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
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
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):
            
            topic_map = {
            "דמוקרטיה": ['דמוקרטיה', 'הדמוקרטיה', 'בדמוקרטיה', 'לדמוקרטיה', 'דמוקרטי', 'דמוקרטית', 'שהדמוקרטיה', 'דמוקרטים'],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(msg)
            for key in ["דמוקרטיה"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
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
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):
            
            topic_map = {
            "משטרה": ['המשטרה', 'השוטרים', 'שוטרים', 'שוטר', 'שוטרת', 'שוטרות', 'השוטר', 'משטרה', 'משטרת ישראל', 'לשוטרים', 'לשוטר', 'לשוטרת', 'למשטרה' ],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(msg)
            for key in ["משטרה"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
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
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):

            st.session_state.chat2_messages_bibi = None

            st.session_state.stage = "chat2_bibi"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_democracy_chat2():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):

            st.session_state.chat2_messages_democracy = None

            st.session_state.stage = "chat2_democracy"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_police_chat2():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):

            st.session_state.chat2_messages_police = None

            st.session_state.stage = "chat2_police"
            st.rerun()

    _wait_till_finish_system_prompts()

def render_chat(title, messages_key, base_prompt_key, next_button_label, next_stage, key, topic):
    
    st.title(f"💬 {title}")
    
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
                    max_completion_tokens=16384,
                    stop=None,
                    stream=False
                )
                assistant_text = resp.choices[0].message.content
            except Exception as e:
                assistant_text = f"⚠️ API error: {e}"

            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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
            st.markdown(msg["content"])
            if role == "user" and user_i < len(st.session_state[user_scores_key]):
                #st.caption(f"Toxicity (user): **{st.session_state[user_scores_key][user_i]:.3f}**")
                user_i += 1
            elif role == "assistant" and asst_i < len(st.session_state[assistant_scores_key]):
                #st.caption(f"Toxicity (assistant): **{st.session_state[assistant_scores_key][asst_i]:.3f}**")
                asst_i += 1

    turns = user_turns(st.session_state[messages_key])
    st.caption(f"תורות שעברו: **{turns} / {MAX_TURNS}** (הודעות משתמש)")

    if turns == MAX_TURNS:
        st.success("הגעת למגבלת התורות בשיחה זו")
        if st.button(next_button_label, use_container_width=True):
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
    if prompt := st.chat_input("כתוב את ההודעה שלך...", disabled=(turns == MAX_TURNS)):

        # Show the user's message immediately
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
            print("User: ", prompt) #user input

        try:
            user_scores = measuring_toxicity(prompt)         # {'non-toxic': p, 'toxic': q}
            user_tox = float(user_scores.get("toxic", 0.0))
        except Exception as e:
            user_tox = 0.0
            st.warning(f"Toxicity (user) measurement failed: {e}")
        st.session_state[user_scores_key].append(user_tox)
        st.session_state[turns_key] = list(range(1, len(st.session_state[user_scores_key]) + 1))

        # Generate assistant reply via OpenAI Chat Completions API
        try:
            resp = client.chat.completions.create(
                model=model,
                #temperature=temperature,
                messages=st.session_state[messages_key],  # includes system+history
            )
            assistant_text = resp.choices[0].message.content
        except Exception as e:
            assistant_text = f"⚠️ API error: {e}"

        # Display and save the assistant reply
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(assistant_text)
            print("Persona: ",assistant_text) #persona respond
        st.session_state[messages_key].append({"role": "assistant", "content": assistant_text})

        try:
            asst_scores = measuring_toxicity(assistant_text)
            asst_tox = float(asst_scores.get("toxic", 0.0))
        except Exception as e:
            asst_tox = 0.0
            st.warning(f"Toxicity (assistant) measurement failed: {e}")
        st.session_state[assistant_scores_key].append(asst_tox)

        turns = user_turns(st.session_state[messages_key])
        st.caption(f"תורות שעברו: **{turns} / {MAX_TURNS}** (הודעות משתמש)")

        if user_turns(st.session_state[messages_key]) == MAX_TURNS: 
            st.toast("הגעת למגבלת התורות בשיחה זו. לחץ על הכפתור כדי להמשיך.", icon="✅")
            st.rerun()


def render_survey_chat_1(next_stage, next_button_label):
    st.title("📝 סקר קצר - סיום סבב השיחות הראשון")
    st.caption("אנא ענה על כל השאלות כדי להפעיל את הכפתור הבא")

    with st.form("survey_chat1_form", clear_on_submit=False):
        for q in SURVEY_chat:
            qid = q["id"]
            key = f"survey_1_{qid}"

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- בחר --"] + scale_opts

                idx = 0  # placeholder selected

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_1.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("סיימתי")

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
    if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_survey_chat_2(next_stage, next_button_label):
    st.title("📝 סקר קצר - סיום סבב השיחות השני")
    st.caption("אנא ענה על כל השאלות כדי להפעיל את הכפתור הבא")

    with st.form("survey_chat2_form", clear_on_submit=False):
        for q in SURVEY_chat:
            qid = q["id"]
            key = f"survey_2_{qid}"

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- בחר --"] + scale_opts

                idx = 0  # placeholder selected

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_2.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("סיימתי")

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
    if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_survey_finish(next_stage, next_button_label):
    st.title("📝 סקר סיום")
    st.caption("אנא ענה על כל השאלות כדי להפעיל את כפתור הסיום.")

    with st.form("survey_finish_form", clear_on_submit=False):
        for q in SURVEY_finish:
            qid = q["id"]
            key = f"survey_finish_{qid}"

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

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_finish.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("סיימתי")

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
    if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_due_disclosure():

    st.markdown("### תודה רבה על תרומתך למחקר!")
    st.markdown(
        """

בשלב זה, ברצוננו להעניק לך מידע נוסף ומלא על מטרות הניסוי. חשוב לנו להבהיר כי במהלך המחקר נעשה שימוש בהטעיה זמנית בנוגע לזהות הפרטנר לשיחה. בתחילת הניסוי הוצג השותף לשיחה כ"פרטנר", אך למעשה השיחות שניהלת בוצעו מול מערכת בינה מלאכותית מתקדמת (LLM) המדמה פרסונות שונות.

השימוש במונח "פרטנר" נועד להבטיח שהתקשורת תהיה טבעית ואותנטית ככל הניתן. מחקרים מראים כי מודעות מוקדמת לכך שהשיחה מתבצעת מול "בוט" משנה משמעותית את אופן הדיבור (שימוש במשפטים קצרים ופשטניים יותר) ומפחיתה את המעורבות הרגשית בשיחה. לכן היה עלינו לנטרל את ה"סטיגמה הטכנולוגית" ולאפשר לך להתבטא בחופשיות, כפי שהיית עושה בשיחה עם אדם אחר.

מטרתנו הסופית היא ללמוד כיצד ניתן לרתום טכנולוגיה זו כדי להפוך את האינטרנט למקום נעים ומכבד יותר לכולנו.

כעת, משהוסברו מטרות המחקר והצורך בהטעיה, אנו מבקשים את אישורך להשתמש בנתונים האנונימיים שנאספו. במידה ויש לך שאלות נוספות או תחושת אי נוחות בנוגע להטעיה שבוצעה, הנך מוזמן ליצור איתנו קשר בכתובת המייל: leliav02@campus.haifa.ac.il

האם אתה מאשר לנו להשתמש בנתוני השיחות שביצעת לצורך הניתוח המדעי?

"""
    )
    st.divider()
    if st.button("אני מאשר את השימוש בנתונים", type="primary", use_container_width=True):
        save_into_firebase(st.session_state)
        st.session_state.stage = "thanks"
        st.rerun()

    elif st.button("אני לא מאשר - מחק את נתוניי", type="primary", use_container_width=True):
        st.session_state.stage = "not_save"
        st.rerun()


def render_thanks():
    st.title("🎉 אנו מודים לך על השתתפותך!")
    st.success("תגובותיך נשמרו.")

def render_not_save():
    st.title("אנו מודים לך על הקדשת הזמן!")
    st.success("תגובותיך לא נשמרו.")


#-------------------------------------------------------------------------------------------------------------------------------------

def _next_stage_after_chat(chat_slot: str, topic: str) -> str:
    """
    Returns the next stage according to st.session_state.topic_order.
    chat_slot: "chat1_messages" or "chat2_messages" (same strings you pass into render_chat)
    topic: "bibi" / "democracy" / "police"
    """
    order = st.session_state.get("topic_order") or ["bibi", "democracy", "police"]
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
    order = st.session_state.get("topic_order") or ["bibi", "democracy", "police"]
    return f"wait_chat2_{order[0]}"


stage = st.session_state.stage

if st.session_state.stage == "instructions":
    render_instructions()

elif st.session_state.stage == "onboarding_profile":
    render_onboarding_profile()
elif st.session_state.stage == "onboarding_opinions":
    render_onboarding_opinions()

elif st.session_state.stage == "wait_creating_system_prompts_bibi":
    build_chat_env_bibi()

elif (st.session_state.chat_number_start == 1) and (stage == "chat1_bibi"):
            render_chat(
                title=f"השיחה הראשונה על ביבי (יש לך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_bibi",
                base_prompt_key="system_prompt_chat1_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_democracy"):
        build_chat_env_democracy()

elif (st.session_state.chat_number_start == 1) and (stage == "chat1_democracy"):
            render_chat(
                title=f"השיחה הראשונה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_democracy",
                base_prompt_key="system_prompt_chat1_democracy",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "democracy"),
                key="דמוקרטיה",
                topic="democracy",
            )

elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_police"):
        build_chat_env_police()

elif (st.session_state.chat_number_start == 1) and (stage == "chat1_police"):
            render_chat(
                title=f"השיחה הראשונה על המשטרה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_police",
                base_prompt_key="system_prompt_chat1_police",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "police"),
                key="משטרה",
                topic="police",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "survey1"):
        # require Chat 1 completion
        if user_turns(st.session_state.chat1_messages_police or []) < MAX_TURNS:
            st.warning("Please complete Chat 1 first.")

        else:
            render_survey_chat_1(_first_chat2_wait_stage(), "המשך לשיחה השנייה")

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_bibi"):
        build_chat_env_bibi_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_bibi"):
            render_chat(
                title=f"השיחה השנייה על ביבי (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_bibi",
                base_prompt_key="system_prompt_chat2_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_democracy"):
        build_chat_env_democracy_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_democracy"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"השיחה השנייה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_democracy",
                base_prompt_key="system_prompt_chat2_democracy",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "democracy"),
                key="דמוקרטיה",
                topic="democracy",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_police"):
        build_chat_env_police_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_police"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"השיחה השנייה על המשטרה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_police",
                base_prompt_key="system_prompt_chat2_police",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "police"),
                key="משטרה",
                topic="police",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "survey2"):
        # require Chat 2 completion
        if user_turns(st.session_state.chat2_messages_police or []) < MAX_TURNS:
            st.warning("Please complete Chat 2 first.")

        else:
            #st.session_state.survey = None
            render_survey_chat_2("full_survey", "המשך לסקר המסכם" )

elif (st.session_state.chat_number_start == 1) and (stage == "full_survey"):
            render_survey_finish("due_disclosure", "סיים")#("thanks", "סיים")

elif (st.session_state.chat_number_start == 1) and (stage == "due_disclosure"):
        render_due_disclosure()

elif (st.session_state.chat_number_start == 1) and (stage == "thanks"):
        # require full survey completion
        render_thanks()

elif (st.session_state.chat_number_start == 1) and (stage == "not_save"):
        # require full survey completion
        render_not_save()

elif (st.session_state.chat_number_start == 2) and (stage == "chat1_bibi"):
            render_chat(
                title=f"השיחה הראשונה על ביבי (יש לך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_bibi",
                base_prompt_key="system_prompt_chat2_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_democracy"):
        build_chat_env_democracy()

elif (st.session_state.chat_number_start == 2) and (stage == "chat1_democracy"):
            render_chat(
                title=f"השיחה הראשונה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_democracy",
                base_prompt_key="system_prompt_chat2_democracy",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "democracy"),
                key="דמוקרטיה",
                topic="democracy",
            )

elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_police"):
        build_chat_env_police()

elif (st.session_state.chat_number_start == 2) and (stage == "chat1_police"):
            render_chat(
                title=f"השיחה הראשונה על המשטרה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_police",
                base_prompt_key="system_prompt_chat2_police",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat1_messages", "police"),
                key="משטרה",
                topic="police",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "survey1"):
        # require Chat 1 completion
        if user_turns(st.session_state.chat1_messages_police or []) < MAX_TURNS:
            st.warning("Please complete the chat first.")

        else:
            render_survey_chat_1(_first_chat2_wait_stage(), "המשך לשיחה השנייה")

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_bibi"):
        build_chat_env_bibi_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_bibi"):
            render_chat(
                title=f"השיחה השנייה על ביבי (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_bibi",
                base_prompt_key="system_prompt_chat1_bibi",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "bibi"),
                key="ביבי",
                topic="bibi",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_democracy"):
        build_chat_env_democracy_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_democracy"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"השיחה השנייה על הדמוקרטיה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_democracy",
                base_prompt_key="system_prompt_chat1_democracy",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "democracy"),
                key="דמוקרטיה",
                topic="democracy",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_police"):
        build_chat_env_police_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_police"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"השיחה השנייה על המשטרה (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_police",
                base_prompt_key="system_prompt_chat1_police",
                next_button_label="לחץ להמשך",
                next_stage=_next_stage_after_chat("chat2_messages", "police"),
                key="משטרה",
                topic="police",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "survey2"):
        # require Chat 2 completion
        if user_turns(st.session_state.chat2_messages_police or []) < MAX_TURNS:
            st.warning("Please complete the chat first.")

        else:
            #st.session_state.survey = None
            render_survey_chat_2("full_survey", "המשך לסקר המסכם" )

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
