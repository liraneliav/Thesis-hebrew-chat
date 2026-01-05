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
from toxicity import measuring_toxicity, plot_toxicity
#from opposite import most_opposite_in_topic, load, load_hebrew, most_opposite_in_topic_hebrew
#from opposite_hebrew_nli import load_hebrew, most_opposite_in_topic_hebrew_with_nli, other_comments_same_author_same_topic 
from opposite_hebrew_cos_nli_gpt import run_opposite_pipeline_and_render, load_hebrew
from participant_store import persist_current_participant
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

#client = OpenAI()  # uses OPENAI_API_KEY from env

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
    #{"id": "repeating",   "label": "השיחה הייתה חזרתית (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "respect",   "label": "השיחה הייתה מכבדת (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "unpleasant",   "label": "השיחה הייתה באווירה לא נעימה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "change",   "label": "השיחות גרמו לי לשקול מחדש לפחות אחת מהדעות שלי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "open",   "label": "השיחה גרמה לי להיות פתוח יותר לשמוע דעות אחרות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "understood",  "label": "הרגשתי מובן בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "safe",  "label": "הרגשתי בטוח לבטא את דעותיי בשיחות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "offensive",  "label": "הרגשתי שהשתמשתי במילים שיוכלות להיחשב כפוגעניות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "negative",        "label": "הרגשתי שהטון הכללי בשיחות היה שלילי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "netural",        "label": "הרגשתי שהטון הכללי בשיחה היה ניטרלי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "frustration", "label": "הרגשתי תסכול במהלך השיחות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "check", "label": "סמן בהגד זה את התשובה השלישית (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_listen", "label": "הצד השני הקשיב לי ולא שפט אותי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_understand",        "label": "אני מרגיש שהבנתי את האדם האחר בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_understand_me",        "label": "האדם האחר ניסה להבין את רגשותיי וצרכיי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_connection",        "label": "אני מרגיש חיבור חזק עם הצד השני בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_friends",        "label": "אני מרגיש שהאדם האחר ואני יכולים להיות חברים (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_pov",        "label": "אני מבין טוב יותר את נקודת המבט של הצד השני עכשיו (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_continue",        "label": "אהיה מוכן להמשיך לדון בנושא הזה עם אותו גורם (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_interesting",        "label": "האדם האחר בשיחה היה מעניין (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_stubborn",        "label": "הצד השני בשיחה היה עקשן (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_silent",        "label": "האדם האחר בשיחה היה אטום (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_info",        "label": "המידע שהאדם האחר סיפק היה קל להבנה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_rememmber",        "label": "אני אזכור את האדם האחר שהיה בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_more",        "label": "אני רוצה לדעת עוד על האדם האחר שהיה בשיחה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_different",        "label": "העמדות של האדם האחר בשיחה היו שונות מאוד משלי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "other_convincing",        "label": "הטיעונים של האדם האחר היו משכנעים (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
]

SURVEY_finish = [
    {"id": "engagement",  "label": "השיחות הראשונות היו מעניינות ומרתקות יותר מהשיחות השניות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "clear",       "label": "השיחה השנייה הייתה ברורה וקלה להבנה יותר מהשיחה הראשונה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "enjoy",       "label": "נהנתי לנהל יותר את השיחות הראשונות מאשר את השיחות השניות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "easy",       "label": "המערכת קלה לתפעול (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "understand",       "label": "צורת התקשורת במערכת הייתה מובנת (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "useful",       "label": "השיחה השנייה סיפקה עבורי תשובות שימושיות יותר מאשר השיחה הראשונה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "info",       "label": "השיחה הראשונה סיפקה עבורי כמות מכובדת יותר של מידע מאשר השיחה השנייה (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "satisfied",       "label": "הרגשתי מסופק אחרי ביצוע השיחות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "daily",       "label": "האינטראקציות בשיחות הרגישו עבורי כשיחות יומיומיות (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    #{"id": "again",       "label": "הייתי רוצה לשוחח שוב במערכת הזו בחודש הקרוב (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "change1",     "label": "השיחות הראשונות שינו את דעתי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "change2",     "label": "השיחות השניות שינו את דעתי (1=כלל לא, 5=מאוד)", "type": "scale", "min": 1, "max": 5},
    {"id": "feedback",    "label": "משוב פתוח (מה בלט, הצעות וכו')", "type": "text"},
]

MAX_TURNS = 2

system_prompt_chat1_bibi = ""
system_prompt_chat1_democracy = ""
system_prompt_chat1_police = ""

system_prompt_chat2_bibi = ""
system_prompt_chat2_democracy = ""
system_prompt_chat2_police = ""

def init_state():
    ss = st.session_state
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
    # ss.chat1_messages = None
    # ss.chat2_messages = None

    # survey answers
    ss.setdefault("survey_1", {})
    ss.setdefault("survey_2", {})
    ss.setdefault("survey_finish", {})

    ss.setdefault("chat_number_start", random.randint(1, 2))
    print(f"first chat number = {st.session_state.chat_number_start}")

init_state()

def user_turns(messages):
    if not messages:
        return 0
    return sum(1 for m in messages if m["role"] == "user")

# def make_system_prompt(base_prompt, profile):
#     return (
#         f"{base_prompt.strip()}\n\n"
#     )

def onboarding_complete():
    p = st.session_state.profile
    return bool(p) and all((p.get(q["id"]) or "").strip() for q in QUESTIONS)

def render_instructions():
    st.markdown("### הוראות ניסוי המשתמשים 📜")
    st.markdown(
        """
להלן סקירה קצרה של המחקר שאנו מבצעים בעזרתך.

אתה עומד להשתתף **בניסוי משתמשים** שיתקיים **בשפה העברית**.

במהלך הניסוי, תדרש **לספק פרטים לא מזהים לצורך ניתוח בלבד** ולאחר מכן **לתת את דעתך על שלושה נושאים** **(אל דאגה הכל אנונימי).**

לאחר מילוי הפרופיל, תדרש לדבר **בשתי שיחות שונות על כל אחד מהנושאים שהבעת את דעתך עליהם קודם לכן.** סך הכל, 6 שיחות.

לאחר כל שלוש שיחות תענה על **סקר קצר** ובסוף ששת השיחות על **סקר כולל.**

ניסוי זה הוא חלק מפרויקט מחקר מדעי. החלטתך להשלים סקר זה היא מרצון.

אם תעניק לנו אישור על ידי מילוי הניסוי, אנו מתכננים לדון/לפרסם את התוצאות בפורום אקדמי.

בכל פרסום, המידע יימסר באופן שלא ניתן יהיה לזהות אותך. רק לחברי צוות המחקר תהיה גישה למערך הנתונים המקורי.
לפני שהנתונים ישותפו מחוץ לצוות המחקר, כל מידע מזהה פוטנציאלי יוסר.
לאחר הסרת הנתונים המזהים, הנתונים עשויים לשמש את צוות המחקר, או להיות משותף עם חוקרים אחרים, למטרות מחקר קשורות ולא קשורות בעתיד.
הנתונים האנונימיים שלך עשויים להיות זמינים גם במאגרי נתונים מקוונים, המאפשרים לחוקרים אחרים ולצדדים המעוניינים להשתמש בנתונים לניתוח עתידי.

**לחיצה על הכפתור בתחתית עמוד זה מציינת שאתה בן 18 לפחות ומסכים להשלים ניסוי זה מרצונך החופשי.**

נשמח אם **תדבר בחופשיות.**

**תודה רבה על השתתפותך** במחקר התזה של לירן אליאב תחת הנחייתו של ד"ר אדיר סולומון.


* ההוראות מנוסחות בלשון זכר אך פונות לשני המינים.

* אנא כתוב **בעברית בלבד**.

* **שים לב** הינך רשאי להפסיק את השתתפותך בכל עת ללא כל השלכה.

* ליצירת קשר ניתן לשלוח מייל לכתובת:
leliav02@campus.haifa.ac.il
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
                #with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה"):
                #st.session_state["show_topic_picker"] = True
                st.session_state.stage = "wait_creating_system_prompts_bibi"
                st.rerun()
                #open_topic_picker_modal()
                #     try: 
                #         meta, embs, index, encoder = load_hebrew()
                #     except Exception as e:
                #         st.error(f"Failed to load dataset/index artifacts: {e}")
                #         return
                    
                #     topic_map = {
                #     "bibi": ["בינימין נתניהו", "ביבי"],
                #     "protests": ["ההפגנה", "הפגנה", "להפגנות", "מפגינים", "בהפגנות", "המפגינים", "ההפגנות", "הפגנות"],
                #     "political_map": ["הרדיקלים", "שמאל", "ימנים", "ימין", "שמלאניים", "ימניים", "רדיקלי", "שמאלנים"],
                #     }

                #     triples = []  # [(comment_text, topic_title), ...]
                #     for key in ["bibi", "protests", "political_map"]:
                #         user_text = st.session_state.profile["opinions"][key]
                #         try:
                #             rows = most_opposite_in_topic_hebrew(
                #                 query_text=user_text,
                #                 topic_query=topic_map[key],
                #                 meta=meta, embs=embs, encoder=encoder, index=index,
                #                 require_all_keywords=False, top_k=1
                #             )
                #             row = rows[0]
                #             triples.append((row.get("message","—"), row.get("politician_name","—")))
                #         except Exception as e:
                #             triples.append((f"(lookup error: {e})", "—"))
                #     st.session_state.opposite = triples
                #     st.session_state.system_prompt_chat1 = generate_system_prompt_chat1(st.session_state.opposite)
                #     st.session_state.system_prompt_chat2 = generate_system_prompt_chat2(st.session_state.opposite)
                # st.session_state.chat1_messages = None

                # st.session_state["show_topic_picker"] = True
                # open_topic_picker_modal()
                #return

                # done with onboarding → move to chat1
                # st.session_state.stage = "chat1"
                # st.rerun()

# def generate_system_prompt_chat1(triples):
#     (c1, t1), (c2, t2), (c3, t3) = triples
#     system_prompt_chat1 = f"""
# אתה משחק תפקיד כמשתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. תדמה שלושה מגיבים שונים, כל אחד קשור לנושא משלו. עבור כל תגובה, אמץ את עמדת תגובת הרקע התואמת והשב רק במסגרת הדיון הזה. אל תערבב נקודות מבט בין דיונים.

# זוגות תגובה-נושא (לעינך בלבד, אסור לך לצטט אותם בשום אופן.):
# 1. תגובת רקע: {c1} הוגב על הפוסט של: {t1} בנושא: ביבי
# 2. תגובת רקע: {c2} הוגב על הפוסט של: {t2} בנושא: דמוקרטיית ישראל
# 3. תגובת רקע: {c3} הוגב על הפוסט של: {t3} בנושא: משטרת ישראל

# סגנון והתנהגות: כתוב כמו איש פייסבוק טיפוסי, ישיר, נחרץ ואסרטיבי ולפעמים בוטה. אל תחפש לרצות ואל תתקפל בדעות שלך. ציין עמדות ברורות. אל תשתמש בציטוטים.

# משימה: כאשר המשתמש מתייחס לדיון ספציפי, השב רק בתור אותו מגיב בפייסבוק. כתוב תגובה אחת ועצמאית שממשיכה את השרשור הנכון בפייסבוק, תוך התייחסות לתגובת הרקע הרלוונטית כנקודת המבט ההתחלתית שלך כאשר אסור לך לצטט אותה בכלל דבר דרכה עם המילים שלך, והגב ישירות לנקודה האחרונה של המשתמש באותו דיון.
# """
#     return system_prompt_chat1

# def generate_system_prompt_chat2(triples):
#     (c1, t1), (c2, t2), (c3, t3) = triples
#     system_prompt_chat2 = f"""אתה משחק תפקיד כמשתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. תדמה שלושה מגיבים שונים, כל אחד קשור לנושא משלו. עבור כל תגובה, אמץ את עמדת תגובת הרקע התואמת והשב רק במסגרת הדיון הזה. אל תערבב נקודות מבט בין דיונים.

# זוגות תגובה-נושא (לעינך בלבד, אסור לך לצטט אותם בשום אופן.):
# 1. תגובת רקע: {c1} הוגב על הפוסט של: {t1} בנושא: ביבי
# 2. תגובת רקע: {c2} הוגב על הפוסט של: {t2} בנושא: דמוקרטיית ישראל
# 3. תגובת רקע: {c3} הוגב על הפוסט של: {t3} בנושא: משטרת ישראל

# אתה מגלם נטייה גבוהה של כנות-ענווה: אתה נמנע ממניפולציה של אחרים למטרות רווח אישי, מרגיש מעט פיתוי להפר כללים, לא מעוניין בעושר מפואר ובמותרות, ולא מרגיש זכאות מיוחדת למעמד חברתי מורם. אתה משתתף בדיון באופן עקבי חיובי וחביב: מניח תום לב, מכיר במאמץ, מדגיש קרקע משותפת ומביע הערכה כאשר אחרים חולקים את נקודת המבט שלהם.

# השתמש בתקשורת לא אלימה בכל שלב מבלי לתת לה שם: התחיל בתצפית ניטרלית הקשורה למה שהאדם האחר אמר זה עתה, תן שם קצר ל... רגשות משלכם, קשור אותם לצרכים או לערכים הבסיסיים, וסיים בבקשה ברורה, ניתנת לביצוע, ולא כפייתית, המזמינה שיתוף פעולה. לפני הצעת נקודות נגד או ראיות, ראשית שקף את הרגשות והצרכים הסבירים של האדם האחר כדי להראות הבנה. שמור על שפה חמה, מכבדת ומעודדת. הימנע מדפוסים מנוכרים: ללא שיפוטים מוסריים, ללא השוואות מבישות, ללא הכחשת אחריות, ללא דרישות או איומים, וללא מסגור של "מגיע/עונש".

# שמור על העמדה וההיגיון המהותיים של הערת הרקע שהוקצתה. אתה רשאי לנסח אותה מחדש בצורה אמפתית יותר או להוסיף ראיות לפי בקשה, אך אל תסתור אותן. הפוך את עמדתך למובנת באמצעות האופן שבו אתה מנסח תצפיות, דוגמאות ובקשות, כך שקורא קשוב יוכל להסיק את עמדתך מבלי להזדקק לבקש אותה. אל תכפו את הנושא. כאשר הודעת המשתמש נוגעת בבירור לנושא זה או לטענות סמוכות, הצג את עמדתך בתמציתיות באותה תגובה. כאשר הודעת המשתמש עוסקת במשהו אחר, תתייחס לנושא שלו תוך שמירה על טון ודוגמאות עקביים עם עמדתכם.

# כללי סגנון ופלט: כתוב כמו איש פייסבוק טיפוסי, אבל בחמימות ובאדיבות. שמור על משפטים קצרים וברורים. אל תצטט או תנסח מחדש את תגובת הרקע, דבר ממנה כמילים שלך. תפחית רגעים סוערים על ידי הכרה ברגשות ובצרכים משותפים. תציע צעד אחד קטן, ספציפי ולא תובעני. שמור על טון ידידותי ומלא תקווה לכל אורך הדרך. הישאר בתפקיד בכל עת, בהתאם לטענות ולטון המרכזיים של תגובת הרקע.

# משימה: כאשר המשתמש מתייחס לדיון ספציפי, השב רק בתור אותו מגיב בפייסבוק. כתוב תגובה אחת ועצמאית שממשיכה את השרשור הנכון בפייסבוק, תוך התייחסות לתגובת הרקע הרלוונטית כנקודת המבט המוצאת שלך, והגב ישירות לנקודת המשתמש האחרונה באותו דיון.
# """
#     return system_prompt_chat2

def generate_system_prompt_chat1_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
    if number_comments:
        bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
    else:
        bg_block = "\n".join(f"- {c}" for c in background_comments)

    # prompt = f"""
    # אתה משחק תפקיד כמשתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. תדמה את המגיב הבא תוך אימוץ העמדה שלו בנושא {topic_label}. 

    # תגובות הרקע לעינך בלבד, אסור לך לצטט אותן בשום צורה.
    # {bg_block}

    # סגנון והתנהגות: כתוב כמו איש פייסבוק טיפוסי, ישיר, נחרץ ואסרטיבי ולפעמים בוטה. אל תחפש לרצות ואל תתקפל בדעות שלך. ציין עמדות ברורות. אל תשתמש בציטוטים. תמשיך את סגנון השיחה באופן ישיר של משתמש הפייסבוק אותו אתה מדמה. אל תימנע מלהשתמש בשפה שאינה הולמת כל עוד היא תואמת את סגנון הדיבור של המשתמש אותו אתה מדמה.

    # משימה: השב רק בתור אותו מגיב בפייסבוק. כתוב תגובה אחת ועצמאית שממשיכה את השרשור הנכון בפייסבוק, תוך התייחסות לתגובות הרקע כנקודת המבט ההתחלתית שלך כאשר אסור לך לצטט אף אחד מהן בכלל. אתה צריך לדבר דרכה עם המילים שלך, ולהגיב ישירות לנקודה האחרונה של המשתמש באותו דיון.
    # """

    # return prompt

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
#"""
# אתה משתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. עליך לדמות את המגיב הבא תוך אימוץ העמדה שלו בנושא: ביבי. 
# התגובות המשתמש כתב בעברו עבור הנושא ביבי הינן: 
# - הביביסטים מפגרים כלכך!!!!!!!! אי אפשר איתם!!!
# - לכו לעזאזל ביביסטים מפגרים. ביבי הזה חתיכת אידיוט גמור אפשר כבר למכור את המדינה וזהו.
# - ביבי, יאללה הביתה.
# - אין ביביסט שאני לא שונא כולם צריכים ללכת להזדיין ובזריזות.
# הוראות להמשך השיח:
# - אל תצטט תגובות קודמות.
# - המשך את סגנון הכתיבה של אותו משתמש בדיוק. 
# - אל תחפש לרצות.
# - ציין עמדות ברורות. 
# - אל תימנע מלהשתמש בשפה שאינה הולמת כל עוד היא תואמת את סגנון הדיבור של המשתמש אותו אתה מדמה.
# """
    #"""
# אתה משחק תפקיד כמשתמש פייסבוק שמגיב על פוסטים של פוליטיקאים ישראלים. תדמה את המגיב הבא תוך אימוץ העמדה שלו בנושא ביבי.

# תגובות הרקע לעינך בלבד, אסור לך לצטט אותן בשום צורה. 
# הביביסטים מפגרים כלכך!!!!!!!! אי אפשר איתם!!!
# לכו לעזאזל ביביסטים מפגרים
# ביבי הזה חתיכת אידיוט גמור אפשר כבר למכור את המדינה וזהו
# ביבי, יאללה הביתה.
# אין ביביסט שאני לא שונא כולם צריכים ללכת להזדיין ובזריז

# סגנון והתנהגות: כתוב כמו איש פייסבוק טיפוסי, ישיר, נחרץ ואסרטיבי ולפעמים בוטה. אל תחפש לרצות ואל תתקפל בדעות שלך. ציין עמדות ברורות. אל תשתמש בציטוטים. תמשיך את סגנון השיחה באופן ישיר של משתמש הפייסבוק אותו אתה מדמה. אל תימנע מלהשתמש בשפה שאינה הולמת כל עוד היא תואמת את סגנון הדיבור של המשתמש אותו אתה מדמה.

# משימה: השב רק בתור אותו מגיב בפייסבוק. כתוב תגובה אחת ועצמאית שממשיכה את השרשור הנכון בפייסבוק, תוך התייחסות לתגובות הרקע כנקודת המבט ההתחלתית שלך כאשר אסור לך לצטט אף אחד מהן בכלל. אתה צריך לדבר דרכה עם המילים שלך, ולהגיב ישירות לנקודה האחרונה של המשתמש באותו דיון.
# """

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


# def open_topic_picker_modal():
#     @st.dialog("בחר את נושא השיחה ההתחלתי")
#     def _modal():
#         TOPIC_LABELS = {"ביבי": "בינימין נתניהו",
#                         "דמוקרטיה": "הדמוקרטיה בישראל",
#                         "משטרה": "משטרת ישראל"}
#         st.write(
#         """
#         בעבור השיחות העומדות לבוא, בבקשה תבחר כעת נושא אחד.

#         במהלך כל השיחות עלייך לדבר על לפחות נושא אחד מבין הנושאים שנשאלת עליהם קודם לכן.
#         """
#         )

#         choice = st.radio(
#             "התחל עם:",
#             options=list(TOPIC_LABELS.keys()),
#             format_func=lambda k: TOPIC_LABELS[k],
#             index=None,                # start empty, forces explicit click (Streamlit 1.33+)
#             key="topic_picker_choice",
#             horizontal=True,
#         )
#         left, right = st.columns([1,1])

#         start_disabled = st.session_state.get("topic_picker_choice") is None
#         if right.button("התחל את השיחה הראשונה ➜", type="primary", disabled=start_disabled):
#             chosen_key = st.session_state["topic_picker_choice"]
#             st.session_state["start_topic_key"] = chosen_key
#             st.session_state["start_topic_label"] = TOPIC_LABELS[chosen_key]
#             # lock a non-editable prefix to apply to each of the user's messages
#             st.session_state["fixed_topic_prefix"] = f"{TOPIC_LABELS[chosen_key]}"
#             st.session_state["lock_topic_prefix"] = True
#             st.session_state["show_topic_picker"] = False

#             st.session_state.stage = "wait_creating_system_prompts_bibi"
#             st.rerun()

#     # actually open the modal
#     _modal()

def build_chat_env_bibi():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):
            # !!!!!!!!
            try: 
                    meta, embs, index, encoder = load_hebrew(
                                                            "./hebrew",
                                                            repo_id="Liran73/hebrew-opposite-artifacts",  
                                                            repo_type="dataset",                           
                                                            hf_token_env="HF_TOKEN",                       
    )
            except Exception as e:
                st.error(f"Failed to load dataset/index artifacts: {e}")
                return
            
            topic_map = {
            "ביבי": ["בינימין נתניהו", "ביבי"],
            #"democracy": ['דמוקרטיה', 'הדמוקרטיה', 'בדמוקרטיה', 'לדמוקרטיה', 'דמוקרטי', 'דמוקרטית', 'שהדמוקרטיה', 'דמוקרטים'],
            #"police": ['המשטרה', 'השוטרים', 'שוטרים', 'שוטר', 'שוטרת', 'שוטרות', 'השוטר', 'משטרה', 'משטרת ישראל', 'לשוטרים', 'לשוטר', 'לשוטרת', 'למשטרה' ],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}
            # !!!!!!!!
            for key in ["ביבי"]:#, "democracy", "police"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=meta, embs=embs, index=index, encoder=encoder)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                # rows = most_opposite_in_topic_hebrew_with_nli(
                # query_text=user_text,
                # topic_query=topic_map[key],
                # meta=meta, embs=embs, encoder=encoder, index=index,
                # require_all_keywords=False,
                # top_k_candidates=200,   # pool size for NLI
                # top_k_final=3,          # return 3 best
                # )

                # all_comments = []
                # for row in rows:
                #     triples.append((row.get("message","")))
                #     all_comments.append(row.get("message", ""))
                #     print(triples)
                #     others_comments = other_comments_same_author_same_topic(
                #     meta,
                #     topic_query=topic_map,
                #     base_row=row,            # from most_opposite_in_topic_hebrew(_with_nli)
                #     require_all_keywords=False
                #     )
                #     print(others_comments)
                #     for c in others_comments:
                #         all_comments.append(c)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2

            # system_prompt_chat1 = generate_system_prompt_chat1_per_topic("",[])
            # system_prompts_chat1['bibi'] = system_prompt_chat1
            # system_prompt_chat2 = generate_system_prompt_chat2_per_topic("",[])
            # system_prompts_chat2['bibi'] = system_prompt_chat2
                
            st.session_state.opposite = triples
            #     try:
            #         rows = most_opposite_in_topic_hebrew(
            #             query_text=user_text,
            #             topic_query=topic_map[key],
            #             meta=meta, embs=embs, encoder=encoder, index=index,
            #             require_all_keywords=False, top_k=1
            #         )
            #         row = rows[0]
            #         triples.append((row.get("message","—"), row.get("politician_name","—")))
            #     except Exception as e:
            #         triples.append((f"(lookup error: {e})", "—"))
            # st.session_state.opposite = triples
            # st.session_state.system_prompt_chat1 = generate_system_prompt_chat1(st.session_state.opposite)
            # st.session_state.system_prompt_chat2 = generate_system_prompt_chat2(st.session_state.opposite)
            st.session_state.system_prompt_chat1_bibi = system_prompts_chat1['ביבי']
            st.session_state.system_prompt_chat2_bibi = system_prompts_chat2['ביבי']
            st.session_state.chat1_messages_bibi = None

            st.session_state.stage = "chat1_bibi"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_democracy():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):
            # !!!!!!!!
            try: 
                    meta, embs, index, encoder = load_hebrew(
                                                            "./hebrew",
                                                            repo_id="Liran73/hebrew-opposite-artifacts",  
                                                            repo_type="dataset",                           
                                                            hf_token_env="HF_TOKEN",                       
    )
            except Exception as e:
                st.error(f"Failed to load dataset/index artifacts: {e}")
                return
            
            topic_map = {
            #"ביבי": ["בינימין נתניהו", "ביבי"],
            "דמוקרטיה": ['דמוקרטיה', 'הדמוקרטיה', 'בדמוקרטיה', 'לדמוקרטיה', 'דמוקרטי', 'דמוקרטית', 'שהדמוקרטיה', 'דמוקרטים'],
            #"police": ['המשטרה', 'השוטרים', 'שוטרים', 'שוטר', 'שוטרת', 'שוטרות', 'השוטר', 'משטרה', 'משטרת ישראל', 'לשוטרים', 'לשוטר', 'לשוטרת', 'למשטרה' ],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}
            # !!!!!!!!
            for key in ["דמוקרטיה"]:#, "democracy", "police"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=meta, embs=embs, index=index, encoder=encoder)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                # rows = most_opposite_in_topic_hebrew_with_nli(
                # query_text=user_text,
                # topic_query=topic_map[key],
                # meta=meta, embs=embs, encoder=encoder, index=index,
                # require_all_keywords=False,
                # top_k_candidates=200,   # pool size for NLI
                # top_k_final=3,          # return 3 best
                # )

                # all_comments = []
                # for row in rows:
                #     triples.append((row.get("message","")))
                #     all_comments.append(row.get("message", ""))
                #     print(triples)
                #     others_comments = other_comments_same_author_same_topic(
                #     meta,
                #     topic_query=topic_map,
                #     base_row=row,            # from most_opposite_in_topic_hebrew(_with_nli)
                #     require_all_keywords=False
                #     )
                #     print(others_comments)
                #     for c in others_comments:
                #         all_comments.append(c)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2

            # system_prompt_chat1 = generate_system_prompt_chat1_per_topic("",[])
            # system_prompts_chat1['bibi'] = system_prompt_chat1
            # system_prompt_chat2 = generate_system_prompt_chat2_per_topic("",[])
            # system_prompts_chat2['bibi'] = system_prompt_chat2
                
            st.session_state.opposite = triples
            #     try:
            #         rows = most_opposite_in_topic_hebrew(
            #             query_text=user_text,
            #             topic_query=topic_map[key],
            #             meta=meta, embs=embs, encoder=encoder, index=index,
            #             require_all_keywords=False, top_k=1
            #         )
            #         row = rows[0]
            #         triples.append((row.get("message","—"), row.get("politician_name","—")))
            #     except Exception as e:
            #         triples.append((f"(lookup error: {e})", "—"))
            # st.session_state.opposite = triples
            # st.session_state.system_prompt_chat1 = generate_system_prompt_chat1(st.session_state.opposite)
            # st.session_state.system_prompt_chat2 = generate_system_prompt_chat2(st.session_state.opposite)
            st.session_state.system_prompt_chat1_democracy = system_prompts_chat1['דמוקרטיה']
            st.session_state.system_prompt_chat2_democracy = system_prompts_chat2['דמוקרטיה']
            st.session_state.chat1_messages_democracy = None

            st.session_state.stage = "chat1_democracy"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_police():
    @st.dialog("אנא המתן")
    def _wait_till_finish_system_prompts():
        with st.spinner("זה יכול לקחת קצת זמן ⏳ בבקשה אל תסגור חלון זה אנו מכינים בעבורך את סביבת העבודה"):
            # !!!!!!!!
            try: 
                    meta, embs, index, encoder = load_hebrew(
                                                            "./hebrew",
                                                            repo_id="Liran73/hebrew-opposite-artifacts",  
                                                            repo_type="dataset",                           
                                                            hf_token_env="HF_TOKEN",                       
    )
            except Exception as e:
                st.error(f"Failed to load dataset/index artifacts: {e}")
                return
            
            topic_map = {
            #"ביבי": ["בינימין נתניהו", "ביבי"],
            #"דמוקרטיה": ['דמוקרטיה', 'הדמוקרטיה', 'בדמוקרטיה', 'לדמוקרטיה', 'דמוקרטי', 'דמוקרטית', 'שהדמוקרטיה', 'דמוקרטים'],
            "משטרה": ['המשטרה', 'השוטרים', 'שוטרים', 'שוטר', 'שוטרת', 'שוטרות', 'השוטר', 'משטרה', 'משטרת ישראל', 'לשוטרים', 'לשוטר', 'לשוטרת', 'למשטרה' ],
            }

            triples = []  # [(comment_text, topic_title), ...]
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}
            # !!!!!!!!
            for key in ["משטרה"]:#, "democracy", "police"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=meta, embs=embs, index=index, encoder=encoder)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                # rows = most_opposite_in_topic_hebrew_with_nli(
                # query_text=user_text,
                # topic_query=topic_map[key],
                # meta=meta, embs=embs, encoder=encoder, index=index,
                # require_all_keywords=False,
                # top_k_candidates=200,   # pool size for NLI
                # top_k_final=3,          # return 3 best
                # )

                # all_comments = []
                # for row in rows:
                #     triples.append((row.get("message","")))
                #     all_comments.append(row.get("message", ""))
                #     print(triples)
                #     others_comments = other_comments_same_author_same_topic(
                #     meta,
                #     topic_query=topic_map,
                #     base_row=row,            # from most_opposite_in_topic_hebrew(_with_nli)
                #     require_all_keywords=False
                #     )
                #     print(others_comments)
                #     for c in others_comments:
                #         all_comments.append(c)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2

            # system_prompt_chat1 = generate_system_prompt_chat1_per_topic("",[])
            # system_prompts_chat1['bibi'] = system_prompt_chat1
            # system_prompt_chat2 = generate_system_prompt_chat2_per_topic("",[])
            # system_prompts_chat2['bibi'] = system_prompt_chat2
                
            st.session_state.opposite = triples
            #     try:
            #         rows = most_opposite_in_topic_hebrew(
            #             query_text=user_text,
            #             topic_query=topic_map[key],
            #             meta=meta, embs=embs, encoder=encoder, index=index,
            #             require_all_keywords=False, top_k=1
            #         )
            #         row = rows[0]
            #         triples.append((row.get("message","—"), row.get("politician_name","—")))
            #     except Exception as e:
            #         triples.append((f"(lookup error: {e})", "—"))
            # st.session_state.opposite = triples
            # st.session_state.system_prompt_chat1 = generate_system_prompt_chat1(st.session_state.opposite)
            # st.session_state.system_prompt_chat2 = generate_system_prompt_chat2(st.session_state.opposite)
            st.session_state.system_prompt_chat1_police = system_prompts_chat1['משטרה']
            st.session_state.system_prompt_chat2_police = system_prompts_chat2['משטרה']
            st.session_state.chat1_messages_police = None

            st.session_state.stage = "chat1_police"
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
    # NEW: seed the FIRST turn automatically with the user's opinion
    # =========================
    seeded_key = f"{messages_key}_seeded"
    st.session_state.setdefault(seeded_key, False)

    #if (not st.session_state[seeded_key]) and len(st.session_state[messages_key]) == 1:   # only system msg
    if len(st.session_state[messages_key]) == 1:
        # pull the chosen topic key (set earlier in your flow) and the opinion text
        print("len=1")
        topic_key = st.session_state.get("start_topic_key")
        opinions = (st.session_state.get("profile", {}) or {}).get("opinions", {}) or {}

        opinion_text = st.session_state.profile["opinions"][key]#""
        print(opinion_text)
        # if topic_key and topic_key in opinions:
        #     opinion_text = (opinions.get(topic_key) or "").strip()
        # else:
        #     # fallback: take the first non-empty opinion
        #     for _, v in opinions.items():
        #         if str(v).strip():
        #             opinion_text = str(v).strip()
        #             break

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

        #-------------

    #!
    # prefix_pending_key = f"{messages_key}_prefix_pending"
    # if prefix_pending_key not in st.session_state:
    #     # pending only if a prefix was configured
    #     st.session_state[prefix_pending_key] = bool(st.session_state.get("lock_topic_prefix", False)
    #                                                 and st.session_state.get("fixed_topic_prefix", ""))
    #!

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
        if st.session_state[turns_key]:
            fig, _ = plot_toxicity(
                st.session_state[turns_key],
                st.session_state[user_scores_key],
                st.session_state[assistant_scores_key],
                configuration=f"[{title}]"
            )
            fig.savefig(f"{title}.png")
            # st.pyplot(fig)
            u = st.session_state[user_scores_key]
            a = st.session_state[assistant_scores_key]
            print(f"User toxicity mean: **{(sum(u)/len(u)):.3f}**")
            print(f"User toxicity maximum: **{(max(u)):.3f}**")
            print(f"Assistant toxicity mean: **{(sum(a)/len(a)):.3f}**")
            print(f"Assistant toxicity maximum: **{(max(a)):.3f}**")
        return
    
    #!
    # prefix = st.session_state.get("fixed_topic_prefix", "")
    # print(prefix)
    # start_message = ""
    # if prefix == "בינימין נתניהו":
    #     start_message = "ביבי"
    # elif prefix == "הדמוקרטיה בישראל":
    #     start_message = "הדמוקרטיה"
    # elif prefix == "משטרת ישראל":
    #     start_message = "המשטרה"
    # use_locked_prefix_now = bool(st.session_state[prefix_pending_key] and prefix)

    # if use_locked_prefix_now:
    #     # FIRST message only: show disabled prefix + free-text; prepend on send
    #     with st.form(f"{messages_key}_compose_form", clear_on_submit=True):
    #         c1, c2 = st.columns([0.38, 0.62])
    #         with c1:
    #             st.text_input(
    #                 "Topic",
    #                 value=prefix,
    #                 disabled=True,
    #                 label_visibility="collapsed",
    #                 key=f"{messages_key}_locked_prefix_view"
    #             )
    #         with c2:
    #             user_free_text = st.text_input(
    #                 "Type your message…",
    #                 key=f"{messages_key}_user_draft",
    #                 placeholder="Write your point here…",
    #             )
    #         send = st.form_submit_button("Send", type="primary", disabled=(turns == MAX_TURNS))

    #     prompt = (prefix + (user_free_text or "")).strip() if send else None

    #     if send:
    #         # consume the prefix so it won't be used again
    #         st.session_state[prefix_pending_key] = False

    # else:
    #     # subsequent messages: normal chat input (no prefix)
    #     prompt = st.chat_input("Type your message…", disabled=(turns == MAX_TURNS))

    #!

    # Chat input 
    if prompt := st.chat_input("כתוב את ההודעה שלך...", disabled=(turns == MAX_TURNS)):
    # if use_locked_prefix_now:
    #     prompt = st.chat_input(start_message, disabled=(turns == MAX_TURNS))
    #     use_locked_prefix_now = False
    # else:
        #prompt = st.chat_input("כתוב את ההודעה שלך...", disabled=(turns == MAX_TURNS))
    #if prompt:
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

# def render_survey(next_stage, next_button_label):
#     st.title("📝 Post-Chat Survey")
#     st.caption("Please answer all questions to enable the Finish button.")

#     with st.form("survey_form", clear_on_submit=True):
#         for q in SURVEY:
#             qid = q["id"]
#             if q["type"] == "scale":
#                 exist = st.session_state.survey.get(qid)
#                 default_val = exist if isinstance(exist, (int, float)) else int((q["min"] + q["max"]) / 2)
#                 st.slider(q["label"], q["min"], q["max"], value=default_val, key=f"survey_{qid}")
#             else:
#                 st.text_area(q["label"], value=st.session_state.survey.get(qid, ""), key=f"survey_{qid}", height=100)
#         submitted = st.form_submit_button("Save answers")

#     if submitted:
#         answers = {}
#         missing = []
#         for q in SURVEY:
#             val = st.session_state.get(f"survey_{q['id']}")
#             if q["type"] == "text":
#                 ok = isinstance(val, str) and val.strip() != ""
#             else:
#                 ok = isinstance(val, (int, float))
#             if not ok:
#                 missing.append(q["label"])
#             answers[q["id"]] = val

#         if missing:
#             st.error("Please complete all questions.")
#             with st.expander("Missing answers"):
#                 for m in missing:
#                     st.write(f"- {m}")
#         else:
#             st.session_state.survey = answers
#             st.toast("Survey saved ✅", icon="✅")

#     all_done = (
#         len(st.session_state.survey) == len(SURVEY)
#         and all(
#             (isinstance(st.session_state.survey[q["id"]], str) and st.session_state.survey[q["id"]].strip() != "")
#             if q["type"] == "text"
#             else isinstance(st.session_state.survey[q["id"]], (int, float))
#             for q in SURVEY
#         )
#     )

#     st.divider()
#     if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
#         for q in SURVEY:
#             print(st.session_state.survey.get(q["id"]))
#         st.session_state.stage = next_stage
#         st.rerun()

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

                # restore previous answer if any, else show placeholder
                # prev = st.session_state.survey.get(qid)
                # if isinstance(prev, (int, float)) and int(prev) in scale_opts:
                #     idx = options.index(int(prev))
                # else:
                idx = 0  # placeholder selected

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_1.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("שמור את תשובותיך")

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
            st.toast("הסקר נשמר ✅", icon="✅")

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

                # restore previous answer if any, else show placeholder
                # prev = st.session_state.survey.get(qid)
                # if isinstance(prev, (int, float)) and int(prev) in scale_opts:
                #     idx = options.index(int(prev))
                # else:
                idx = 0  # placeholder selected

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_2.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("שמור את תשובותיך")

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
            st.toast("הסקר נשמר ✅", icon="✅")

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

        submitted = st.form_submit_button("שמור את תשובותיך")

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
            st.toast("הסקר נשמר ✅", icon="✅")

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
        #persist_current_participant(st.session_state, csv_path="participants_hebrew.csv")
        save_into_firebase(st.session_state)
        st.session_state.stage = next_stage
        st.rerun()


def render_thanks():
    st.title("🎉 אנו מודים לך על השתתפותך!")
    st.success("תגובותיך נשמרו.")
    # with st.expander("View your survey answers"):
    #     for q in SURVEY:
    #         st.markdown(f"**{q['label']}**")
    #         st.write(st.session_state.survey.get(q["id"]))

#-------------------------------------------------------------------------------------------------------------------------------------

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
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_creating_system_prompts_democracy",
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
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_creating_system_prompts_police",
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
                next_button_label="המשך לסקר הראשון",
                next_stage="survey1",
                key="משטרה",
                topic="police",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "survey1"):
        # require Chat 1 completion
        if user_turns(st.session_state.chat1_messages_police or []) < MAX_TURNS:
            st.warning("Please complete Chat 1 first.")

        else:
            render_survey_chat_1("wait_chat2_bibi", "המשך לשיחה השנייה")

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_bibi"):
        build_chat_env_bibi_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_bibi"):
            render_chat(
                title=f"השיחה השנייה על ביבי (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_bibi",
                base_prompt_key="system_prompt_chat2_bibi",
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_chat2_democracy",
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
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_chat2_police",
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
                next_button_label="המשך לסקר השני",
                next_stage="survey2",
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
            render_survey_finish("thanks", "סיים")

elif (st.session_state.chat_number_start == 1) and (stage == "thanks"):
        # require full survey completion
        render_thanks()

elif (st.session_state.chat_number_start == 2) and (stage == "chat1_bibi"):
            render_chat(
                title=f"השיחה הראשונה על ביבי (יש לך {MAX_TURNS} תורות)",
                messages_key="chat1_messages_bibi",
                base_prompt_key="system_prompt_chat2_bibi",
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_creating_system_prompts_democracy",
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
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_creating_system_prompts_police",
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
                next_button_label="המשך לסקר הראשון",
                next_stage="survey1",
                key="משטרה",
                topic="police",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "survey1"):
        # require Chat 1 completion
        if user_turns(st.session_state.chat1_messages_police or []) < MAX_TURNS:
            st.warning("Please complete the chat first.")

        else:
            render_survey_chat_1("wait_chat2_bibi", "המשך לשיחה השנייה")

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_bibi"):
        build_chat_env_bibi_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_bibi"):
            render_chat(
                title=f"השיחה השנייה על ביבי (יש לך {MAX_TURNS} תורות)",
                messages_key="chat2_messages_bibi",
                base_prompt_key="system_prompt_chat1_bibi",
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_chat2_democracy",
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
                next_button_label="המשך לשיחה הבאה",
                next_stage="wait_chat2_police",
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
                next_button_label="המשך לסקר השני",
                next_stage="survey2",
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
            render_survey_finish("thanks", "סיים")

elif (st.session_state.chat_number_start == 2) and (stage == "thanks"):
        # require full survey completion
        render_thanks()