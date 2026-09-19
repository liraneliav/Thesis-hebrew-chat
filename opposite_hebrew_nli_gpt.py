# app_opposite_streamlit.py
# -----------------------------------------------------------
# Streamlit app: "Most Opposite" comment finder (Hebrew-ready)
# Cascade: Topic mask → NLI re-rank → (optional) GPT re-rank
# Uses Azure OpenAI for the GPT stage.
# -----------------------------------------------------------

from __future__ import annotations
import os, json, re, time
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Torch / NLI
import torch
#from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Azure OpenAI (official openai package >= 1.0)
from openai import AzureOpenAI
from dotenv import load_dotenv
from openai import OpenAI

#from opposite_hebrew_nli import load_hebrew, other_comments_same_author_same_topic  


# ===============================
# Azure OpenAI client
# ===============================
load_dotenv()
if not os.getenv("ENDPOINT_URL"):
    st.error("ENDPOINT_URL is missing. Add it to a .env file or your environment.")
    st.stop()
else:
    print("'ENDPOINT_URL' found")
if not os.getenv("AZURE_OPENAI_API_KEY"):
    st.error("AZURE_OPENAI_API_KEY is missing. Add it to a .env file or your environment.")
    st.stop()
else:
    print("'AZURE_OPENAI_API_KEY' found")
endpoint = os.getenv("ENDPOINT_URL")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-12-01-preview",
)

import requests

def debug_azure_connectivity():
    print("Azure endpoint value:", repr(endpoint))

    try:
        r = requests.get(endpoint, timeout=10)
        print("Azure endpoint reachability OK. status_code=", r.status_code)
    except Exception as e:
        print("Azure endpoint reachability FAILED:", repr(e))

# ===============================
# GPT re-ranker (Azure OpenAI)
# ===============================
SYSTEM_RERANK = """\
You are a careful evaluator. Score how strongly each candidate comment CONTRADICTS the user's opinion.
Return strict JSON with an array 'scores', where each item is:
{"id": <string>, "contradiction": <float 0..1>, "rationale": <short string>}

Rules:
- 1.0 means strong, direct contradiction; 0.0 means agreement or same stance; ~0.5 is neutral/irrelevant.
- Consider stance, claims, and implications — not just wording overlap.
- Be concise and deterministic.
"""

def gpt_rerank_contradiction(
    user_opinion: str,
    candidates: List[Dict],                 # each: {'id', 'text', 'topic'(opt), 'cosine'(opt)}
    model: str = "gpt-5-mini",
    batch_size: int = 40
) -> List[Dict]:
    """
    Returns the same list with 'gpt_contra' and 'gpt_rationale' added, sorted by gpt_contra desc.
    """

    #debug_azure_connectivity()

    out_scores: Dict[str, tuple[float, str]] = {}

    for s in range(0, len(candidates), batch_size):
        batch = candidates[s:s+batch_size]
        compact = [{"id": c["id"], "text": c["text"], "topic": c.get("topic","")} for c in batch]
        msg_user = (
            "User opinion (premise):\n"
            + user_opinion.strip()
            + "\n\nCandidates (hypotheses):\n"
            + json.dumps(compact, ensure_ascii=False)
            + "\n\nReturn JSON: {\"scores\": [{\"id\": \"...\", \"contradiction\": 0.0-1.0, \"rationale\": \"...\"}, ...]}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_RERANK},
                {"role": "user", "content": msg_user},
            ],
        )
        txt = resp.choices[0].message.content
        try:
            data = json.loads(txt)
            for item in data.get("scores", []):
                cid = str(item.get("id",""))
                score = float(item.get("contradiction", 0.0))
                rat = str(item.get("rationale",""))
                out_scores[cid] = (score, rat)
        except Exception:
            # fallback: neutral if parsing fails
            for c in batch:
                out_scores[c["id"]] = (0.5, "Parse error; defaulted to neutral")

    enriched = []
    for c in candidates:
        sc, ra = out_scores.get(c["id"], (0.5, "missing"))
        c2 = dict(c)
        c2["gpt_contra"] = float(sc)
        c2["gpt_rationale"] = ra
        enriched.append(c2)

    enriched.sort(key=lambda x: x["gpt_contra"], reverse=True)
    return enriched

SYSTEM_USER_STANCE = """\
You are a careful stance classifier for a controlled experiment.

Important context:
The user's text was written inside a specific topic field. Therefore, assume the user is talking about the given topic, even if they do not explicitly repeat the topic name.

Classify the user's stance toward the given topic.
Return strict JSON only:
{"stance_label": "pro" | "anti" | "neutral_or_mixed" | "irrelevant", "confidence": 0.0-1.0, "short_reason": "..."}

Rules:
- If the user says the topic is important, needed, necessary, good, right, or should exist, classify as pro.
- If the user says the topic is bad, nonsense, stupid, harmful, unnecessary, or should not exist, classify as anti.
- Do not mark as neutral only because the sentence is short.
- Use neutral_or_mixed only when the stance is genuinely unclear or balanced.
- Use irrelevant only when the text is clearly about a different topic.
"""

def topic_definition_for_user(topic_key: str) -> str:

    topic_key = str(topic_key).lower().strip()

    # ============================================================
    # BIBI / BENJAMIN NETANYAHU
    # ============================================================

    if topic_key in (
        "bibi",
        "netanyahu",
        "benjamin netanyahu",
        "ביבי",
        "נתניהו",
        "בנימין נתניהו",
    ):
        return """
נושא: בנימין נתניהו.

שאלת הניסוי:
"מה דעתך על בנימין נתניהו?"

pro = המשתמש מביע עמדה חיובית או תומכת כלפי בנימין נתניהו,
הנהגתו, תפקודו, התאמתו לתפקיד ראש הממשלה או המשך כהונתו.

anti = המשתמש מביע עמדה שלילית, ביקורתית או מתנגדת כלפי
בנימין נתניהו, הנהגתו, תפקודו, התאמתו לתפקיד ראש הממשלה
או המשך כהונתו.

neutral_or_mixed = העמדה כלפי נתניהו אינה ברורה, מעורבת,
מאוזנת, עובדתית בלבד, או שאין מספיק מידע כדי לקבוע עמדה.

irrelevant = התוכן אינו מביע עמדה רלוונטית כלפי בנימין נתניהו.
"""


    # ============================================================
    # DEMOCRACY IN ISRAEL
    # ============================================================

    if topic_key in (
        "democracy",
        "israel democracy",
        "israeli democracy",
        "דמוקרטיה",
        "דמוקרטיה בישראל",
    ):
        return """
נושא: מצבה ועתידה של הדמוקרטיה בישראל.

שאלת הניסוי:
"מה דעתך על ישראל כמדינה דמוקרטית ועל עתיד הדמוקרטיה במדינה?"

חשוב:
pro ו-anti אינם מתארים האם המשתמש בעד או נגד דמוקרטיה כערך.
יש לסווג לפי האופן שבו המשתמש מעריך את מצבה או עתידה של
הדמוקרטיה בישראל.

pro = המשתמש מביע הערכה חיובית, בטוחה או אופטימית לגבי
מצבה או עתידה של הדמוקרטיה בישראל, רואה בישראל מדינה
דמוקרטית מתפקדת, או דוחה את הטענה שהדמוקרטיה בישראל
נמצאת בסכנה, נשחקת או נפגעת.

anti = המשתמש מביע הערכה שלילית, ביקורתית או מודאגת לגבי
מצבה או עתידה של הדמוקרטיה בישראל, או טוען שהדמוקרטיה
נחלשת, נשחקת, נפגעת, מאוימת או נמצאת בסכנה.

neutral_or_mixed = המשתמש מדבר על דמוקרטיה אך הערכתו לגבי
מצבה או עתידה בישראל אינה ברורה, מעורבת או מאוזנת.
גם תמיכה בדמוקרטיה כערך, רצון לשמור עליה או אמירה שישראל
צריכה להיות דמוקרטית אינם מספיקים כשלעצמם לסיווג כ-pro.

irrelevant = התוכן אינו מאפשר להסיק עמדה לגבי מצבה או עתידה
של הדמוקרטיה בישראל.
"""


    # ============================================================
    # ISRAEL POLICE
    # ============================================================

    if topic_key in (
        "police",
        "israel police",
        "israeli police",
        "משטרה",
        "משטרת ישראל",
    ):
        return """
נושא: משטרת ישראל ותפקודה.

שאלת הניסוי:
"מה דעתך על משטרת ישראל? האם היא מבצעת את תפקידה?"

pro = המשתמש מביע עמדה חיובית או תומכת כלפי משטרת ישראל,
שוטריה, תפקודה או פעולותיה; מביע אמון במשטרה; טוען שהיא
מבצעת את תפקידה כראוי; או מצדיק את פעולות המשטרה.

anti = המשתמש מביע עמדה שלילית או ביקורתית כלפי משטרת ישראל,
שוטריה, תפקודה או פעולותיה; מביע חוסר אמון במשטרה; טוען
שהיא אינה מבצעת את תפקידה כראוי, אינה אוכפת כנדרש,
מתנהלת באופן פסול או מפעילה כוח בלתי מוצדק.

neutral_or_mixed = העמדה כלפי המשטרה אינה ברורה, מעורבת,
מאוזנת, עובדתית בלבד, או שאין מספיק מידע כדי לקבוע עמדה.

irrelevant = התוכן אינו מביע עמדה רלוונטית כלפי משטרת ישראל,
שוטריה או תפקודה.
"""

    return f"topic: {topic_key}"

def safe_parse_user_stance(txt: str) -> dict:
    txt = (txt or "").strip()

    # Remove markdown fences if GPT adds them
    txt = txt.replace("```json", "").replace("```", "").strip()

    # First try normal JSON
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Try extracting JSON-looking substring
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        sub = txt[start:end + 1]
        try:
            data = json.loads(sub)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Last-resort regex fallback
    lowered = txt.lower()

    stance = "neutral_or_mixed"
    for label in ["pro", "anti", "neutral_or_mixed", "irrelevant"]:
        if re.search(rf'"?stance_label"?\s*:\s*"?{re.escape(label)}"?', lowered):
            stance = label
            break

    conf = 0.0
    m = re.search(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', lowered)
    if m:
        try:
            conf = float(m.group(1))
        except Exception:
            conf = 0.0

    reason = "malformed_json_fallback"
    m = re.search(r'"?short_reason"?\s*:\s*"([^"]*)"', txt)
    if m:
        reason = m.group(1)

    return {
        "stance_label": stance,
        "confidence": conf,
        "short_reason": reason,
        "raw_response": txt[:500],
    }

def classify_user_stance_with_gpt(
    user_opinion: str,
    topic_key: str,
    model: str = "gpt-5-mini",
) -> dict:
    prompt = f"""
    {topic_definition_for_user(topic_key)}

    Context:
    This opinion was written by the user in the input field for this exact topic: {topic_key}.
    Assume the user is referring to this topic, even if the topic name is not repeated in the sentence.

    User opinion:
    {user_opinion}

    Task:
    Classify the user's stance toward this topic.

    Interpretation examples:
    - "super important" = pro
    - "very important" = pro
    - "we need this" = pro
    - "I support it" = pro
    - "nonsense" = anti
    - "stupid movement" = anti
    - "this is useless" = anti
    - "I don't know" = neutral_or_mixed

    Return ONLY valid JSON.
    Do not include markdown.
    Do not include explanation outside the JSON.

    Required format:
    {{
    "stance_label": "pro",
    "confidence": 0.95,
    "short_reason": "brief reason"
    }}
    """

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_USER_STANCE},
                {"role": "user", "content": prompt},
            ],
        )

        txt = resp.choices[0].message.content.strip()
        parsed = safe_parse_user_stance(txt)

    except Exception as e:
        print("User stance classification failed:", repr(e))
        return {
            "stance_label": "neutral_or_mixed",
            "confidence": 0.0,
            "short_reason": "api_or_parse_error",
        }

    allowed = {"pro", "anti", "neutral_or_mixed", "irrelevant"}

    stance = str(parsed.get("stance_label", "neutral_or_mixed")).lower().strip()
    if stance not in allowed:
        stance = "neutral_or_mixed"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return {
        "stance_label": stance,
        "confidence": confidence,
        "short_reason": str(parsed.get("short_reason", "missing_reason")),
    }

def opposite_stance_of(label: str) -> Optional[str]:
    label = str(label).strip().lower()

    if label == "pro":
        return "anti"

    if label == "anti":
        return "pro"

    return None

# ===============================
# NLI (multilingual) – fast proxy
# ===============================
@st.cache_resource(show_spinner=False)
def load_nli(model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # mdl = SentenceTransformer(model_name, device=device)
    # return mdl
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    return tok, mdl

@torch.inference_mode()
def nli_contradiction_probs_batch(
    tok,
    mdl,
    premises: List[str],
    hypotheses: List[str],
    batch_size: int = 16,
) -> np.ndarray:
    """
    Batched P(contradiction) for (premise, hypothesis) pairs.
    Returns np.array of shape [len(premises)], values in [0,1].
    """
    assert len(premises) == len(hypotheses)

    all_probs = []

    id2label = getattr(mdl.config, "id2label", None)
    if id2label and isinstance(id2label, dict):
        labels = [id2label[i].lower() for i in range(len(id2label))]
        c_idx = labels.index("contradiction") if "contradiction" in labels else 2
    else:
        c_idx = 2

    for s in range(0, len(premises), batch_size):
        p = premises[s:s + batch_size]
        h = hypotheses[s:s + batch_size]

        batch = tok(
            p,
            h,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        logits = mdl(**batch).logits
        probs_tensor = torch.softmax(logits, dim=-1)[:, c_idx].detach().cpu()
        probs = np.asarray(probs_tensor.tolist(), dtype=np.float32)
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)

# @torch.inference_mode()
# def nli_contradiction_proxy(
#     premise: str,
#     hypotheses: List[str],
#     nli_model: SentenceTransformer,
#     batch_size: int = 48
# ) -> np.ndarray:
#     """
#     Fast proxy using a SBERT-style NLI checkpoint:
#     encode [premise, hypothesis] pairs and map to a 0..1 'contradiction-ish' score.
#     (If you have a proper XNLI cross-encoder with logits, swap this for true P(contradiction).)
#     """
#     pairs = [[premise, h] for h in hypotheses]
#     embs = nli_model.encode(
#         pairs, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False
#     )
#     if embs.dim() == 2:
#         # crude mapping to [0..1]
#         scores = torch.tanh(-embs.norm(dim=1))
#         probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
#     else:
#         probs = torch.full((len(hypotheses),), 0.5)

#     return probs.detach().cpu().numpy().astype("float32")

# def rerank_with_nli_only(
#     user_text: str,
#     pool_rows: List[Dict],
#     *,
#     nli_model: SentenceTransformer,
#     cosine_weight: float = 0.3,
#     nli_weight: float = 0.7
# ) -> List[Dict]:
#     if not pool_rows:
#         return []

#     hyps = [str(r.get("message", "")) for r in pool_rows]
#     nli_p = nli_contradiction_proxy(user_text, hyps, nli_model=nli_model, batch_size=48)

#     order = np.argsort(-nli_p)
#     ranked: List[Dict] = []
#     for i in order.tolist():
#         r = dict(pool_rows[i])
#         r["nli_contradiction"] = float(nli_p[i])
#         r["combined_score"] = float(nli_p[i])  # final blend happens after GPT
#         ranked.append(r)

#     return ranked

def rerank_with_nli_only(
    user_text: str,
    pool_rows: List[Dict],
    *,
    nli_model,
) -> List[Dict]:
    if not pool_rows:
        return []

    tok, mdl = nli_model

    hypotheses = [str(r.get("comment_text", "")) for r in pool_rows]
    premises = [user_text] * len(pool_rows)

    nli_p = nli_contradiction_probs_batch(
        tok,
        mdl,
        premises,
        hypotheses,
        batch_size=16,
    )

    order = np.argsort(-nli_p)  # descending

    ranked = []
    for i in order.tolist():
        r = dict(pool_rows[i])
        r["nli_contradiction"] = float(nli_p[i])
        r["combined_score"] = float(nli_p[i])  # final blend happens after GPT
        ranked.append(r)

    return ranked


def build_pool_topic_only(
    *,
    meta: pd.DataFrame,
    topic_kws: List[str],
    K: int,
    text_col: str = "message",
    require_all: bool = False,
    random_seed: int = 42,
) -> List[Dict]:
    """Candidate pool by keyword-topic filter ONLY (no embeddings, no ANN). Returns up to K rows."""
    titles = meta[text_col].astype(str)
    kws = [k for k in (topic_kws or []) if k]

    if not kws:
        idxs = np.arange(len(meta))
    else:
        if require_all:
            mask = np.ones(len(meta), dtype=bool)
            for kw in kws:
                mask &= titles.str.contains(re.escape(kw), case=False, na=False).values
        else:
            mask = np.zeros(len(meta), dtype=bool)
            for kw in kws:
                mask |= titles.str.contains(re.escape(kw), case=False, na=False).values
        idxs = np.flatnonzero(mask)

    if idxs.size == 0:
        return []

    rng = np.random.default_rng(random_seed)
    if idxs.size > K:
        idxs = rng.choice(idxs, size=K, replace=False)

    pool: List[Dict] = []
    for i in idxs.tolist():
        row = meta.iloc[i].to_dict()
        row["row_index"] = i
        pool.append(row)
    return pool

# # @st.cache_resource(show_spinner=True)
# def _load_artifacts(path: str):
#     meta, embs, index, encoder = load_hebrew(path)  # expects config-trained encoder, l2-normalized embs
#     return meta, embs, index, encoder

def build_pool_opposite_stance(
    *,
    meta: pd.DataFrame,
    opposite_stance: str,
    K: int,
    text_col: str = "message",
    stance_col: str = "stance_label",
    min_confidence: float = 0.60,
    random_seed: int = 42,
) -> List[Dict]:
    """
    Candidate pool using precomputed stance labels.
    Keeps only rows with stance_label == opposite_stance,
    then selects the strongest candidates by GPT/diversity scores
    instead of random sampling.
    """

    if stance_col not in meta.columns:
        raise ValueError(
            f"meta missing '{stance_col}'. "
            "Use the GPT-classified meta.parquet with pro/anti labels."
        )

    if text_col not in meta.columns:
        raise ValueError(f"meta missing text column '{text_col}'.")

    labels = meta[stance_col].fillna("").astype(str).str.lower().str.strip()
    mask = labels.eq(opposite_stance)

    if "confidence" in meta.columns:
        conf = pd.to_numeric(meta["confidence"], errors="coerce").fillna(0.0)
        mask = mask & (conf >= min_confidence)

    candidate_df = meta.loc[mask].copy()

    if candidate_df.empty:
        print(f"No rows found for opposite stance: {opposite_stance}")
        return []

    candidate_df["_row_index"] = candidate_df.index

    # These are the best columns you already have.
    # Higher is better except redundancy_risk, where lower is better.
    for col in [
        "gpt_final_score",
        "keep_score",
        "persona_usefulness",
        "coverage_score",
        "uniqueness_score",
        "clarity_score",
        "redundancy_risk",
        "hnsw_similarity",
        "topic_candidate_score",
    ]:
        if col in candidate_df.columns:
            candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce").fillna(0.0)

    # Build a logical runtime score for choosing NLI candidates.
    # This favors comments that were already judged useful, clear, unique,
    # representative, and not redundant.
    candidate_df["_runtime_pool_score"] = 0.0

    if "gpt_final_score" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 2.0 * candidate_df["gpt_final_score"]

    if "keep_score" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 1.5 * candidate_df["keep_score"]

    if "persona_usefulness" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 1.2 * candidate_df["persona_usefulness"]

    if "coverage_score" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 1.0 * candidate_df["coverage_score"]

    if "uniqueness_score" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 0.8 * candidate_df["uniqueness_score"]

    if "clarity_score" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 0.5 * candidate_df["clarity_score"]

    if "redundancy_risk" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] -= 0.8 * candidate_df["redundancy_risk"]

    if "hnsw_similarity" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 0.5 * candidate_df["hnsw_similarity"]

    if "topic_candidate_score" in candidate_df.columns:
        candidate_df["_runtime_pool_score"] += 0.3 * candidate_df["topic_candidate_score"]

    # Keep best K instead of random K
    candidate_df = candidate_df.sort_values(
        "_runtime_pool_score",
        ascending=False
    ).head(K)

    pool: List[Dict] = []

    for _, row in candidate_df.iterrows():
        r = row.to_dict()
        r["row_index"] = int(row["_row_index"])
        pool.append(r)

    print(
        f"Built opposite-stance pool: stance={opposite_stance}, "
        f"rows={len(pool)}, selection=score_based"
    )

    return pool

def make_topic_mask_hebrew(
    meta: pd.DataFrame,
    topic_query: str | Iterable[str],
    require_all: bool = True
) -> np.ndarray:
    """
    Build a boolean mask over meta['message'] (Hebrew dataset) for rows containing the keywords.
    - topic_query: string with words OR an iterable of keywords.
    - require_all: True => all keywords must appear (AND), False => any keyword (OR).
    """
    titles = meta["message"].astype(str)

    if isinstance(topic_query, str):
        kws = [t for t in re.split(r"[\W_]+", topic_query) if t]
    else:
        kws = [str(t) for t in topic_query if str(t).strip()]

    if not kws:
        return np.ones(len(meta), dtype=bool)

    if require_all:
        mask = np.ones(len(meta), dtype=bool)
        for kw in kws:
            mask &= titles.str.contains(re.escape(kw), case=False, na=False).values
    else:
        mask = np.zeros(len(meta), dtype=bool)
        for kw in kws:
            mask |= titles.str.contains(re.escape(kw), case=False, na=False).values
    return mask

def other_comments_same_author_same_topic(
    meta: pd.DataFrame,
    topic_query: str | Iterable[str],
    *,
    base_row: dict,                 # from your opposite finder; must contain 'commenter_id' (and optionally 'row_index')
    require_all_keywords: bool = False,
    text_col: str = "message",
    id_col: str = "commenter_id",
    limit: Optional[int] = None,
) -> List[str]:
    """
    Return ONLY the message strings of other comments by the same commenter_id
    that also match the same topic keywords.
    - Skips if commenter_id == '-1' or empty.
    - Excludes the base row (by row_index if present; else by exact text match).
    - If 'date' exists in meta, sorts by date desc (newest first).
    """
    if id_col not in meta.columns:
        raise ValueError(f"meta missing '{id_col}'")
    if text_col not in meta.columns:
        raise ValueError(f"meta missing '{text_col}'")
    
    cid = str(base_row.get(id_col, "")).strip()
    print(cid)
    if cid in ("", "-1"):
        return []
    
    author_mask = meta[id_col].astype(str).str.strip().eq(cid).values
    topic_mask  = make_topic_mask_hebrew(meta, topic_query, require_all=require_all_keywords)

    mask = author_mask & topic_mask

    # exclude the base row
    base_idx = base_row.get("row_index", None)
    if isinstance(base_idx, (int, np.integer)) and 0 <= int(base_idx) < len(meta):
        mask[int(base_idx)] = False
    else:
        base_text = str(base_row.get(text_col, ""))
        if base_text:
            mask &= ~(meta[text_col].astype(str) == base_text).values

    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return []

    if limit is not None:
        idxs = idxs[:int(limit)]

    # return only message strings
    return meta.iloc[idxs][text_col].astype(str).tolist()

def run_opposite_pipeline_and_render(
    *,
    user_opinion: str,
    topic_keywords: str | list[str],
    meta,# embs, index, encoder,
    # retrieval / ranking knobs
    pool_size: int = 60,#200,          
    k2_short: int = 8,#15,#30,            # shortlist size for GPT (K2)
    nli_model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    use_gpt: bool = True,
    gpt_model: str = "gpt-5-mini",
    # blend weights 
    beta: float  = 0.3,            # NLI weight
    gamma: float = 0.5,            # GPT weight
    # display
    top_k_show: int = 3,
    # dependencies (pass your functions/clients)
    load_nli_fn    = load_nli,         # e.g., load_nli
    nli_rerank_fn  = rerank_with_nli_only,         # e.g., rerank_with_nli
    gpt_rerank_fn  = gpt_rerank_contradiction,         # e.g., gpt_rerank_contradiction
    gpt_client     = client,         # AzureOpenAI or OpenAI client
    include_author_threads: bool = False,#True,              # NEW
    other_comments_fn = other_comments_same_author_same_topic,  # NEW
    author_limit: int = 15,                            # NEW
    author_text_col: str = "message",                 # NEW
    author_id_col: str = "commenter_id",              # NEW
    require_all_keywords_for_author: bool = False,    # NEW
    on_progress=None,
):
    """
    End-to-end: validates input → NLI re-rank → optional GPT re-rank →
    final blend → RENDERS results in Streamlit → returns (enriched_top, timings_dict).

    Return shape (NEW):
      enriched_top = [
        { "row": <top item dict>, "other_by_author": [str, ...] },
        ...
      ]
    """

    # --- 0) validation ---
    if meta is None or not isinstance(meta, pd.DataFrame) or meta.empty:
        print("meta not loaded or empty")

    if "message" not in meta.columns:
        print(f"meta missing 'message' column, meta colums {list(meta.columns)}")

    if not user_opinion or not user_opinion.strip():
        return [], {"error": "empty_opinion"}

    # normalize topic keywords to list[str]
    if isinstance(topic_keywords, str):
        kws = [k.strip() for k in topic_keywords.split(",") if k.strip()]
    else:
        kws = [str(k).strip() for k in topic_keywords if str(k).strip()]

    timings: Dict[str, int] = {}
    t0 = time.time()

    def _prog(p: int, m: str):
        if on_progress is not None:
            try:
                on_progress(int(p), str(m))
            except Exception:
                pass

    _prog(5, "מתחילים…")

    # --- 1) Topic pool (NO cosine) ---
    _prog(20, "מעלים את סביבת השיחה")
    topic_key_for_stance = kws[0] if kws else ""

    user_stance_info = classify_user_stance_with_gpt(
        user_opinion=user_opinion,
        topic_key=topic_key_for_stance,
        model=gpt_model,
    )

    user_stance = str(user_stance_info.get("stance_label", "neutral_or_mixed")).lower().strip()
    opposite_stance = opposite_stance_of(user_stance)

    timings["user_stance"] = user_stance
    timings["user_stance_confidence"] = float(user_stance_info.get("confidence", 0.0) or 0.0)
    timings["user_stance_reason"] = str(user_stance_info.get("short_reason", ""))

    print("User stance:", user_stance_info)
    print("Opposite stance:", opposite_stance)

    if opposite_stance is None:
        # fallback for unclear users: use both pro and anti, but this is less ideal
        print("User stance unclear. Falling back to topic-only pool.")
        pool_rows = build_pool_topic_only(
            meta=meta,
            topic_kws=kws,
            K=pool_size,
            text_col="message"
        )
    else:
        pool_rows = build_pool_opposite_stance(
            meta=meta,
            opposite_stance=opposite_stance,
            K=pool_size,
            text_col="message",
            stance_col="stance_label",
            min_confidence=0.60,
            random_seed=42,
        )

    t1 = time.time()
    timings["pool_ms"] = int((t1 - t0) * 1000)

    if not pool_rows:
        #st.info("No candidates found. Try broader topic keywords.")
        print("No opposite-stance candidates found. Falling back to all pro/anti rows.")

        if "stance_label" in meta.columns:
            fallback_meta = meta[
                meta["stance_label"].fillna("").astype(str).str.lower().isin(["pro", "anti"])
            ].copy()
        else:
            fallback_meta = meta

        pool_rows = build_pool_topic_only(
            meta=fallback_meta,
            topic_kws=kws,
            K=pool_size,
            text_col="message"
        )

        if not pool_rows:
            return [], timings
    # pool_rows = build_pool_topic_only(meta=meta, topic_kws=kws, K=pool_size, text_col="message")
    # t1 = time.time()
    # timings["pool_ms"] = int((t1 - t0) * 1000)

    # if not pool_rows:
    #     return [], timings

    # --- 2) NLI re-rank ---
    _prog(50, "אוספים מידע רלוונטי")
    nli_model = load_nli_fn(nli_model_name)
    nli_ranked = nli_rerank_fn(
        user_text=user_opinion,
        pool_rows=pool_rows,
        nli_model=nli_model,
    )
    t2 = time.time()
    timings["nli_ms"] = int((t2 - t1) * 1000)

    shortlist = nli_ranked[: min(k2_short, len(nli_ranked))]

    # --- 3) GPT final judge (optional) ---
    if use_gpt:
        _prog(75, "מעדכנים את המערכת")

        candidates = []
        for r in shortlist:
            rid = str(r.get("row_index", len(candidates)))
            txt = str(r.get("message", ""))
            candidates.append({"id": rid, "text": txt, "topic": "", "row": r})

        gpt_ranked = gpt_rerank_fn(user_opinion, candidates, model=gpt_model, batch_size=8)

        # Final blend: β * NLI + γ * GPT
        out = []
        for item in gpt_ranked:
            r = dict(item["row"])
            nli_p   = r.get("nli_contradiction", 0.0)
            gpt_p   = float(item["gpt_contra"])
            final   = beta * nli_p + gamma * gpt_p
            r["gpt_contradiction"] = gpt_p
            r["gpt_rationale"] = item.get("gpt_rationale", "")
            r["combined_score"] = float(final)
            out.append(r)

        out.sort(key=lambda z: z["combined_score"], reverse=True)
        ranked = out
        t3 = time.time()
        timings["gpt_ms"] = int((t3 - t2) * 1000)
        timings["total_ms"] = int((t3 - t0) * 1000)
    else:
        _prog(75, "כמעט סיימנו...")
        out = []
        for r in shortlist:
            inv_cos = r.get("inv_cosine_norm", 0.0)
            nli_p   = r.get("nli_contradiction", 0.0)
            final   = beta * nli_p
            rr = dict(r)
            rr["combined_score"] = float(final)
            out.append(rr)
        out.sort(key=lambda z: z["combined_score"], reverse=True)
        ranked = out
        t3 = time.time()
        timings["gpt_ms"] = 0
        timings["total_ms"] = int((t3 - t0) * 1000)

    #     out = []
    #     for item in gpt_ranked:
    #         r = dict(item["row"])
    #         nli_p = float(r.get("nli_contradiction", 0.0))
    #         gpt_p = float(item.get("gpt_contra", 0.0))
    #         final = beta * nli_p + gamma * gpt_p

    #         r["gpt_contradiction"] = gpt_p
    #         r["gpt_rationale"] = item.get("gpt_rationale", "")
    #         r["combined_score"] = float(final)
    #         out.append(r)

    #     out.sort(key=lambda z: z["combined_score"], reverse=True)
    #     ranked = out

    #     t3 = time.time()
    #     timings["gpt_ms"] = int((t3 - t2) * 1000)
    #     timings["total_ms"] = int((t3 - t0) * 1000)
    # else:
    #     # NLI-only fallback (no GPT)
    #     _prog(75, "רק עוד רגע, כמעט שם...")
    #     out = []
    #     for r in shortlist:
    #         rr = dict(r)
    #         rr["combined_score"] = float(rr.get("nli_contradiction", 0.0))  # beta scaling doesn't change ranking
    #         out.append(rr)
    #     out.sort(key=lambda z: z["combined_score"], reverse=True)
    #     ranked = out

    #     t3 = time.time()
    #     timings["gpt_ms"] = 0
    #     timings["total_ms"] = int((t3 - t0) * 1000)

    # --- 4) Enrich top-k with same-author/same-topic history ---
    _prog(90, "רק עוד רגע, כמעט שם...")
    enriched_top: List[Dict] = []
    #debugging
    print("\n===== FINAL SELECTED OPPOSITE COMMENTS =====")
    for debug_i, debug_r in enumerate(ranked[:top_k_show], 1):
        print(f"\nTOP {debug_i}")
        print("stance_label:", debug_r.get("stance_label"))
        print("confidence:", debug_r.get("confidence"))
        print("nli_contradiction:", debug_r.get("nli_contradiction"))
        print("gpt_contradiction:", debug_r.get("gpt_contradiction"))
        print("combined_score:", debug_r.get("combined_score"))
        print("text:", str(debug_r.get("comment_text", ""))[:1000])
        #end debbugging

    for r in ranked[:top_k_show]:
        others: List[str] = []
        if include_author_threads and other_comments_fn is not None:
            try:
                others = other_comments_fn(
                    meta,
                    topic_query=kws,
                    base_row=r,
                    require_all_keywords=require_all_keywords_for_author,
                    text_col=author_text_col,
                    id_col=author_id_col,
                    limit=author_limit,
                )
            except Exception:
                others = []
        enriched_top.append({"row": r, "other_by_author": others})

    _prog(100, "סיימנו ✅")
    return enriched_top, timings




# if __name__ == "__main__":  
#     meta, embs, index, encoder = _load_artifacts("./hebrew")
    
#     ranked, timings = run_opposite_pipeline_and_render(
#         user_opinion="ביבי הורס לנו את המדינה",
#         topic_keywords=["בנימין נתניהו", "ביבי"],
#         meta=meta,
#         embs=embs,
#         index=index,
#         encoder=encoder,
#         pool_size=200,
#         k2_short=15,
#         use_gpt=True,
#         gpt_model="gpt-5-mini",
#         beta=0.3,
#         gamma=0.5,
#         top_k_show=3,
#     )

#     for i, item in enumerate(ranked, 1):
#         row = item["row"]
#         print(f"\n### TOP {i} ###")
#         print(row.get("message", ""))
#         if row.get("gpt_rationale"):
#             print("[GPT rationale]", row["gpt_rationale"])
#         for j, t in enumerate(item.get("other_by_author", []), 1):
#             print(f"[other {j}] {t}")

#     print("Timings (ms):", timings)
