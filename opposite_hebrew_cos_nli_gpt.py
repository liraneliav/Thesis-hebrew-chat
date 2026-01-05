# app_opposite_streamlit.py
# -----------------------------------------------------------
# Streamlit app: "Most Opposite" comment finder (Hebrew-ready)
# Cascade: Topic mask → ANN/cosine pool → NLI re-rank → (optional) GPT re-rank
# Uses Azure OpenAI for the GPT stage.
# -----------------------------------------------------------

from __future__ import annotations
import os, json, re, time
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Torch / NLI
import torch
from sentence_transformers import SentenceTransformer

# Azure OpenAI (official openai package >= 1.0)
from openai import AzureOpenAI
from dotenv import load_dotenv

# ---------- YOUR EXISTING HELPERS ----------
# These must be available in your project:
# - load_hebrew(artifacts_dir) -> (meta, embs, index, encoder)
# - most_opposite_in_topic_hebrew(query_text, topic_query, meta, embs, encoder, index, require_all_keywords, top_k)
from opposite_hebrew_nli import load_hebrew, most_opposite_in_topic_hebrew, other_comments_same_author_same_topic  


# ===============================
# Azure OpenAI client
# ===============================
load_dotenv()
endpoint = os.getenv("ENDPOINT_URL", "https://YOUR-ENDPOINT.openai.azure.com/")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

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


# ===============================
# NLI (multilingual) – fast proxy
# ===============================
@st.cache_resource(show_spinner=False)
def load_nli(model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = SentenceTransformer(model_name, device=device)
    return mdl

@torch.inference_mode()
def nli_contradiction_proxy(
    premise: str,
    hypotheses: List[str],
    nli_model: SentenceTransformer,
    batch_size: int = 48
) -> np.ndarray:
    """
    Fast proxy using a SBERT-style NLI checkpoint:
    encode [premise, hypothesis] pairs and map to a 0..1 'contradiction-ish' score.
    (If you have a proper XNLI cross-encoder with logits, swap this for true P(contradiction).)
    """
    pairs = [[premise, h] for h in hypotheses]
    embs = nli_model.encode(
        pairs, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False
    )
    if embs.dim() == 2:
        # crude mapping to [0..1]
        scores = torch.tanh(-embs.norm(dim=1))
        probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        probs = torch.full((len(hypotheses),), 0.5)

    return probs.detach().cpu().numpy().astype("float32")

def rerank_with_nli(
    user_text: str,
    pool_rows: List[Dict],
    *,
    nli_model: SentenceTransformer,
    cosine_weight: float = 0.3,
    nli_weight: float = 0.7
) -> List[Dict]:
    if not pool_rows:
        return []

    hyps = [str(r.get("message", "")) for r in pool_rows]
    nli_p = nli_contradiction_proxy(user_text, hyps, nli_model=nli_model, batch_size=48)

    # inverse cosine (lower cosine → more opposite → higher score), min-max normalize
    cos = np.array([r.get("similarity", 0.0) for r in pool_rows], dtype="float32")
    inv_cos = -cos
    inv_cos = (inv_cos - inv_cos.min()) / (inv_cos.max() - inv_cos.min() + 1e-8)

    final = cosine_weight * inv_cos + nli_weight * nli_p
    order = np.argsort(-final)  # descending

    ranked = []
    for i in order.tolist():
        r = dict(pool_rows[i])
        r["inv_cosine_norm"] = float(inv_cos[i])
        r["nli_contradiction"] = float(nli_p[i])
        r["combined_score"] = float(final[i])
        ranked.append(r)
    return ranked


# ===============================
# Streamlit UI
# ===============================
# st.set_page_config(page_title="Most Opposite Finder (Cosine → NLI → GPT)", page_icon="🔎", layout="wide")
# st.title("🔎 Most Opposite Comment (Cosine → NLI → GPT)")

# with st.sidebar:
#     st.header("Artifacts & Models")
#     artifacts_dir = st.text_input("Artifacts folder (for load_hebrew)", "./hebrew")
#     nli_model_name = st.text_input("NLI model", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
#     gpt_model = st.text_input("Azure OpenAI model (deployment name)", "gpt-5-mini")
#     use_gpt = st.toggle("Use GPT final re-ranker", value=True)

#     st.header("Candidate sizes")
#     pool_size = st.slider("Cosine pool size (K1)", 50, 400, 200, 50)
#     k2_short = st.slider("GPT shortlist size (K2)", 10, 80, 30, 5)
#     top_k_show = st.slider("Show top K results", 1, 10, 3, 1)

#     st.header("Blend Weights")
#     alpha = st.slider("α: inverse cosine", 0.0, 1.0, 0.2, 0.05)
#     beta  = st.slider("β: NLI contradiction", 0.0, 1.0, 0.3, 0.05)
#     gamma = st.slider("γ: GPT contradiction", 0.0, 1.0, 0.5, 0.05)

# Load artifacts (your helper)
# @st.cache_resource(show_spinner=True)
def _load_artifacts(path: str):
    meta, embs, index, encoder = load_hebrew(path)  # expects config-trained encoder, l2-normalized embs
    return meta, embs, index, encoder

# try:
#     meta, embs, index, encoder = _load_artifacts(artifacts_dir)
#     st.success(f"Loaded artifacts: N={len(meta)}")
# except Exception as e:
#     st.error(f"Failed to load artifacts from '{artifacts_dir}': {e}")
#     st.stop()

# # Inputs
# col1, col2 = st.columns([2,1])
# with col1:
#     user_opinion = st.text_area("Write your opinion:", height=140, placeholder="אני חושב/ת ש...")
# with col2:
#     topic_keywords = st.text_input("Topic keywords (comma-separated, searched in 'message')", "ביבי, בנימין נתניהו")

# go = st.button("Find most opposite")

# if go:
#     if not user_opinion.strip():
#         st.warning("Please type your opinion first.")
#         st.stop()

#     kws = [k.strip() for k in topic_keywords.split(",") if k.strip()]

#     t0 = time.time()
#     with st.spinner("Retrieving candidate comments (ANN/cosine)…"):
#         pool_rows = most_opposite_in_topic_hebrew(
#             query_text=user_opinion,
#             topic_query=kws,
#             meta=meta, embs=embs, encoder=encoder, index=index,
#             require_all_keywords=False,
#             top_k=pool_size,   # we want a pool, not just 1
#         )
#     t1 = time.time()

#     if not pool_rows:
#         st.info("No candidates found. Try broader topic keywords.")
#         st.stop()

#     with st.spinner("Scoring stance contradiction with NLI…"):
#         nli_model = load_nli(nli_model_name)
#         # blend inv-cos + NLI; keep α in play by adding to cosine_weight if you prefer
#         nli_ranked = rerank_with_nli(
#             user_text=user_opinion,
#             pool_rows=pool_rows,
#             nli_model=nli_model,
#             cosine_weight=alpha + beta * 0.0,  # keep α separate in final blend, so here just neutral
#             nli_weight=1.0,                     # we’ll blend again with α, γ later
#         )
#     t2 = time.time()

#     # Shortlist for GPT
#     shortlist = nli_ranked[:min(k2_short, len(nli_ranked))]

#     # Prepare display container
#     st.subheader("Results")
#     st.caption(f"⏱ Cosine: {(t1-t0)*1000:.0f} ms · NLI: {(t2-t1)*1000:.0f} ms")

#     if use_gpt:
#         with st.spinner("Letting GPT pick the strongest contradiction…"):
#             # pack compact candidates
#             candidates = []
#             for r in shortlist:
#                 rid = str(r.get("row_index", len(candidates)))
#                 txt = str(r.get("message", ""))[:1200]
#                 candidates.append({
#                     "id": rid,
#                     "text": txt,
#                     "topic": "",
#                     "cosine": r.get("similarity", 0.0),
#                     "row": r,  # keep original row to stitch back
#                 })

#             gpt_ranked = gpt_rerank_contradiction(user_opinion, candidates, model=gpt_model, batch_size=40)

#             # Final blend: α * inv_cos + β * NLI + γ * GPT
#             out = []
#             for item in gpt_ranked:
#                 r = dict(item["row"])
#                 inv_cos = r.get("inv_cosine_norm", 0.0)
#                 nli_p   = r.get("nli_contradiction", 0.0)
#                 gpt_p   = float(item["gpt_contra"])
#                 final   = alpha * inv_cos + beta * nli_p + gamma * gpt_p
#                 r["gpt_contradiction"] = gpt_p
#                 r["gpt_rationale"] = item.get("gpt_rationale", "")
#                 r["combined_score"] = float(final)
#                 out.append(r)

#             out.sort(key=lambda z: z["combined_score"], reverse=True)
#             ranked = out
#             t3 = time.time()
#             st.caption(f"⏱ GPT: {(t3-t2)*1000:.0f} ms · Total: {(t3-t0)*1000:.0f} ms")
#     else:
#         # NLI-only mode (final blend α & β only)
#         out = []
#         for r in shortlist:
#             inv_cos = r.get("inv_cosine_norm", 0.0)
#             nli_p   = r.get("nli_contradiction", 0.0)
#             final   = alpha * inv_cos + beta * nli_p
#             rr = dict(r)
#             rr["combined_score"] = float(final)
#             out.append(rr)
#         out.sort(key=lambda z: z["combined_score"], reverse=True)
#         ranked = out

#     # Show top K
#     for i, r in enumerate(ranked[:top_k_show], 1):
#         with st.container(border=True):
#             st.markdown(f"**{i}. row_index={r.get('row_index','?')} · cosine={r.get('similarity',0.0):.4f}**")
#             st.write(r.get("message", ""))

#             cols = st.columns(4)
#             with cols[0]:
#                 if "inv_cosine_norm" in r:
#                     st.metric("inv-cos (norm)", f"{r['inv_cosine_norm']:.3f}")
#             with cols[1]:
#                 if "nli_contradiction" in r:
#                     st.metric("NLI contradiction", f"{r['nli_contradiction']:.3f}")
#             with cols[2]:
#                 if "gpt_contradiction" in r:
#                     st.metric("GPT contradiction", f"{r['gpt_contradiction']:.3f}")
#             with cols[3]:
#                 st.metric("Final score", f"{r.get('combined_score',0.0):.3f}")

#             if r.get("gpt_rationale"):
#                 with st.expander("Why GPT thinks this contradicts"):
#                     st.write(r["gpt_rationale"])


def run_opposite_pipeline_and_render(
    *,
    user_opinion: str,
    topic_keywords: str | list[str],
    meta, embs, index, encoder,
    # retrieval / ranking knobs
    pool_size: int = 200,          # ANN/cosine candidate pool (K1)
    k2_short: int = 30,            # shortlist size for GPT (K2)
    nli_model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    use_gpt: bool = True,
    gpt_model: str = "gpt-5-mini",
    # blend weights
    alpha: float = 0.2,            # inverse-cosine weight
    beta: float  = 0.3,            # NLI weight
    gamma: float = 0.5,            # GPT weight
    # display
    top_k_show: int = 3,
    # dependencies (pass your functions/clients)
    cosine_pool_fn = most_opposite_in_topic_hebrew,         # e.g., most_opposite_in_topic_hebrew
    load_nli_fn    = load_nli,         # e.g., load_nli
    nli_rerank_fn  = rerank_with_nli,         # e.g., rerank_with_nli
    gpt_rerank_fn  = gpt_rerank_contradiction,         # e.g., gpt_rerank_contradiction
    gpt_client     = client,         # AzureOpenAI or OpenAI client
    include_author_threads: bool = True,              # NEW
    other_comments_fn = other_comments_same_author_same_topic,  # NEW
    author_limit: int = 15,                            # NEW
    author_text_col: str = "message",                 # NEW
    author_id_col: str = "commenter_id",              # NEW
    require_all_keywords_for_author: bool = False,    # NEW
):
    """
    End-to-end: validates input → cosine/ANN pool → NLI re-rank → optional GPT re-rank →
    final blend → RENDERS results in Streamlit → returns (enriched_top, timings_dict).

    Return shape (NEW):
      enriched_top = [
        { "row": <top item dict>, "other_by_author": [str, ...] },
        ...
      ]
    """

    # --- 0) validation ---
    if not user_opinion or not user_opinion.strip():
        #st.warning("Please type your opinion first.")
        return [], {"error": "empty_opinion"}

    if cosine_pool_fn is None or load_nli_fn is None or nli_rerank_fn is None:
        #st.error("Internal configuration error: missing required functions.")
        return [], {"error": "missing_functions"}

    if use_gpt and (gpt_rerank_fn is None or gpt_client is None):
        #st.error("GPT re-rank is enabled but GPT client or function is missing.")
        return [], {"error": "missing_gpt"}

    # normalize topic keywords
    if isinstance(topic_keywords, str):
        kws = [k.strip() for k in topic_keywords.split(",") if k.strip()]
    else:
        kws = [str(k).strip() for k in topic_keywords if str(k).strip()]

    timings = {}
    t0 = time.time()

    # --- 1) cosine/ANN pool ---
    #with st.spinner("Retrieving candidate comments (ANN/cosine)…"):
    pool_rows = cosine_pool_fn(
        query_text=user_opinion,
        topic_query=kws,
        meta=meta, embs=embs, encoder=encoder, index=index,
        require_all_keywords=False,
        top_k=pool_size,
    )
    t1 = time.time()
    timings["cosine_ms"] = int((t1 - t0) * 1000)

    if not pool_rows:
        #st.info("No candidates found. Try broader topic keywords.")
        return [], timings

    # --- 2) NLI re-rank (fast) ---
    #with st.spinner("Scoring stance contradiction with NLI…"):
    nli_model = load_nli_fn(nli_model_name)
    # keep α for final blend with GPT; NLI-only ordering here
    nli_ranked = nli_rerank_fn(
        user_text=user_opinion,
        pool_rows=pool_rows,
        nli_model=nli_model,
        cosine_weight=0.0,
        nli_weight=1.0,
    )
    t2 = time.time()
    timings["nli_ms"] = int((t2 - t1) * 1000)

    # shortlist for GPT
    shortlist = nli_ranked[:min(k2_short, len(nli_ranked))]

    # header
    #st.subheader("Results")
    #st.caption(f"⏱ Cosine: {timings['cosine_ms']} ms · NLI: {timings['nli_ms']} ms")

    # --- 3) GPT final judge (optional) ---
    if use_gpt:
        #with st.spinner("Letting GPT pick the strongest contradiction…"):
        # pack compact candidates
        candidates = []
        for r in shortlist:
            rid = str(r.get("row_index", len(candidates)))
            txt = str(r.get("message", ""))[:1200]
            candidates.append({
                "id": rid,
                "text": txt,
                "topic": "",
                "cosine": r.get("similarity", 0.0),
                "row": r,
            })

        gpt_ranked = gpt_rerank_fn(user_opinion, candidates, model=gpt_model, batch_size=40)

        # Final blend: α * inv_cos + β * NLI + γ * GPT
        out = []
        for item in gpt_ranked:
            r = dict(item["row"])
            inv_cos = r.get("inv_cosine_norm", 0.0)
            nli_p   = r.get("nli_contradiction", 0.0)
            gpt_p   = float(item["gpt_contra"])
            final   = alpha * inv_cos + beta * nli_p + gamma * gpt_p
            r["gpt_contradiction"] = gpt_p
            r["gpt_rationale"] = item.get("gpt_rationale", "")
            r["combined_score"] = float(final)
            out.append(r)

        out.sort(key=lambda z: z["combined_score"], reverse=True)
        ranked = out
        t3 = time.time()
        timings["gpt_ms"] = int((t3 - t2) * 1000)
        timings["total_ms"] = int((t3 - t0) * 1000)
        #st.caption(f"⏱ GPT: {timings['gpt_ms']} ms · Total: {timings['total_ms']} ms")
    else:
        # NLI-only final blend: α & β
        out = []
        for r in shortlist:
            inv_cos = r.get("inv_cosine_norm", 0.0)
            nli_p   = r.get("nli_contradiction", 0.0)
            final   = alpha * inv_cos + beta * nli_p
            rr = dict(r)
            rr["combined_score"] = float(final)
            out.append(rr)
        out.sort(key=lambda z: z["combined_score"], reverse=True)
        ranked = out
        t3 = time.time()
        timings["gpt_ms"] = 0
        timings["total_ms"] = int((t3 - t0) * 1000)

    # ===== NEW: enrich the top-k with same-author/same-topic history =====
    enriched_top = []
    top_slice = ranked[:top_k_show]
    for r in top_slice:
        others = []
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

    # --- render top K (with author history) ---
    # for i, item in enumerate(enriched_top, 1):
    #     r = item["row"]
        # with st.container(border=True):
        #     st.markdown(f"**{i}. row_index={r.get('row_index','?')} · cosine={r.get('similarity',0.0):.4f}**")
        #     st.write(r.get("message", ""))

            # cols = st.columns(4)
            # with cols[0]:
            #     if "inv_cosine_norm" in r:
            #         st.metric("inv-cos (norm)", f"{r['inv_cosine_norm']:.3f}")
            # with cols[1]:
            #     if "nli_contradiction" in r:
            #         st.metric("NLI contradiction", f"{r['nli_contradiction']:.3f}")
            # with cols[2]:
            #     if "gpt_contradiction" in r:
            #         st.metric("GPT contradiction", f"{r['gpt_contradiction']:.3f}")
            # with cols[3]:
            #     st.metric("Final score", f"{r.get('combined_score',0.0):.3f}")

            # NEW: show other comments by same author
        # if item["other_by_author"]:
            #with st.expander("Other comments by this author (same topic)"):
                # for j, txt in enumerate(item["other_by_author"], 1):
                #     st.markdown(f"{j}. {txt}")

        # if r.get("gpt_rationale"):
        #     with st.expander("Why GPT thinks this contradicts"):
        #         st.write(r["gpt_rationale"])

    # return enriched items + timings
    return enriched_top, timings




if __name__ == "__main__":  
    meta, embs, index, encoder = _load_artifacts("./hebrew")
    # ranked, timings = run_opposite_pipeline_and_render(user_opinion="ביבי הרס לנו את המדינה", topic_keywords=["בנימין נתניהו","ביבי"], meta=meta, embs=embs, index=index, encoder=encoder)
    # print(ranked)
    # print(timings)

    ranked, timings = run_opposite_pipeline_and_render(user_opinion="ביבי הורס לנו את המדינה", topic_keywords=["בנימין נתניהו","ביבי"], meta=meta, embs=embs, index=index, encoder=encoder)
    all_comments = []
    for i, item in enumerate(ranked, 1):
        row = item["row"]
        print(f"\n### TOP {i} ###")
        print(row.get("message", ""))  # main opposite comment
        all_comments.append(row.get("message", ""))
        for j, t in enumerate(item.get("other_by_author", []), 1):
            print(f"[other {j}] {t}")
            all_comments.append(t)

    print(all_comments)
    print("Timings (ms):", timings)