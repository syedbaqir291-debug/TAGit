"""
JCI 8th Edition Observation Tagger
Matches free-text survey observations to JCI Measurable Elements (MEs)
using offline semantic search (no API key required).
"""

import streamlit as st
import json
import numpy as np
import os
import pickle
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JCI Observation Tagger by S M Baqir",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────────
DB_PATH       = Path("mes_database.json")
INDEX_PATH    = Path("mes_index.pkl")          # cached embeddings
MODEL_NAME    = "all-MiniLM-L6-v2"
TOP_K         = 5                              # candidates from embedding search
FINAL_SHOW    = 2                              # MEs shown to user
MIN_SCORE     = 0.30                           # minimum similarity threshold

CHAPTER_ICONS = {
    "APR": "📋", "ACC": "🚪", "AOP": "🔍", "COP": "💊",
    "ASC": "🔧", "MMU": "💉", "PCC": "🤝", "QPS": "📊",
    "GLD": "🏛️", "FMS": "🏗️", "SQE": "👨‍⚕️", "MOI": "📁",
    "PCI": "🦠", "GHI": "🌍", "IPSG": "⚠️", "HCT": "🖥️",
    "HRP": "👥", "MPE": "🎓",
}


# ── Load data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_mes():
    with open(DB_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading AI model (first run only — takes ~30 sec)…")
def load_model_and_index(mes):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            embeddings = pickle.load(f)
    else:
        texts = [f"{m['standard']} {m['standard_title']} {m['text']}" for m in mes]
        with st.spinner("Building search index for the first time… (one-time, ~1 min)"):
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(embeddings, f)

    return model, embeddings


def cosine_similarity_scores(query_vec, corpus_vecs):
    """Fast cosine similarity using numpy (corpus already L2-normalized)."""
    return corpus_vecs @ query_vec  # dot product = cosine when both normalized


def search(query: str, model, embeddings, mes, top_k=TOP_K):
    query_vec = model.encode(query, normalize_embeddings=True)
    scores    = cosine_similarity_scores(query_vec, embeddings)
    top_idx   = np.argsort(scores)[::-1][:top_k]
    return [(mes[i], float(scores[i])) for i in top_idx]


def score_label(score: float) -> tuple[str, str]:
    if score >= 0.70: return "🟢 High", "#1a7a4a"
    if score >= 0.50: return "🟡 Medium", "#8a6d00"
    return "🔴 Low", "#a02020"


# ── Sidebar ─────────────────────────────────────────────────────────────────────
def sidebar(mes):
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Joint_Commission_International_logo.svg/200px-Joint_Commission_International_logo.svg.png",
        width=160,
    )
    st.sidebar.title("JCI Observation Tagger")
    st.sidebar.caption("8th Edition · Effective 1 January 2025")
    st.sidebar.divider()

    # Chapter filter
    all_chapters = sorted(set(m["chapter_code"] for m in mes))
    selected = st.sidebar.multiselect(
        "Filter by chapter",
        options=all_chapters,
        default=[],
        format_func=lambda c: f"{CHAPTER_ICONS.get(c,'📌')} {c}",
        help="Leave empty to search all chapters",
    )

    st.sidebar.divider()
    st.sidebar.markdown(f"**Database:** {len(mes)} MEs across {len(all_chapters)} chapters")
    st.sidebar.markdown("**Model:** all-MiniLM-L6-v2 (offline)")
    st.sidebar.markdown("**Edition:** JCI Hospital Standards 8th Ed.")

    return selected


# ── Result card ─────────────────────────────────────────────────────────────────
def render_result(rank: int, me: dict, score: float):
    label, color = score_label(score)
    pct = int(score * 100)
    icon = CHAPTER_ICONS.get(me["chapter_code"], "📌")

    with st.container():
        st.markdown(
            f"""
            <div style="
                border:1px solid {'#1a7a4a' if score>=0.70 else '#8a6d00' if score>=0.50 else '#a02020'};
                border-left: 5px solid {color};
                border-radius:8px;
                padding:16px 20px;
                margin-bottom:14px;
                background:var(--background-color)
            ">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <span style="font-size:1.1rem;font-weight:600">
                        #{rank} &nbsp; {icon} {me['standard']} — ME {me['me_number']}
                    </span>
                    <span style="font-size:0.85rem;color:{color};font-weight:600">
                        {label} &nbsp; {pct}%
                    </span>
                </div>
                <div style="font-size:0.8rem;color:#888;margin-bottom:8px">
                    {me['chapter']}
                </div>
                {"<div style='font-size:0.85rem;font-style:italic;margin-bottom:6px;color:#aaa'>" + me['standard_title'][:120] + "…</div>" if me.get('standard_title') else ""}
                <div style="font-size:0.95rem;line-height:1.6">
                    {me['text'][:450]}{"…" if len(me['text'])>450 else ""}
                </div>
                <div style="margin-top:10px;font-size:0.78rem;color:#666">
                    ID: <code>{me['id']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    mes = load_mes()

    selected_chapters = sidebar(mes)

    st.title("🏥 JCI Observation Tagger S M Baqir")
    st.markdown(
        "Type your survey observation in plain language. "
        "The tool will find the **most relevant Measurable Elements** from JCI 8th Edition."
    )
    st.divider()

    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        observation = st.text_area(
            "Observation",
            height=130,
            placeholder=(
                "e.g.  Nursing staff unable to demonstrate the crash cart check process. "
                "No documentation found for last weekly inspection."
            ),
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("##### Options")
        n_results = st.selectbox("Show top", [1, 2, 3, 4, 5], index=1, key="n_results")
        st.markdown(" ")
        search_btn = st.button("🔍 Find MEs", type="primary", use_container_width=True)

    # Batch mode expander
    with st.expander("📋 Batch mode — paste multiple observations"):
        batch_text = st.text_area(
            "One observation per line",
            height=120,
            placeholder="Observation 1\nObservation 2\nObservation 3",
        )
        batch_btn = st.button("Run batch", key="batch_btn")

    st.divider()

    # ── Single search ────────────────────────────────────────────────────────────
    if search_btn and observation.strip():
        model, embeddings = load_model_and_index(mes)

        # Filter corpus if chapters selected
        if selected_chapters:
            filtered = [(i, m) for i, m in enumerate(mes) if m["chapter_code"] in selected_chapters]
            f_idx    = [x[0] for x in filtered]
            f_mes    = [x[1] for x in filtered]
            f_emb    = embeddings[f_idx]
        else:
            f_mes = mes
            f_emb = embeddings

        results = search(observation, model, f_emb, f_mes, top_k=max(n_results + 3, 8))
        shown   = [r for r in results if r[1] >= MIN_SCORE][:n_results]

        if not shown:
            st.warning("No MEs found above the confidence threshold. Try rephrasing your observation.")
        else:
            st.subheader(f"Top {len(shown)} matched ME{'s' if len(shown)>1 else ''}")
            for rank, (me, score) in enumerate(shown, 1):
                render_result(rank, me, score)

            # Export button
            export = [
                {
                    "rank": rank,
                    "observation": observation,
                    "standard": me["standard"],
                    "me_number": me["me_number"],
                    "chapter": me["chapter"],
                    "me_text": me["text"],
                    "similarity_score": round(score, 4),
                }
                for rank, (me, score) in enumerate(shown, 1)
            ]
            st.download_button(
                "⬇ Export as JSON",
                data=json.dumps(export, indent=2),
                file_name="jci_observation_tags.json",
                mime="application/json",
            )

    elif search_btn:
        st.info("Please enter an observation above.")

    # ── Batch mode ───────────────────────────────────────────────────────────────
    if batch_btn and batch_text.strip():
        model, embeddings = load_model_and_index(mes)

        observations = [line.strip() for line in batch_text.strip().split("\n") if line.strip()]
        batch_results = []

        progress = st.progress(0)
        for i, obs in enumerate(observations):
            results = search(obs, model, embeddings, mes, top_k=5)
            top     = [r for r in results if r[1] >= MIN_SCORE][:2]
            batch_results.append({"observation": obs, "matches": top})
            progress.progress((i + 1) / len(observations))

        progress.empty()

        st.subheader(f"Batch results — {len(observations)} observations")

        export_rows = []
        for row in batch_results:
            with st.expander(f"📝 {row['observation'][:80]}…"):
                if row["matches"]:
                    for rank, (me, score) in enumerate(row["matches"], 1):
                        render_result(rank, me, score)
                        export_rows.append({
                            "observation": row["observation"],
                            "rank": rank,
                            "standard": me["standard"],
                            "me_number": me["me_number"],
                            "chapter": me["chapter"],
                            "me_text": me["text"],
                            "similarity_score": round(score, 4),
                        })
                else:
                    st.warning("No confident match found.")

        if export_rows:
            import csv, io
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=export_rows[0].keys())
            writer.writeheader()
            writer.writerows(export_rows)
            st.download_button(
                "⬇ Export batch as CSV",
                data=buf.getvalue(),
                file_name="jci_batch_tags.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
