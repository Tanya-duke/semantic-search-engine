"""Streamlit semantic search app over AG News."""

from __future__ import annotations

import os
import html

import streamlit as st

from search_engine import (
    SemanticSearchEngine,
    configured_dataset_limit,
    relevance_threshold,
)

ENGINE_CACHE_VERSION = "relevance-threshold-v1"


st.set_page_config(
    page_title="Signal | Semantic Search",
    page_icon="S",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
    :root { --ink: #172321; --muted: #596761; --green: #267c72; --coral: #ee6b4d; --paper: #f3f5f2; }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1040px; padding-top: 2rem; padding-bottom: 4rem; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .hero { border-top: 5px solid var(--coral); padding: 26px 0 22px; }
    .hero-kicker { color: var(--green); font: 500 12px/1.2 'DM Mono', monospace; }
    .hero h1 { margin: 10px 0 8px; color: var(--ink); font: 600 42px/1.08 'Newsreader', Georgia, serif; }
    .hero p { margin: 0; max-width: 680px; color: var(--muted); font-size: 15px; line-height: 1.6; }
    .section-label { margin: 22px 0 7px; color: var(--muted); font: 500 12px/1.4 'DM Mono', monospace; }
        [data-testid="stForm"] { border: 0; padding: 0; }
        [data-testid="stTextInput"] label,
        [data-testid="stSelectSlider"] label { color: var(--ink) !important; }
        [data-testid="stTextInput"] input {
            background: #fff !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            border-color: #9caca5 !important;
        }
        [data-testid="stTextInput"] input::placeholder { color: #64716c !important; opacity: 1; }
        [data-testid="stBaseButton-secondary"] {
            background: #fff !important;
            border: 1px solid #9caca5 !important;
            color: var(--ink) !important;
        }
        [data-testid="stBaseButton-secondary"] p { color: var(--ink) !important; }
        [data-testid="stBaseButton-secondary"]:hover {
            background: #e4efeb !important;
            border-color: var(--green) !important;
            color: var(--ink) !important;
        }
        [data-testid="stBaseButton-primary"] {
            background: var(--green) !important;
            border-color: var(--green) !important;
            color: #fff !important;
        }
        [data-testid="stBaseButton-primary"] p { color: #fff !important; }
        [data-testid="stSlider"] [data-baseweb="slider"] { color: var(--green) !important; }
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color: var(--ink) !important; }
    .result-card { margin: 12px 0; padding: 16px 18px; background: #fff; border: 1px solid #dce3de; border-left: 4px solid var(--green); border-radius: 4px; }
    .result-meta { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 9px; }
    .result-source { color: var(--green) !important; font: 500 11px/1.3 'DM Mono', monospace; text-transform: uppercase; text-decoration: none; }
    .result-source:hover { text-decoration: underline; }
    .result-score { color: var(--muted); font: 400 12px/1.3 'DM Mono', monospace; white-space: nowrap; }
    .result-text { color: #27332f; font-size: 14px; line-height: 1.65; }
    @media (max-width: 640px) { .hero h1 { font-size: 34px; } .result-card { padding: 14px; } }
    </style>
    <header class="hero">
      <div class="hero-kicker">AG NEWS · VECTOR RETRIEVAL</div>
      <h1>Find the story behind the words.</h1>
      <p>Search AG News articles by meaning, not just exact keywords. The default index covers 500 articles, embedded with MiniLM and ranked in Chroma.</p>
    </header>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_engine(dataset_limit: int, db_path: str, cache_version: str) -> SemanticSearchEngine:
    return SemanticSearchEngine(dataset_limit=dataset_limit, db_path=db_path)


examples = (
    "space exploration and satellites",
    "major business mergers",
    "international elections",
    "championship sports results",
)
st.markdown('<div class="section-label">TRY A SEARCH</div>', unsafe_allow_html=True)
example_columns = st.columns(4)
for column, example in zip(example_columns, examples):
    if column.button(example, use_container_width=True):
        st.session_state["search_query"] = example

query_column, relevance_column = st.columns([5, 1.5])
with query_column:
    with st.form("semantic-search", clear_on_submit=False):
        input_column, submit_column = st.columns([5, 1])
        with input_column:
            query = st.text_input(
                "Search the news",
                placeholder="e.g. new treatments for a disease",
                key="search_query",
            )
        with submit_column:
            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

with relevance_column:
    relevance = st.slider(
        "Relevance",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        key="relevance_level",
    )
    st.caption("0 broad · 10 strict")

if submitted:
    st.session_state["submitted_query"] = query.strip()

submitted_query = st.session_state.get("submitted_query", "")
if submitted and not submitted_query:
    st.info("Enter a search phrase to find related articles.")
elif submitted_query:
    try:
        with st.spinner(
            "First search: downloading AG News and MiniLM, then building the vector index. "
            "This can take a few minutes."
        ):
            engine = get_engine(
                configured_dataset_limit(),
                os.getenv("CHROMA_DB_DIR", "chroma_db"),
                ENGINE_CACHE_VERSION,
            )
        with st.spinner("Finding related articles..."):
            threshold = relevance_threshold(relevance)
            matches = engine.search(submitted_query, int(relevance))

        st.caption(
            f"{len(matches):,} matches above {threshold:.0%} similarity "
            f"({relevance}/10) from {engine.indexed_chunks:,} indexed text chunks."
        )
        if not matches:
            st.info("No matches meet this relevance level. Move the control toward 0 for broader results.")
        for match in matches:
            category = html.escape(str(match["category"]))
            text = html.escape(str(match["text"]))
            article_index = int(match["article_id"])
            article_id = article_index + 1
            source_url = (
                "https://huggingface.co/datasets/fancyzhx/ag_news/viewer/default/train"
                f"?row={article_index}"
            )
            score = float(match["similarity"]) * 100
            st.markdown(
                f"""
                <article class="result-card">
                  <div class="result-meta">
                    <a class="result-source" href="{source_url}" target="_blank" rel="noopener noreferrer">{category} · Article {article_id} ↗</a>
                    <span class="result-score">{score:.1f}% match</span>
                  </div>
                  <div class="result-text">{text}</div>
                </article>
                """,
                unsafe_allow_html=True,
            )
    except Exception as error:
        st.error(f"Search setup failed: {error}")
else:
    st.caption("Submit a search phrase to find related articles.")