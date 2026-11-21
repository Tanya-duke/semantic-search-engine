"""Streamlit semantic search app over AG News."""

from __future__ import annotations

import os

import streamlit as st

from search_engine import SemanticSearchEngine, configured_dataset_limit


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
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1040px; padding-top: 2rem; padding-bottom: 4rem; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .hero { border-top: 5px solid var(--coral); padding: 26px 0 22px; }
    .hero-kicker { color: var(--green); font: 500 12px/1.2 'DM Mono', monospace; }
    .hero h1 { margin: 10px 0 8px; color: var(--ink); font: 600 42px/1.08 'Newsreader', Georgia, serif; }
    .hero p { margin: 0; max-width: 680px; color: var(--muted); font-size: 15px; line-height: 1.6; }
    .section-label { margin: 22px 0 7px; color: var(--muted); font: 500 12px/1.4 'DM Mono', monospace; }
    [data-testid="stForm"] { border: 0; padding: 0; }
    [data-testid="stTextInput"] input { background: #fff; }
    [data-testid="stButton"] button[kind="primary"] { background: var(--green); border-color: var(--green); }
    .result-card { margin: 12px 0; padding: 16px 18px; background: #fff; border: 1px solid #dce3de; border-left: 4px solid var(--green); border-radius: 4px; }
    .result-meta { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 9px; }
    .result-category { color: var(--green); font: 500 11px/1.3 'DM Mono', monospace; text-transform: uppercase; }
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
def get_engine(dataset_limit: int, db_path: str) -> SemanticSearchEngine:
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

with st.form("semantic-search", clear_on_submit=False):
    query_column, count_column = st.columns([5, 1])
    with query_column:
        query = st.text_input(
            "Search the news",
            placeholder="e.g. new treatments for a disease",
            key="search_query",
        )
    with count_column:
        result_count = st.select_slider(
            "Results",
            options=list(range(1, 11)),
            value=5,
        )
    submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

if submitted:
    if not query.strip():
        st.info("Enter a search phrase to find related articles.")
    else:
        try:
            with st.spinner(
                "First search: downloading AG News and MiniLM, then building the vector index. "
                "This can take a few minutes."
            ):
                engine = get_engine(
                    configured_dataset_limit(),
                    os.getenv("CHROMA_DB_DIR", "chroma_db"),
                )
            with st.spinner("Finding related articles..."):
                matches = engine.search(query, int(result_count))

            st.caption(
                f"Showing {len(matches)} of {engine.indexed_chunks:,} indexed text chunks "
                f"from {engine.article_count:,} AG News articles."
            )
            if not matches:
                st.info("No matches found. Try a broader topic or different phrasing.")
            for match in matches:
                category = str(match["category"]).replace("<", "&lt;").replace(">", "&gt;")
                text = str(match["text"]).replace("<", "&lt;").replace(">", "&gt;")
                article_id = int(match["article_id"]) + 1
                score = float(match["similarity"]) * 100
                st.markdown(
                    f"""
                    <article class="result-card">
                      <div class="result-meta">
                        <span class="result-category">{category} · Article {article_id}</span>
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
    st.caption("The search index initializes the first time you submit a query.")