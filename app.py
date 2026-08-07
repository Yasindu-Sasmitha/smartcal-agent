"""Streamlit web UI for the Smart Calculator + Web Search agent.

Run locally:        streamlit run app.py
Deploy:            push to GitHub → share.streamlit.io (free tier)
                   set GEMINI_API_KEY in Secrets before deploying
"""

from __future__ import annotations

import streamlit as st

from core import run_agent_stream_dispatch

# --------------------------------------------------------------------------- #
# Page setup                                                                  #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="SmartCal · Gemini Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styles                                                                       #
# --------------------------------------------------------------------------- #

CUSTOM_CSS = """
<style>
  .stApp header { background: transparent; }
  .block-container { padding-top: 2rem; max-width: 820px; }

  /* --- Pin the chat composer to the bottom of the viewport --- */
  /* Streamlit wraps the st.form in <form data-testid="stForm">. We pin it
     with !important so Streamlit's own styles never win, and we move it
     visually below the footer using bottom:-40px so it clears the
     Streamlit "Made with Streamlit" footer area too. */
  section.main div[data-testid="stForm"] {
      position: fixed !important;
      left: 50% !important;
      right: auto !important;
      bottom: 1rem !important;
      top: auto !important;
      transform: translateX(-50%);
      width: min(calc(100% - 2rem), 820px);
      z-index: 999999;
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 0.75rem;
      box-shadow: 0 -6px 20px rgba(0, 0, 0, 0.35);
      padding: 0.75rem 1rem;
      max-height: 60vh;
      overflow-y: auto;
  }
  /* Strip Streamlit's default form border so it doesn't double up. */
  section.main div[data-testid="stForm"] > div {
      border: none !important;
  }
  /* Light theme override (Streamlit sets data-theme on <html> or <body>). */
  [data-theme="light"] section.main div[data-testid="stForm"],
  html[data-theme="light"] section.main div[data-testid="stForm"] {
      background: #ffffff;
      border-color: #e2e8f0;
  }
  /* Also raise the chat history's scroll container above the pinned form
     so the last message is never clipped behind it. */
  section.main .block-container {
      padding-bottom: 14rem !important;
  }

  .cal-badge {
      display: inline-block;
      padding: 0.15rem 0.55rem;
      border-radius: 0.4rem;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.02em;
  }
  .cal-badge.search { background: #1e3a8a; color: #dbeafe; }
  .cal-badge.calc   { background: #14532d; color: #dcfce7; }
  .cal-badge.user   { background: #374151; color: #f9fafb; }
  .cal-badge.agent  { background: #6d28d9; color: #ede9fe; }
  .cal-badge.save   { background: #0e7490; color: #cffafe; }
  .cal-badge.recall { background: #be185d; color: #fce7f3; }
  .cal-badge.afc    { background: #86198f; color: #fae8ff; }

  .result-card {
      background: #0f172a;
      color: #e2e8f0;
      padding: 0.85rem 1rem;
      border-radius: 0.5rem;
      border: 1px solid #1e293b;
      font-family: ui-monospace, "JetBrains Mono", Consolas, monospace;
      font-size: 0.92rem;
  }
  .search-result {
      background: #0c1a36;
      padding: 0.75rem 0.95rem;
      border-radius: 0.5rem;
      border-left: 3px solid #2563eb;
      margin-bottom: 0.55rem;
  }
  .search-result .title { font-weight: 600; color: #bfdbfe; }
  .search-result .url   { font-size: 0.78rem; color: #60a5fa; }
  .search-result .body  { font-size: 0.85rem; color: #cbd5e1; margin-top: 0.2rem; }
  .empty-warn {
      background: #422006;
      color: #fde68a;
      padding: 0.65rem 0.9rem;
      border-radius: 0.5rem;
      border-left: 3px solid #f59e0b;
  }
  .agent-final {
      background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
      color: #ede9fe;
      padding: 1rem 1.25rem;
      border-radius: 0.75rem;
      border: 1px solid #4f46e5;
  }
  .afc-card {
      background: linear-gradient(135deg, #4a044e 0%, #86198f 100%);
      color: #fae8ff;
      padding: 0.85rem 1rem;
      border-radius: 0.5rem;
      border: 1px solid #c026d3;
      font-family: ui-monospace, "JetBrains Mono", Consolas, monospace;
      font-size: 0.85rem;
      margin-top: 0.5rem;
  }
  .afc-card .heading { font-weight: 600; margin-bottom: 0.4rem; }
  .afc-card .call-line { margin-left: 0.5rem; }
  .memory-card {
      background: #0e7490;
      color: #cffafe;
      padding: 0.55rem 0.85rem;
      border-radius: 0.5rem;
      border: 1px solid #06b6d4;
      margin-top: 0.5rem;
      font-size: 0.9rem;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### 🧠 SmartCal")
    st.caption("Gemini-powered agent · Calculator + Web Search")
    st.divider()
    st.markdown("**Try asking:**")
    suggestions = [
        "What is the square root of the population of Tokyo in 2024?",
        "Convert 15.7 parsecs to light-years, then to Proxima Centauri distances.",
        "Write a script to simulate tossing two fair six-sided dice 10,000 times. Calculate the experimental probability of rolling a sum of 7 or 11, and compare it to the theoretical probability.",
        "Compute 125 * 37 + cos(0).",
        "Write a haiku about recursion.",
    ]
    for s in suggestions:
        if st.button(s, key=f"suggest_{hash(s)}", use_container_width=True):
            st.session_state.pending_prompt = s
            st.rerun()
    st.divider()
    st.checkbox(
        "⚡ Use automatic function calling",
        value=False,
        key="use_afc",
        help="When enabled, the SDK drives the tool loop automatically instead of the manual dispatch.",
    )
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Created By : Yasindu Sasmitha"
    )

# --------------------------------------------------------------------------- #
# Chat state                                                                  #
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle pending prompt from sidebar — injects into the text input via a
# session_state key the form reads as its initial value.
pending = st.session_state.pop("pending_prompt", None)
if pending is not None:
    st.session_state["chat_text_input"] = pending

# --------------------------------------------------------------------------- #
# Header                                                                      #
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1.5rem 0;">
      <h1 style="margin:0; font-size: 2rem;">🧠 SmartCal Agent</h1>
      <p style="color:#94a3b8; margin:0.25rem 0 0 0;">
        Calculator + Web Search
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Render history                                                              #
# --------------------------------------------------------------------------- #


def _render_tool_call(name: str, args: dict) -> str:
    badge_map = {
        "web_search": ("search", "🔎"),
        "calculator": ("calc", "🧮"),
        "save_fact": ("save", "💾"),
        "recall_facts": ("recall", "🧠"),
    }
    badge_cls, icon = badge_map.get(name, ("calc", "🛠"))
    body = "\n".join(f"{k} = {v!r}" for k, v in args.items())
    return (
        f'<span class="cal-badge {badge_cls}">{icon} {name}</span>\n\n'
        f'<div class="result-card">{body}</div>'
    )


def _render_tool_result(name: str, payload) -> str:
    if name == "web_search":
        if payload == "__NO_RESULTS__":
            return '<div class="empty-warn">⚠ No results returned by the search backend.</div>'
        if isinstance(payload, str):  # error
            return f'<div class="empty-warn">⚠ {payload}</div>'
        # list of dicts
        rows = []
        for i, r in enumerate(payload, 1):
            title = r.get("title", "No title")
            url = r.get("href") or r.get("url") or ""
            body = r.get("body") or r.get("snippet") or "No description"
            rows.append(
                f'<div class="search-result">'
                f'<div class="title">{i}. {title}</div>'
                f'<div class="url"><a href="{url}" target="_blank">{url}</a></div>'
                f'<div class="body">{body}</div>'
                f'</div>'
            )
        return "".join(rows)
    if name == "calculator":
        if isinstance(payload, str) and payload.startswith("Error"):
            return f'<div class="empty-warn">⚠ {payload}</div>'
        return f'<div class="result-card">{payload}</div>'
    if name == "save_fact":
        return f'<div class="memory-card">💾 {payload}</div>'
    if name == "recall_facts":
        if payload == "No matching facts.":
            return '<div class="empty-warn">🧠 No matching facts.</div>'
        # Render each fact on its own line
        rows = "\n".join(line for line in str(payload).split("\n"))
        return f'<div class="memory-card">🧠 Recalled:<br><pre style="margin:0.4rem 0 0 0; white-space:pre-wrap;">{rows}</pre></div>'
    return f'<div class="result-card">{payload}</div>'


def _render_afc_summary(calls: list) -> str:
    if not calls:
        return '<div class="empty-warn">⚡ AFC: no tool calls were made.</div>'
    call_lines = "".join(
        f'<div class="call-line">• <b>{c["name"]}</b>('
        + ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
        + ")</div>"
        for c in calls
    )
    return (
        f'<span class="cal-badge afc">⚡ afc</span>\n\n'
        f'<div class="afc-card">'
        f'<div class="heading">{len(calls)} call(s) handled automatically by the SDK:</div>'
        f'{call_lines}'
        f'</div>'
    )


def _render_message(role: str, content) -> None:
    """Render a single stored message. Three shapes are supported:
    - `str` for assistant final text OR legacy text-only user prompts.
    - `dict` with `{"text", "images"}` for user turns that may carry
      attached images (images list may be empty).
    - `list[dict]` of structured events for assistant turns (tool calls,
      tool results, final text) — produced by `run_agent_stream_dispatch`.
    """
    if role == "user":
        st.markdown(
            f'<span class="cal-badge user"> you</span>',
            unsafe_allow_html=True,
        )
        # Legacy text-only user message.
        if isinstance(content, str):
            st.markdown(content)
            return
        # New multimodal user message.
        if isinstance(content, dict) and "images" in content:
            if content.get("images"):
                cols = st.columns(min(len(content["images"]), 4))
                for i, img_bytes in enumerate(content["images"]):
                    cols[i % len(cols)].image(img_bytes, width=120)
            if content.get("text"):
                st.markdown(content["text"])
            return
        # Unknown shape — fall back to a stringified display so we don't crash.
        st.markdown(str(content))
        return

    st.markdown(
        f'<span class="cal-badge agent">🤖 agent</span>',
        unsafe_allow_html=True,
    )
    if isinstance(content, str):
        st.markdown(
            f'<div class="agent-final">{content}</div>',
            unsafe_allow_html=True,
        )
        return

    # content is a list of events from run_agent_stream_dispatch
    for ev in content:
        kind = ev["type"]
        if kind == "tool_call":
            st.markdown(_render_tool_call(ev["name"], ev["args"]), unsafe_allow_html=True)
        elif kind == "tool_result":
            st.markdown(_render_tool_result(ev["name"], ev["payload"]), unsafe_allow_html=True)
        elif kind == "afc_summary":
            st.markdown(_render_afc_summary(ev.get("calls", [])), unsafe_allow_html=True)
        elif kind == "final":
            st.markdown(
                f'<div class="agent-final">{ev["text"]}</div>',
                unsafe_allow_html=True,
            )
        elif kind == "aborted":
            st.markdown(
                f'<div class="empty-warn">{ev["text"]}</div>',
                unsafe_allow_html=True,
            )


for role, content in st.session_state.messages:
    with st.chat_message(role):
        _render_message(role, content)

# --------------------------------------------------------------------------- #
# Chat input                                                                  #
# --------------------------------------------------------------------------- #
#
# `st.chat_input` is text-only and can't host a file uploader alongside it, so
# the form below is the cleanest replacement: a multi-file uploader + text
# input + send button. The text input reads its initial value from
# `st.session_state["chat_text_input"]`, which the sidebar suggestions populate.

with st.form("chat_form", clear_on_submit=True):
    uploaded = st.file_uploader(
        "Attach images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="chat_files",
    )
    text = st.text_input(
        "Ask anything…",
        label_visibility="collapsed",
        key="chat_text_input",
        placeholder="Ask anything…  (drop images above to attach)",
    )
    submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and (text or uploaded):
    images = [f.getvalue() for f in uploaded] if uploaded else []
    payload = {"text": text or "", "images": images}
    st.session_state.messages.append(("user", payload))
    with st.chat_message("user"):
        _render_message("user", payload)

    # Stream the agent.
    use_afc = st.session_state.get("use_afc", False)
    events: list[dict] = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            for event in run_agent_stream_dispatch(
                text or "(image only)", use_afc=use_afc, images=images
            ):
                events.append(event)
                # Re-render the entire trace so far inside the spinner block.
                html_parts = ['<span class="cal-badge agent">🤖 agent</span>']
                for ev in events:
                    k = ev["type"]
                    if k == "tool_call":
                        html_parts.append(_render_tool_call(ev["name"], ev["args"]))
                    elif k == "tool_result":
                        html_parts.append(_render_tool_result(ev["name"], ev["payload"]))
                    elif k == "afc_summary":
                        html_parts.append(_render_afc_summary(ev.get("calls", [])))
                    elif k == "final":
                        html_parts.append(
                            f'<div class="agent-final">{ev["text"]}</div>'
                        )
                    elif k == "aborted":
                        html_parts.append(f'<div class="empty-warn">{ev["text"]}</div>')
                placeholder.markdown("\n".join(html_parts), unsafe_allow_html=True)

    st.session_state.messages.append(("assistant", events))
