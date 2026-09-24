import os
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from utils import (
    ensure_dirs,
    save_uploaded_files,
    load_all_notes,
    build_context,
    query_llm,
    load_tasks,
    save_tasks,
)

st.set_page_config(page_title="Study Coach", page_icon="S", layout="wide")

ensure_dirs()

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
    }
    .stChatMessage {
        padding-left: 0.5rem;
    }
    .response-study h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: #1f2937;
    }
    .response-plan h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: #1f2937;
    }
    .meta-note {
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
        padding-top: 0.5rem;
    }
    .sidebar-section {
        margin-bottom: 1.25rem;
    }
    .sidebar-section h2 {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .task-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        border-bottom: 1px solid #f3f4f6;
        font-size: 0.9rem;
    }
    .task-row:last-child {
        border-bottom: none;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        background: #e5e7eb;
        color: #374151;
    }
    .badge-high {
        background: #fee2e2;
        color: #991b1b;
    }
    .badge-medium {
        background: #fef3c7;
        color: #92400e;
    }
    .badge-low {
        background: #d1fae5;
        color: #065f46;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "provider" not in st.session_state:
        st.session_state.provider = "openai"
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            st.session_state.provider = "anthropic"


def render_sidebar() -> None:
    st.sidebar.markdown("## Study Coach")

    st.sidebar.markdown("### Add notes")
    uploaded = st.sidebar.file_uploader(
        "Upload PDF, TXT, MD, or CSV",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        saved = save_uploaded_files(uploaded)
        st.sidebar.caption(f"Saved {len(saved)} file(s).")

    notes = load_all_notes()
    st.sidebar.markdown("### Notes")
    if notes:
        for name in notes.keys():
            st.sidebar.caption(f"- {name}")
    else:
        st.sidebar.caption("No notes in ./data/notes.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tasks")
    tasks = load_tasks()
    with st.sidebar.form("add_task_form", clear_on_submit=True):
        task_name = st.text_input("Task", placeholder="e.g. Read chapter 4")
        task_due = st.date_input("Due", value=datetime.today())
        task_priority = st.selectbox("Priority", ["High", "Medium", "Low"])
        submitted = st.form_submit_button("Add")
        if submitted and task_name.strip():
            tasks.append({
                "task": task_name.strip(),
                "due": task_due.strftime("%Y-%m-%d"),
                "priority": task_priority,
                "status": "Open",
            })
            save_tasks(tasks)
            st.rerun()

    if tasks:
        for t in tasks:
            badge_class = f"badge badge-{t['priority'].lower()}"
            st.sidebar.markdown(
                f"<div class='task-row'><span>{t['task']}</span><span class='{badge_class}'>{t['priority']}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("No tasks yet.")


def detect_query_type(message: str) -> str:
    lower = message.lower()
    plan_keywords = ["break down", "assignment", "steps", "actionable", "plan", "schedule", "study plan", "2-hour", "hour study"]
    study_keywords = ["explain", "topic", "concept", "define", "what is", "how does", "why does", "notes"]
    if any(k in lower for k in plan_keywords):
        return "planning"
    if any(k in lower for k in study_keywords):
        return "study"
    return "study"


def format_study_response(content: str) -> str:
    return f"""
<div class="response-study">
<h3>Notes-based answer</h3>
<p><strong>Answer</strong><br/>{content}</p>
<div class="meta-note">If this feels incomplete, I can check another chunk of your notes.</div>
</div>
"""


def format_planning_response(content: str) -> str:
    return f"""
<div class="response-plan">
<h3>Actionable plan</h3>
<p>{content}</p>
<div class="meta-note">Refine priorities or add more detail if you need.</div>
</div>
"""


def render_chat() -> None:
    st.title("Study Coach")
    st.caption("Answers are based strictly on your uploaded notes.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)

    if prompt := st.chat_input("Ask about your notes or request a study plan..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        notes = load_all_notes()
        if not notes:
            reply = "I don't see any uploaded notes yet. Add files in the sidebar to get started."
        else:
            context = build_context(notes, max_chars=8000)
            try:
                raw = query_llm(prompt, context, provider=st.session_state.provider)
            except RuntimeError as e:
                raw = f"Configuration error: {e}"
            except Exception as e:
                raw = f"Error contacting model: {e}"

            qtype = detect_query_type(prompt)
            if qtype == "planning":
                reply = format_planning_response(raw)
            else:
                reply = format_study_response(raw)

        with st.chat_message("assistant"):
            st.markdown(reply, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": reply})


def main() -> None:
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
