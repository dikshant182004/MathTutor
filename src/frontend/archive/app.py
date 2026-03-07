import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from backend.agents import chatbot, retrieve_all_threads
from backend.tools.tools import ingest_pdf, get_store_info, clear_store   
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat() -> None:
    old_thread = st.session_state.get("thread_id")
    if old_thread:
        clear_store(old_thread)                      # drop RAG store for old thread

    thread_id = generate_thread_id()
    st.session_state["thread_id"]      = thread_id
    st.session_state["message_history"] = []
    st.session_state["pdf_ingested"]    = False
    add_thread(thread_id)


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "pdf_ingested" not in st.session_state:
    st.session_state["pdf_ingested"] = False

add_thread(st.session_state["thread_id"])

# Make sure uploads/ dir exists for image / audio saves
os.makedirs("uploads", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🧮 Math Tutor Agent")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

# ── PDF / Document Upload ──────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("📄 Study Material")
st.sidebar.caption(
    "Upload a PDF (textbook chapter, notes, etc.). "
    "The agent will use it to answer questions in this session."
)

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
    key="pdf_uploader",
    label_visibility="collapsed",
)

if uploaded_pdf is not None:
    thread_id   = st.session_state["thread_id"]
    file_bytes  = uploaded_pdf.read()

    with st.sidebar:
        with st.spinner(f"Chunking & embedding **{uploaded_pdf.name}** via Cohere…"):
            try:
                stats = ingest_pdf(
                    file_bytes=file_bytes,
                    thread_id=thread_id,
                    filename=uploaded_pdf.name,
                )
                st.session_state["pdf_ingested"] = True
                st.success(
                    f"✅ **{stats['filename']}** indexed\n\n"
                    f"📃 {stats['pages']} pages · 🔖 {stats['chunks']} chunks"
                )
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {exc}")

# Show persistent badge once a PDF is indexed
if st.session_state["pdf_ingested"]:
    info = get_store_info(st.session_state["thread_id"])
    if info:
        st.sidebar.info(
            f"📚 **Active doc:** {info['filename']}\n\n"
            f"🔖 {info['chunks']} chunks in vector store"
        )

# ── Conversation History ───────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("💬 My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    label = f"🗨 {thread_id[:8]}…"
    if st.sidebar.button(label, key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)
        temp: list = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp

        # Refresh pdf_ingested badge for the switched-to thread
        st.session_state["pdf_ingested"] = get_store_info(thread_id) is not None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🧮 Math Tutor Agent")

# Render conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# ── Input mode selector ────────────────────────────────────────────────────────
input_mode = st.radio(
    "Select Input Type",
    ["Text", "Image", "Audio"],
    horizontal=True,
)

text_input  = None
image_file  = None
audio_file  = None

if input_mode == "Text":
    text_input = st.chat_input("Enter your math problem…")

elif input_mode == "Image":
    image_file = st.file_uploader(
        "Upload image of question",
        type=["png", "jpg", "jpeg"],
        key="img_uploader",
    )

elif input_mode == "Audio":
    audio_file = st.file_uploader(
        "Upload audio question",
        type=["wav", "mp3", "m4a"],
        key="audio_uploader",
    )

# ══════════════════════════════════════════════════════════════════════════════
#  HANDLE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

if text_input or image_file or audio_file:

    thread_id = st.session_state["thread_id"]

    agent_payload: dict = {
        "input_mode":  None,
        "raw_text":    None,
        "image_path":  None,
        "audio_path":  None,
        "thread_id":   thread_id,          # ← passed through so retriever_node can use it
    }

    # ── TEXT ──────────────────────────────────────────────────────────────────
    if text_input:
        agent_payload["input_mode"] = "text"
        agent_payload["raw_text"]   = text_input

        st.session_state["message_history"].append({"role": "user", "content": text_input})
        with st.chat_message("user"):
            st.text(text_input)

    # ── IMAGE ──────────────────────────────────────────────────────────────────
    if image_file:
        file_path = f"uploads/{image_file.name}"
        with open(file_path, "wb") as f:
            f.write(image_file.getbuffer())

        agent_payload["input_mode"] = "image"
        agent_payload["image_path"] = file_path

        st.session_state["message_history"].append({"role": "user", "content": "[Image Uploaded]"})
        with st.chat_message("user"):
            st.image(image_file, caption="Uploaded Question")

    # ── AUDIO ──────────────────────────────────────────────────────────────────
    if audio_file:
        file_path = f"uploads/{audio_file.name}"
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())

        agent_payload["input_mode"] = "audio"
        agent_payload["audio_path"] = file_path

        st.session_state["message_history"].append({"role": "user", "content": "[Audio Uploaded]"})
        with st.chat_message("user"):
            st.audio(audio_file)

    # ── Invoke agent ───────────────────────────────────────────────────────────
    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata":     {"thread_id": thread_id},
        "run_name":     "chat_turn",
    }

    with st.chat_message("assistant"):
        def stream_response():
            for chunk, metadata in chatbot.stream(
                agent_payload,
                config=CONFIG,
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(stream_response())

    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )