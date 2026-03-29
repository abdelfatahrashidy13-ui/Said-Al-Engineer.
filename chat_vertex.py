import streamlit as st
from anthropic import AnthropicVertex
import json
import os
import tempfile
import gc

st.set_page_config(
    page_title="Claude · Vertex AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MAX_HISTORY = 10
MAX_TOKENS = 1024

with st.sidebar:
    st.title("⚙️ الإعدادات")

    project_id = st.text_input("Project ID", value="gemini-projectsaid01")
    region = st.selectbox("Region", ["us-east5", "us-central1", "europe-west1"], index=0)
    model_choice = st.selectbox(
        "النموذج",
        ["claude-sonnet-4-5", "claude-haiku-4-5"],
        index=0,
    )
    max_tok = st.slider("Max Tokens للرد", 256, 2048, MAX_TOKENS, step=128)

    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        gc.collect()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Claude via Vertex AI")
st.caption("محادثة خفيفة — محمية من استهلاك الذاكرة")

if not project_id:
    st.info("👈 أدخل Project ID.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_client(proj: str, reg: str):
    # لو شغال على Streamlit Cloud
    if "gcp_service_account" in st.secrets:
        creds = dict(st.secrets["gcp_service_account"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(creds, f)
            cred_path = f.name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    return AnthropicVertex(project_id=proj, region=reg)


try:
    client = get_client(project_id, region)
except Exception as e:
    st.error(f"❌ فشل إنشاء الـ Client: {e}")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("اكتب رسالتك هنا…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with client.messages.stream(
                model=model_choice,
                max_tokens=max_tok,
                messages=st.session_state.messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_response += text_chunk
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ خطأ: {e}"
            placeholder.error(full_response)
        finally:
            gc.collect()

    st.session_state.messages.append({"role": "assistant", "content": full_response})
