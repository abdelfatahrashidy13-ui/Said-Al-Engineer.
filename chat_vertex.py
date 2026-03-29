import streamlit as st
from anthropic import AnthropicVertex
from google.oauth2 import service_account
import json
import gc

st.set_page_config(page_title="Said AI | Engineer", page_icon="🏗️")

# بيانات المشروع
PROJECT_ID = "gemini-projectsaid01"
REGION = "us-central1"
MODEL_ID = "claude-3-5-sonnet@20240620"


@st.cache_resource
def get_client():
    if "gcp_service_account" not in st.secrets:
        raise ValueError("المفتاح gcp_service_account غير موجود في Secrets")

    creds_dict = dict(st.secrets["gcp_service_account"])

    credentials = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    return AnthropicVertex(
        project_id=PROJECT_ID,
        region=REGION,
        credentials=credentials,
    )


st.title("🏗️ مساعد المهندس سعيد رشيدي")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اسألني أي سؤال هندسي..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            client = get_client()

            with client.messages.stream(
                model=MODEL_ID,
                max_tokens=2048,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"تكة فنية: {str(e)}")

        finally:
            gc.collect()

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
