import streamlit as st
from anthropic import AnthropicVertex
import os
import gc

# 1. إعدادات الصفحة
st.set_page_config(page_title="Said AI | Engineer", page_icon="🏗️")

# 2. بيانات المشروع
PROJECT_ID = "gemini-projectsaid01"
REGION = "us-central1"
MODEL_ID = "claude-3-5-sonnet@20240620"

# 3. الربط بالمفتاح والـ Client
@st.cache_resource
def get_client():
    if os.path.exists("key.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
    # التعديل هنا: استخدام project_id ليتوافق مع النسخة الجديدة
    return AnthropicVertex(project_id=PROJECT_ID, region=REGION)

# 4. واجهة المستخدم
st.title("🏗️ مساعد المهندس سعيد رشيدي")
st.caption("Claude 3.5 Sonnet | جاهز للعمل")

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

# استقبال السؤال
if prompt := st.chat_input("اسألني أي سؤال هندسي يا بشمهندس..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            client = get_client()
            with client.messages.stream(
                model=MODEL_ID,
                max_tokens=2048,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"⚠️ تكة فنية: {str(e)}")
        finally:
            gc.collect()

    st.session_state.messages.append({"role": "assistant", "content": full_response})
