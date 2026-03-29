import gc
import html
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from uuid import uuid4

import anthropic
import streamlit as st
from anthropic import AnthropicVertex
from pypdf import PdfReader

st.set_page_config(
    page_title="Said AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background: #0f0f0f; color: #ececec; }
    #MainMenu, footer, header { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background: #171717;
        border-right: 1px solid #2a2a2a;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;
        max-width: 1200px;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .app-subtitle {
        color: #a0a0a0;
        margin-bottom: 1rem;
    }

    .stChatMessage {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    .chat-wrap {
        max-width: 920px;
        margin: 0 auto;
    }

    .msg-row {
        display: flex;
        width: 100%;
        margin: 14px 0;
        gap: 10px;
        align-items: flex-start;
    }

    .msg-row.user { justify-content: flex-end; }
    .msg-row.assistant { justify-content: flex-start; }

    .msg-card {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        max-width: 86%;
    }

    .msg-row.user .msg-card {
        flex-direction: row-reverse;
    }

    .avatar {
        width: 34px;
        height: 34px;
        min-width: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        background: #1f2937;
        border: 1px solid #374151;
    }

    .avatar.user {
        background: #1d4ed8;
        border-color: #2563eb;
    }

    .avatar.assistant {
        background: #111827;
        border-color: #374151;
    }

    .msg-bubble-wrap { width: 100%; }

    .msg-label {
        font-size: 12px;
        color: #9ca3af;
        margin-bottom: 6px;
        padding: 0 4px;
    }

    .msg-bubble {
        padding: 14px 16px;
        border-radius: 18px;
        line-height: 1.75;
        font-size: 15px;
        word-break: break-word;
        white-space: pre-wrap;
    }

    .msg-user {
        background: #343541;
        color: #f3f3f3;
        border-bottom-right-radius: 6px;
    }

    .msg-assistant {
        background: #1f1f1f;
        color: #f3f3f3;
        border-bottom-left-radius: 6px;
        border: 1px solid #2b2b2b;
    }

    .mode-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #1e293b;
        color: #cbd5e1;
        font-size: 12px;
        margin-bottom: 12px;
        border: 1px solid #334155;
    }

    .pdf-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #2a1b08;
        color: #fed7aa;
        font-size: 12px;
        margin-bottom: 10px;
        border: 1px solid #7c2d12;
    }

    .stChatInputContainer {
        background: rgba(15,15,15,0.94) !important;
        backdrop-filter: blur(8px);
        border-top: 1px solid #272727;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }

    textarea {
        background: #1c1c1c !important;
        color: #f3f3f3 !important;
        border: 1px solid #303030 !important;
        border-radius: 14px !important;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

CHAT_DIR = "chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)

MAX_HISTORY = 12
DEFAULT_MAX_TOKENS = 1024
MAX_PDF_CHARS = 12000


def safe_text(text: str) -> str:
    return html.escape(text)


def extract_pdf_text(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue

        full_text = "\n\n".join(pages).strip()
        return full_text[:MAX_PDF_CHARS] if full_text else ""
    except Exception:
        return ""


def render_copy_block(content: str):
    st.code(content, language=None)
    st.caption("انسخ النص من الصندوق أعلاه.")


def render_message(role: str, content: str, index: int):
    row_class = "user" if role == "user" else "assistant"
    bubble_class = "msg-user" if role == "user" else "msg-assistant"
    label = "أنت" if role == "user" else "المساعد"
    avatar = "👤" if role == "user" else "🤖"
    avatar_class = "user" if role == "user" else "assistant"

    st.markdown(
        f"""
        <div class="chat-wrap">
            <div class="msg-row {row_class}">
                <div class="msg-card">
                    <div class="avatar {avatar_class}">{avatar}</div>
                    <div class="msg-bubble-wrap">
                        <div class="msg-label">{label}</div>
                        <div class="msg-bubble {bubble_class}">{safe_text(content)}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if role == "assistant":
        with st.expander(f"📋 Copy الرد #{index + 1}", expanded=False):
            render_copy_block(content)


def save_chat(messages, chat_id=None):
    if not chat_id:
        chat_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]

    payload = {
        "chat_id": chat_id,
        "saved_at": datetime.now().isoformat(),
        "messages": messages,
    }

    file_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return chat_id


def load_chat(chat_id):
    file_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])
    except Exception:
        return []


def list_chats():
    chats = []
    for filename in os.listdir(CHAT_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(CHAT_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chats.append(
                    {
                        "chat_id": data.get("chat_id", filename.replace(".json", "")),
                        "saved_at": data.get("saved_at", ""),
                    }
                )
            except Exception:
                pass

    chats.sort(key=lambda x: x["saved_at"], reverse=True)
    return chats


def delete_chat(chat_id):
    file_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def write_service_account_to_tempfile() -> str | None:
    if "gcp_service_account" not in st.secrets:
        return None

    # الحل هنا: نحول AttrDict إلى dict عادي
    creds = dict(st.secrets["gcp_service_account"])

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as f:
        json.dump(creds, f)
        return f.name


def build_client(project_id: str, region: str):
    cred_path = write_service_account_to_tempfile()
    if cred_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    return AnthropicVertex(project_id=project_id, region=region)


@st.cache_resource(show_spinner=False)
def get_client(project_id: str, region: str):
    return build_client(project_id, region)


def engineer_prompts(user_input: str):
    return {
        "analysis": f"""
أنت مهندس تحليل إنشائي خبير.

حلل المشكلة الهندسية التالية وحدد:
1) وصف المشكلة
2) الأسباب المحتملة
3) المخاطر المتوقعة
4) ملاحظات مهمة قبل التنفيذ

المشكلة:
{user_input}
""",
        "solutions": f"""
أنت مهندس حلول تنفيذية خبير.

اقترح حلولًا عملية وواقعية للمشكلة التالية، مع:
1) الحلول المقترحة
2) ترتيبها حسب الأولوية
3) التنبيهات التنفيذية
4) متى نحتاج استشاري أو اختبار إضافي

المشكلة:
{user_input}
""",
        "report": f"""
أنت مهندس تقارير فنية محترف.

اكتب تقريرًا هندسيًا احترافيًا ومنظمًا عن المشكلة التالية، يشمل:
- عنوان واضح
- وصف المشكلة
- الأسباب المحتملة
- التوصيات
- الخلاصة النهائية

المشكلة:
{user_input}
""",
    }


def run_single_agent(project_id: str, region: str, model: str, prompt_text: str, max_tokens: int):
    local_client = build_client(project_id, region)
    full = ""

    with local_client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt_text}],
    ) as stream:
        for text_chunk in stream.text_stream:
            full += text_chunk

    return full


def run_hive_parallel(project_id: str, region: str, model: str, user_input: str, max_tokens: int):
    prompts = engineer_prompts(user_input)
    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(
                run_single_agent,
                project_id,
                region,
                model,
                prompt_text,
                max_tokens,
            ): role
            for role, prompt_text in prompts.items()
        }

        for future in as_completed(future_map):
            role = future_map[future]
            try:
                results[role] = future.result()
            except Exception as e:
                results[role] = f"خطأ في {role}: {e}"

    return {
        "analysis": results.get("analysis", "لا يوجد رد"),
        "solutions": results.get("solutions", "لا يوجد رد"),
        "report": results.get("report", "لا يوجد رد"),
    }


def build_user_prompt(prompt: str, pdf_name: str = "", pdf_text: str = "", include_pdf: bool = False):
    if include_pdf and pdf_text:
        return f"""
السؤال الأساسي:
{prompt}

يوجد ملف PDF مرفق بعنوان:
{pdf_name}

محتوى مستخرج من الملف:
{pdf_text}

المطلوب:
اعتمد على الملف المرفق في الإجابة، وإذا كانت المعلومات غير كافية فاذكر ذلك بوضوح.
"""
    return prompt


if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


with st.sidebar:
    st.title("💬 Said AI")

    project_id = st.text_input(
        "Project ID",
        value="gen-lang-client-0753032603",
        help="معرف مشروع Google Cloud",
    )

    region = st.selectbox(
        "Region",
        ["us-central1", "us-east5", "europe-west1"],
        index=0,
    )

    model_choice = st.selectbox(
        "Model",
        ["claude-sonnet-4-5", "claude-haiku-4-5"],
        index=0,
    )

    max_tokens = st.slider(
        "Max Tokens",
        256,
        2048,
        DEFAULT_MAX_TOKENS,
        step=128,
    )

    advanced_mode = st.toggle("🧠 تحليل متقدم (Hive)", value=False)

    st.markdown("---")
    st.subheader("📄 ملفات PDF")

    uploaded_pdf = st.file_uploader("ارفع ملف PDF", type=["pdf"])
    include_pdf_in_prompt = st.checkbox("استخدم محتوى الـ PDF في السؤال", value=True)

    if uploaded_pdf is not None:
        pdf_text = extract_pdf_text(uploaded_pdf)
        st.session_state.pdf_text = pdf_text
        st.session_state.pdf_name = uploaded_pdf.name

        if pdf_text:
            st.success(f"تم تحميل: {uploaded_pdf.name}")
            st.caption(f"تم استخراج حوالي {len(pdf_text)} حرف من الملف")
        else:
            st.warning("تم رفع الملف لكن لم أستطع استخراج نص واضح منه")

    if st.session_state.pdf_name:
        st.caption(f"الملف الحالي: {st.session_state.pdf_name}")
        if st.button("🗑️ إزالة الـ PDF", use_container_width=True):
            st.session_state.pdf_text = ""
            st.session_state.pdf_name = ""
            st.rerun()

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ محادثة جديدة", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            gc.collect()
            st.rerun()

    with c2:
        if st.button("🗑️ مسح الحالية", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            gc.collect()
            st.rerun()

    st.caption(f"عدد الرسائل الحالية: {len(st.session_state.messages)} / {MAX_HISTORY}")

    st.markdown("---")
    st.subheader("💾 المحادثات المحفوظة")

    if st.button("حفظ المحادثة الحالية", use_container_width=True):
        if st.session_state.messages:
            chat_id = save_chat(
                st.session_state.messages,
                st.session_state.current_chat_id,
            )
            st.session_state.current_chat_id = chat_id
            st.success("تم الحفظ")
        else:
            st.warning("لا توجد محادثة للحفظ")

    saved_chats = list_chats()

    if saved_chats:
        options = {
            f"{item['chat_id']} | {item['saved_at'][:19]}": item["chat_id"]
            for item in saved_chats
        }

        selected_label = st.selectbox("اختر محادثة", options=list(options.keys()))
        selected_chat_id = options[selected_label]

        c3, c4 = st.columns(2)
        with c3:
            if st.button("📂 فتح", use_container_width=True):
                st.session_state.messages = load_chat(selected_chat_id)
                st.session_state.current_chat_id = selected_chat_id
                st.rerun()

        with c4:
            if st.button("❌ حذف", use_container_width=True):
                if delete_chat(selected_chat_id):
                    if st.session_state.current_chat_id == selected_chat_id:
                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                    st.success("تم الحذف")
                    st.rerun()
    else:
        st.caption("لا توجد محادثات محفوظة بعد")


st.markdown('<div class="app-title">🤖 Said AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">مساعد هندسي عربي باستخدام Claude على Vertex AI</div>',
    unsafe_allow_html=True,
)

if advanced_mode:
    st.markdown('<div class="mode-badge">وضع التحليل المتقدم مفعّل</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="mode-badge">الوضع العادي مفعّل</div>', unsafe_allow_html=True)

if include_pdf_in_prompt and st.session_state.pdf_name:
    st.markdown(
        f'<div class="pdf-badge">سيتم استخدام PDF الحالي: {safe_text(st.session_state.pdf_name)}</div>',
        unsafe_allow_html=True,
    )

if not project_id:
    st.info("أدخل Project ID من الشريط الجانبي.")
    st.stop()

try:
    client = get_client(project_id, region)
except Exception as e:
    st.error(f"❌ فشل إنشاء الاتصال بـ Vertex AI: {e}")
    st.stop()

for idx, msg in enumerate(st.session_state.messages):
    render_message(msg["role"], msg["content"], idx)

prompt = st.chat_input("اكتب رسالتك هنا...")

if prompt:
    final_user_prompt = build_user_prompt(
        prompt=prompt,
        pdf_name=st.session_state.pdf_name,
        pdf_text=st.session_state.pdf_text,
        include_pdf=include_pdf_in_prompt,
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    render_message("user", prompt, len(st.session_state.messages) - 1)

    if advanced_mode:
        with st.chat_message("assistant"):
            st.info("🧠 جاري تشغيل فريق التحليل المتقدم بالتوازي...")

            try:
                results = run_hive_parallel(
                    project_id=project_id,
                    region=region,
                    model=model_choice,
                    user_input=final_user_prompt,
                    max_tokens=max_tokens,
                )

                tab1, tab2, tab3 = st.tabs(["🔍 التحليل", "🛠️ الحلول", "📄 التقرير"])

                with tab1:
                    st.markdown(results["analysis"])

                with tab2:
                    st.markdown(results["solutions"])

                with tab3:
                    st.markdown(results["report"])

                combined = f"""## 🔍 التحليل
{results["analysis"]}

## 🛠️ الحلول
{results["solutions"]}

## 📄 التقرير
{results["report"]}
"""

                with st.expander("📋 Copy الرد الكامل", expanded=False):
                    render_copy_block(combined)

            except Exception as e:
                combined = f"⚠️ خطأ غير متوقع في التحليل المتقدم: {e}"
                st.error(combined)
            finally:
                gc.collect()

        st.session_state.messages.append({"role": "assistant", "content": combined})

    else:
        full_response = ""

        with st.chat_message("assistant"):
            placeholder = st.empty()

            try:
                api_messages = st.session_state.messages[:-1] + [
                    {"role": "user", "content": final_user_prompt}
                ]

                with client.messages.stream(
                    model=model_choice,
                    max_tokens=max_tokens,
                    messages=api_messages,
                ) as stream:
                    for text_chunk in stream.text_stream:
                        full_response += text_chunk
                        placeholder.markdown(full_response + " ▌")

                placeholder.empty()
                render_message("assistant", full_response, len(st.session_state.messages))

            except anthropic.APIStatusError as e:
                full_response = f"⚠️ خطأ من API: {e.status_code} — {e.message}"
                placeholder.error(full_response)

            except anthropic.APIConnectionError:
                full_response = "⚠️ تعذر الاتصال. تحقق من الشبكة وإعدادات المشروع."
                placeholder.error(full_response)

            except Exception as e:
                full_response = f"⚠️ خطأ غير متوقع: {e}"
                placeholder.error(full_response)

            finally:
                gc.collect()

            if full_response:
                with st.expander("📋 Copy الرد", expanded=False):
                    render_copy_block(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.session_state.current_chat_id = save_chat(
        st.session_state.messages,
        st.session_state.current_chat_id,
)
