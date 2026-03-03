import streamlit as st
import wikipedia
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------
# Load Model (only once)
# ----------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ----------------------
# Greeting / Chitchat Handler
# ----------------------
GREETING_PATTERNS = {
    r"\bhi\b|\bhello\b|\bhey\b|\bwassup\b|\bsup\b": "Hey there! I'm Hasa AI — ask me anything, do some math, or explore a topic!",
    r"\bhow are you\b|\bhow do you do\b|\bwhat's up\b": "I'm doing great and ready to help! What's on your mind?",
    r"\bbye\b|\bgoodbye\b|\bsee you\b|\btake care\b": "Goodbye! Come back anytime you need help. 👋",
    r"\bthank(s| you)\b": "You're welcome! Happy to help anytime.",
    r"\bwho are you\b|\bwhat are you\b|\bwhat is hasa\b": "I'm Hasa AI — a smart assistant powered by FLAN-T5, Wikipedia, and a math engine. Ask me anything!",
    r"\bwhat can you do\b|\bhelp\b|\byour (features|abilities|capabilities)\b": "I can:\n- 🧮 Solve math expressions\n- 📖 Search Wikipedia for facts\n- 💬 Answer general questions\n- 🧠 Hold context across our conversation",
}

def check_greeting(text):
    text_lower = text.lower().strip()
    for pattern, response in GREETING_PATTERNS.items():
        if re.search(pattern, text_lower):
            return response
    return None

# ----------------------
# Math Tool
# ----------------------
def calculate_if_math(text):
    cleaned = text.strip().replace("^", "**")
    if re.fullmatch(r"[\d\s\+\-\*/\.\(\)\^%]+", cleaned):
        try:
            result = eval(cleaned, {"__builtins__": {}})
            return f"🧮 Result: **{result}**"
        except Exception:
            return None
    return None

# ----------------------
# Wikipedia Tool
# ----------------------
def search_wikipedia(query):
    try:
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return None
        page = wikipedia.page(search_results[0], auto_suggest=False)
        summary = wikipedia.summary(page.title, sentences=4, auto_suggest=False)
        return f"📖 **{page.title}**\n\n{summary}"
    except wikipedia.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return f"📖 **{page.title}**\n\n{wikipedia.summary(page.title, sentences=4, auto_suggest=False)}"
        except Exception:
            return None
    except wikipedia.PageError:
        return None
    except Exception:
        return None

# ----------------------
# FLAN-T5 Generator
# ----------------------
def generate_with_model(prompt, temperature=0.7, max_tokens=200):
    if "memory" not in st.session_state:
        st.session_state.memory = []

    # Build context from recent memory (last 4 exchanges)
    context = " | ".join(st.session_state.memory[-4:]) if st.session_state.memory else ""

    formatted_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:" if context else f"Question: {prompt}\n\nAnswer:"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_length=10,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            early_stopping=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Strip prompt remnants
    for marker in ["Answer:", "Provide a direct and accurate answer:", "Question:"]:
        if marker in response:
            response = response.split(marker)[-1].strip()

    return response if len(response) >= 8 else None

# ----------------------
# Hasa AI Core
# ----------------------
def hasa_ai(prompt, temperature=0.7, max_tokens=200):
    prompt = prompt.strip()
    if not prompt:
        return "Please type something — I'm listening!"

    # 1. Handle greetings/chitchat
    greeting = check_greeting(prompt)
    if greeting:
        return greeting

    # 2. Math
    math_result = calculate_if_math(prompt)
    if math_result:
        return math_result

    # 3. Wikipedia
    wiki_result = search_wikipedia(prompt)
    if wiki_result:
        if "memory" not in st.session_state:
            st.session_state.memory = []
        st.session_state.memory.append(prompt)
        st.session_state.memory.append(wiki_result[:200])
        return wiki_result

    # 4. FLAN-T5
    model_response = generate_with_model(prompt, temperature, max_tokens)

    if "memory" not in st.session_state:
        st.session_state.memory = []
    st.session_state.memory.append(prompt)

    if model_response:
        st.session_state.memory.append(model_response[:200])
        return model_response

    # 5. Final fallback
    return "I'm not sure about that one. Try rephrasing, or ask me something else!"

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(
    page_title="Hasa AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* Global reset & theme */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e !important;
}
[data-testid="stSidebar"] * { color: #c0c0d0 !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #7b7bff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
}

/* Main header */
.hasa-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}
.hasa-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #7b7bff, #ff6b9d);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 24px #7b7bff44;
}
.hasa-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7b7bff, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.hasa-sub {
    font-size: 0.72rem;
    color: #555577;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
div[class*="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #13131f !important;
}

.stChatMessage .stMarkdown p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: #d0d0e8 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #0f0f1a !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'DM Mono', monospace !important;
    color: #e8e8f0 !important;
    box-shadow: 0 0 20px #7b7bff11 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7b7bff !important;
    box-shadow: 0 0 24px #7b7bff33 !important;
}

/* Buttons */
.stButton > button {
    background: #13131f !important;
    color: #7b7bff !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover {
    background: #7b7bff22 !important;
    border-color: #7b7bff !important;
    box-shadow: 0 0 12px #7b7bff33 !important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #7b7bff !important;
}

/* Metrics / stat cards */
[data-testid="stMetric"] {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: #7b7bff !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    color: #555577 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0f1f0f;
    border: 1px solid #1a3a1a;
    color: #4dff91;
    font-size: 0.7rem;
    padding: 4px 12px;
    border-radius: 100px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}
.status-dot {
    width: 6px; height: 6px;
    background: #4dff91;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Dividers */
hr { border-color: #1e1e2e !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #7b7bff; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.markdown("## ⚡ Hasa AI")
    st.markdown("---")

    st.markdown("### 🎛 Generation")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05,
                            help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max Tokens", 50, 500, 200, 25,
                           help="Max length of generated response")

    st.markdown("---")
    st.markdown("### 🧠 Memory")
    mem_count = len(st.session_state.get("memory", []))
    st.markdown(f"`{mem_count}` items in context")
    if st.button("🗑 Clear Memory"):
        st.session_state.memory = []
        st.success("Memory cleared!")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    msg_count = len(st.session_state.get("messages", []))
    col_a, col_b = st.columns(2)
    col_a.metric("Messages", msg_count)
    col_b.metric("Device", "GPU" if device == "cuda" else "CPU")

    st.markdown("---")
    st.markdown("### 🔧 Actions")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem; color:#333355; line-height:1.8;'>
    FLAN-T5 Base · Wikipedia · Math Engine<br>
    Built with Streamlit + HuggingFace
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Main Area
# ----------------------
st.markdown("""
<div class="hasa-header">
    <div class="hasa-logo">⚡</div>
    <div>
        <div class="hasa-title">Hasa AI</div>
        <div class="hasa-sub">Intelligent Assistant · v2.0</div>
    </div>
    <div style="margin-left: auto;">
        <div class="status-badge"><div class="status-dot"></div>Online</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div style="
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-left: 3px solid #7b7bff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        color: #8888aa;
        line-height: 1.8;
    ">
        👋 Welcome! I'm <strong style="color:#7b7bff">Hasa AI</strong>.<br>
        Try asking me a question, doing some math like <code>12 * 8 + 5</code>, or looking up a topic!
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(""):
            response = hasa_ai(user_input, temperature=temperature, max_tokens=max_tokens)
        st.markdown(response)

    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))