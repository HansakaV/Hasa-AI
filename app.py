import streamlit as st
import wikipedia
import requests
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ─────────────────────────────────────────
# Greeting Handler
# ─────────────────────────────────────────
GREETING_MAP = {
    r"\bhi\b|\bhello\b|\bhey\b|\bwassup\b|\bsup\b|\byo\b": "Hello! I'm Hasa AI. How can I help you today?",
    r"\bhow are you\b|\bhow do you do\b": "Running great and ready to help! What's your question?",
    r"\bbye\b|\bgoodbye\b|\bsee you\b|\btake care\b": "Goodbye! Feel free to return anytime.",
    r"\bthank(s| you)\b": "You're welcome! Let me know if you need anything else.",
    r"\bwho are you\b|\bwhat are you\b|\bwhat is hasa\b": "I'm Hasa AI — an intelligent assistant powered by FLAN-T5, Wikidata, Wikipedia, and a math engine.",
    r"\bwhat can you do\b|\byour (features|abilities|capabilities)\b": "I can solve math, look up facts via Wikidata & Wikipedia, hold conversation context, and generate responses with FLAN-T5.",
}

def check_greeting(text):
    t = text.lower().strip()
    for pattern, resp in GREETING_MAP.items():
        if re.search(pattern, t):
            return resp
    return None

# ─────────────────────────────────────────
# Math Tool
# ─────────────────────────────────────────
def calculate_if_math(text):
    cleaned = text.strip().replace("^", "**")
    if re.fullmatch(r"[\d\s\+\-\*/\.\(\)\^%]+", cleaned):
        try:
            result = eval(cleaned, {"__builtins__": {}})
            return f"**Result:** {result}"
        except Exception:
            return None
    return None

# ─────────────────────────────────────────
# Intent Detection
# ─────────────────────────────────────────
def detect_intent(query):
    q = query.lower().strip()
    if re.search(r"\b(founder|co-founder|who (founded|started|created|built|invented|made))\b", q):
        return "founder"
    if re.search(r"\b(ceo|chief executive|who (runs|leads|is the head))\b", q):
        return "ceo"
    if re.search(r"\b(born|birthday|birth date|when was .* born)\b", q):
        return "born"
    if re.search(r"\b(capital (city|of)|what is the capital)\b", q):
        return "capital"
    if re.search(r"^who (is|was|are|were)\b", q):
        return "who"
    if re.search(r"^what (is|was|are|were)\b", q):
        return "what"
    if re.search(r"^when (was|did|is)\b", q):
        return "when"
    if re.search(r"^where (is|was|did)\b", q):
        return "where"
    return "general"

# ─────────────────────────────────────────
# Wikidata Lookup
# ─────────────────────────────────────────
WIKIDATA_PROPS = {
    "founder": "P112",
    "ceo":     "P169",
    "born":    "P569",
    "capital": "P36",
}

def wikidata_lookup(entity_name, prop_key):
    prop = WIKIDATA_PROPS.get(prop_key)
    if not prop:
        return None
    try:
        api = "https://www.wikidata.org/w/api.php"
        r = requests.get(api, params={"action": "wbsearchentities", "search": entity_name,
                                       "language": "en", "format": "json", "limit": 1}, timeout=5)
        results = r.json().get("search", [])
        if not results:
            return None
        qid = results[0]["id"]

        r2 = requests.get(api, params={"action": "wbgetentities", "ids": qid,
                                        "props": "claims|labels", "languages": "en", "format": "json"}, timeout=5)
        entity = r2.json().get("entities", {}).get(qid, {})
        prop_claims = entity.get("claims", {}).get(prop, [])
        if not prop_claims:
            return None

        names = []
        for claim in prop_claims[:3]:
            val = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if isinstance(val, dict) and "id" in val:
                r3 = requests.get(api, params={"action": "wbgetentities", "ids": val["id"],
                                                "props": "labels", "languages": "en", "format": "json"}, timeout=5)
                label = r3.json().get("entities", {}).get(val["id"], {}).get("labels", {}).get("en", {}).get("value")
                if label:
                    names.append(label)
            elif isinstance(val, str):
                names.append(val)
        return ", ".join(names) if names else None
    except Exception:
        return None

# ─────────────────────────────────────────
# Wikipedia Fallback
# ─────────────────────────────────────────
def search_wikipedia_fallback(query, intent):
    try:
        results = wikipedia.search(query, results=5)
        if not results:
            return None
        page = None
        for r in results:
            try:
                page = wikipedia.page(r, auto_suggest=False)
                break
            except Exception:
                continue
        if not page:
            return None
        summary = wikipedia.summary(page.title, sentences=6, auto_suggest=False)
        sentences = [s.strip() for s in summary.split(". ") if s.strip()]
        kw_map = {
            "founder": r"found(ed|er|ing)|co-found|creat(ed|or)|start(ed|er)|invented by",
            "ceo":     r"ceo|chief executive|president|serves as",
            "born":    r"\bborn\b|\bbirth\b",
            "capital": r"capital",
        }
        kw = kw_map.get(intent)
        if kw:
            for sent in sentences:
                if re.search(kw, sent, re.I):
                    return sent.strip() + "."
        if intent in ("who", "what", "when", "where"):
            return ". ".join(sentences[:2]).strip() + "."
        return ". ".join(sentences[:3]).strip() + "."
    except Exception:
        return None

# ─────────────────────────────────────────
# Smart Answer Engine
# ─────────────────────────────────────────
def smart_answer(query):
    intent = detect_intent(query)
    subject = query
    for remove in ["who founded", "who created", "who started", "who built", "who invented",
                   "who is the ceo of", "who runs", "who leads", "founder of",
                   "when was", "born", "what is the capital of", "the"]:
        subject = re.sub(re.escape(remove), "", subject, flags=re.I).strip()
    subject = subject.strip("?. ").strip()

    wikidata_prop = {"founder": "founder", "ceo": "ceo", "born": "born", "capital": "capital"}.get(intent)
    if wikidata_prop and subject:
        result = wikidata_lookup(subject, wikidata_prop)
        if result:
            labels = {
                "founder": f"The founder(s) of **{subject.title()}**: **{result}**",
                "ceo":     f"The CEO of **{subject.title()}**: **{result}**",
                "born":    f"**{subject.title()}** was born on: **{result}**",
                "capital": f"The capital of **{subject.title()}**: **{result}**",
            }
            return labels.get(wikidata_prop, result)

    wiki = search_wikipedia_fallback(query, intent)
    return wiki

# ─────────────────────────────────────────
# FLAN-T5 Generator
# ─────────────────────────────────────────
def generate_with_model(prompt, temperature=0.7, max_tokens=200):
    if "memory" not in st.session_state:
        st.session_state.memory = []
    context = " | ".join(st.session_state.memory[-4:]) if st.session_state.memory else ""
    formatted = (f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:" if context
                 else f"Question: {prompt}\n\nAnswer:")
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, min_length=10,
            temperature=temperature, top_p=0.92, top_k=50, do_sample=True,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            num_beams=4, no_repeat_ngram_size=3, length_penalty=1.2, early_stopping=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    for marker in ["Answer:", "Question:", "Context:"]:
        if marker in response:
            response = response.split(marker)[-1].strip()
    return response if len(response) >= 8 else None

# ─────────────────────────────────────────
# Hasa AI Core
# ─────────────────────────────────────────
def hasa_ai(prompt, temperature=0.7, max_tokens=200):
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a question or statement."
    g = check_greeting(prompt)
    if g:
        return g
    m = calculate_if_math(prompt)
    if m:
        return m
    smart = smart_answer(prompt)
    if smart:
        if "memory" not in st.session_state:
            st.session_state.memory = []
        st.session_state.memory.append(prompt)
        st.session_state.memory.append(smart[:200])
        return smart
    r = generate_with_model(prompt, temperature, max_tokens)
    if "memory" not in st.session_state:
        st.session_state.memory = []
    st.session_state.memory.append(prompt)
    if r:
        st.session_state.memory.append(r[:200])
        return r
    return "I don't have enough information to answer that. Please try rephrasing."

# ═══════════════════════════════════════════
#  UI  —  PREMIUM DARK  +  AMBER
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="Hasa AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&family=Playfair+Display:wght@700;900&display=swap');

:root {
    --bg:          #0E0F13;
    --surface:     #16181F;
    --surface2:    #1E2028;
    --border:      #2A2D38;
    --border-hi:   #F59E0B;
    --amber:       #F59E0B;
    --amber-dim:   #92400E22;
    --amber-glow:  #F59E0B33;
    --text:        #F0F0F5;
    --sub:         #6B7280;
    --green:       #10B981;
    --font-body:   'Lexend', sans-serif;
    --font-display:'Playfair Display', serif;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

/* Subtle noise grain on the whole page */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }
[data-testid="stSidebar"] * {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--sub) !important;
    margin: 1.2rem 0 0.6rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stCaption { 
    font-size: 0.78rem !important; 
    color: var(--sub) !important;
    line-height: 1.6 !important;
}

/* Sidebar logo */
.sb-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.4rem;
}
.sb-gem {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--amber) 0%, #D97706 100%);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-display);
    font-size: 1.1rem;
    color: #0E0F13;
    font-weight: 900;
    box-shadow: 0 0 18px var(--amber-glow);
}
.sb-name {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
.sb-ver {
    font-size: 0.62rem;
    color: var(--sub);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Slider ── */
[data-baseweb="slider"] [role="slider"] {
    background: var(--amber) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: 0 0 10px var(--amber-glow) !important;
}
[data-baseweb="slider"] div[data-testid] { accent-color: var(--amber) !important; }

/* ── Buttons ── */
.stButton > button {
    width: 100% !important;
    background: var(--surface2) !important;
    color: var(--sub) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
    background: var(--amber-dim) !important;
    box-shadow: 0 0 12px var(--amber-glow) !important;
}

/* ── Stat chips ── */
.stat-row {
    display: flex;
    gap: 8px;
    margin: 0.5rem 0 1rem;
}
.stat-chip {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.55rem 0.5rem;
    text-align: center;
}
.stat-val {
    font-family: var(--font-display);
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--amber);
    line-height: 1;
}
.stat-lbl {
    font-size: 0.58rem;
    color: var(--sub);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}

/* Online pill */
.online-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #022C22;
    border: 1px solid #065F46;
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.green-dot {
    width: 5px; height: 5px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:0.25; }
}

/* ── Main header ── */
.main-header {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    gap: 20px;
}
.header-title {
    font-family: var(--font-display);
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 40%, #FEF3C7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-right {
    margin-left: auto;
    text-align: right;
    padding-bottom: 4px;
}
.header-sub {
    font-size: 0.7rem;
    color: var(--sub);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}
.header-model {
    font-size: 0.68rem;
    color: var(--border);
    letter-spacing: 0.06em;
    margin-top: 3px;
}

/* ── Welcome banner ── */
.welcome-banner {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 2rem;
    font-size: 0.83rem;
    color: #9CA3AF;
    line-height: 1.9;
}
.welcome-banner strong { color: var(--text); font-weight: 600; }
.chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 1px 8px;
    font-size: 0.75rem;
    color: var(--amber);
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    margin: 0 2px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1rem !important;
}
[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] .stMarkdown li {
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    line-height: 1.8 !important;
    color: var(--text) !important;
    font-weight: 300 !important;
}
[data-testid="stChatMessage"] .stMarkdown strong {
    color: var(--amber) !important;
    font-weight: 600 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--amber) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}

/* ── Chat avatars ── */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    border-color: var(--amber) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0 0 transparent !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px var(--amber-glow) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    font-weight: 300 !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--sub) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; opacity: 1 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Init state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []

# ══════════════════════════════
#  SIDEBAR
# ══════════════════════════════
with st.sidebar:
    msg_count = len(st.session_state.messages)
    mem_count = len(st.session_state.get("memory", []))

    st.markdown("""
    <div class="sb-logo">
        <div class="sb-gem">H</div>
        <div>
            <div class="sb-name">Hasa AI</div>
            <div class="sb-ver">v5.0 · FLAN-T5</div>
        </div>
    </div>
    <div class="online-pill"><div class="green-dot"></div>System Online</div>
    """, unsafe_allow_html=True)

    st.markdown("### Generation")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05,
                            help="Higher = creative · Lower = focused")
    max_tokens = st.slider("Max Tokens", 50, 500, 200, 25)

    st.markdown("### Session")
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip">
            <div class="stat-val">{msg_count}</div>
            <div class="stat-lbl">Messages</div>
        </div>
        <div class="stat-chip">
            <div class="stat-val">{mem_count}</div>
            <div class="stat-lbl">Memory</div>
        </div>
        <div class="stat-chip">
            <div class="stat-val">{"GPU" if device == "cuda" else "CPU"}</div>
            <div class="stat-lbl">Device</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Actions")
    if st.button("↺  Clear Memory"):
        st.session_state.memory = []
        st.rerun()
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("⊘  Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()

    st.markdown("<hr style='margin:1.5rem 0 1rem'>", unsafe_allow_html=True)
    st.markdown("""
    <p>Powered by FLAN-T5 Base · Wikidata<br>
    Wikipedia · Streamlit · HuggingFace</p>
    """, unsafe_allow_html=True)

# ══════════════════════════════
#  MAIN AREA
# ══════════════════════════════
st.markdown("""
<div class="main-header">
    <div class="header-title">Hasa AI</div>
    <div class="header-right">
        <div class="header-sub">Intelligent Assistant</div>
        <div class="header-model">FLAN-T5 Base · Wikidata · Wikipedia</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-banner">
        <strong>Welcome.</strong> Ask me anything — I combine Wikidata's structured knowledge,
        Wikipedia's depth, and FLAN-T5's language understanding to give you accurate answers.<br><br>
        Try:
        <span class="chip">who founded Meta?</span>
        <span class="chip">125 * 8 + 44</span>
        <span class="chip">what is quantum computing?</span>
        <span class="chip">who is the CEO of Apple?</span>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ──
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)

# ── Input ──
user_input = st.chat_input("Ask anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = hasa_ai(user_input, temperature=temperature, max_tokens=max_tokens)
        st.markdown(response)
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))