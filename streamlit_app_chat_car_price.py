import os
import json
import joblib
import difflib
import pandas as pd
import streamlit as st
from typing import Dict, Any
import google.generativeai as genai

# =========================
# KONFIGURASI GEMINI API
# =========================
API_KEY = "AIzaSyAd6HXeP_2NCM-60tjMB41CCLJ2gBMtwU8"
if not API_KEY:
    st.error("❌ API Key Gemini belum diatur.")
else:
    genai.configure(api_key=API_KEY)

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="🚘 Prediksi Harga Mobil + Chat Gemini",
    layout="wide",
    page_icon="🚗",
)

# =========================
# CSS & JS TAMBAHAN
# =========================
st.markdown("""
<style>
/* Background & Font */
div[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #00111a, #000000);
    color: #e5f2ff !important;
    font-family: 'Poppins', sans-serif;
}

/* Header animasi gradient */
.main-header {
    font-size: 2.5em;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00f5d4, #00b4d8, #48cae4, #0077b6);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 8s ease infinite;
    margin-bottom: 20px;
}
@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Tab animasi */
div[data-baseweb="tab-list"] {
    justify-content: center;
}
button[role="tab"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    margin: 0 10px !important;
    font-weight: 600 !important;
    color: #e5f2ff !important;
    transition: all 0.3s ease !important;
}
button[aria-selected="true"] {
    background: linear-gradient(90deg, #0077b6, #00b4d8) !important;
    box-shadow: 0 0 15px #00b4d8 !important;
    transform: scale(1.05);
}

/* Tombol futuristik */
button[kind="primary"] {
    background: linear-gradient(90deg, #0077b6, #00b4d8);
    color: white;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.3s ease;
}
button[kind="primary"]:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00b4d8;
}

/* Chat container */
.chat-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 0 25px rgba(0, 180, 216, 0.2);
    max-height: 420px;
    overflow-y: auto;
    transition: all 0.4s ease;
}

/* Chat bubble user */
.chat-bubble-user {
    background: linear-gradient(135deg, #0077b6, #0096c7);
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 10px 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    animation: fadeInRight 0.4s ease;
}
@keyframes fadeInRight {
    from {opacity: 0; transform: translateX(30px);}
    to {opacity: 1; transform: translateX(0);}
}

/* Chat bubble AI */
.chat-bubble-ai {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(0,180,216,0.3);
    border-radius: 18px 18px 18px 0;
    padding: 10px 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 80%;
    animation: fadeInLeft 0.4s ease;
}
@keyframes fadeInLeft {
    from {opacity: 0; transform: translateX(-30px);}
    to {opacity: 1; transform: translateX(0);}
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0, 180, 216, 0.05);
    border-right: 1px solid rgba(0,180,216,0.2);
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const bubbles = document.querySelectorAll('.chat-bubble-user, .chat-bubble-ai');
  bubbles.forEach(b => {
    b.addEventListener('mouseenter', () => b.style.transform = 'scale(1.02)');
    b.addEventListener('mouseleave', () => b.style.transform = 'scale(1)');
  });
});
</script>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🚘 Prediksi Harga Mobil + Chat Gemini AI</div>", unsafe_allow_html=True)

# =========================
# SIDEBAR STATUS FILE
# =========================
st.sidebar.header("📁 Status Berkas")
def file_status(path): return "✅" if os.path.exists(path) else "❌"
st.sidebar.write(f"mo.pkl.gz: {file_status('mo.pkl.gz')}")
st.sidebar.write(f"columns.json: {file_status('columns.json')}")
st.sidebar.write(f"toyota.csv: {file_status('toyota.csv')}")
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Gemini")
enable_gemini = st.sidebar.toggle("Aktifkan Gemini Chat", value=True)
selected_llm = st.sidebar.selectbox("Model LLM", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.5, 0.05)

st.session_state["enable_gemini"] = enable_gemini
st.session_state["selected_llm"] = selected_llm
st.session_state["temperature"] = temperature

# =========================
# FUNGSI PENDUKUNG
# =========================
@st.cache_resource
def load_model_if_exists(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def prepare_input(form_data: Dict[str, Any], example_schema: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{k: form_data.get(k, v) for k, v in example_schema.items()}])

def gemini_estimate_price(prompt: str):
    try:
        if not st.session_state.get("enable_gemini", True):
            return "🤖 Gemini dinonaktifkan."
        model_gemini = genai.GenerativeModel(st.session_state.get("selected_llm", "gemini-2.0-flash"))
        response = model_gemini.generate_content(
            f"""
Kamu adalah asisten harga mobil di Indonesia.
Berikan estimasi harga mobil dalam Rupiah berdasarkan deskripsi pengguna.

Pertanyaan pengguna:
{prompt}

Format:
💬 Penjelasan singkat
💰 Perkiraan harga: Rp [kisaran]
""",
            generation_config={"temperature": st.session_state.get("temperature", 0.5)},
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gagal memanggil Gemini API: {e}"

def local_chat_response(user_message: str, df=None):
    msg = user_message.lower().strip()
    if msg in ["model", "daftar model", "model mobil"]:
        if df is not None and not df.empty and "model" in df.columns:
            models = sorted(df["model"].dropna().unique().tolist())
            return "🚗 Model tersedia:\n" + ", ".join(models[:50])
        return "Dataset belum dimuat."
    if "harga" in msg:
        if df is not None and not df.empty and "model" in df.columns and "price" in df.columns:
            models = df["model"].dropna().unique().tolist()
            best_match = difflib.get_close_matches(msg, [m.lower() for m in models], n=1, cutoff=0.6)
            if best_match:
                found = best_match[0]
                matched = next((m for m in models if m.lower() == found), found)
                avg_price = df[df["model"].str.lower() == matched.lower()]["price"].mean()
                if pd.notna(avg_price):
                    harga_rp = avg_price * 20000
                    return f"💰 Rata-rata {matched}: **Rp {harga_rp:,.0f}**"
        return gemini_estimate_price(user_message)
    if any(x in msg for x in ["halo", "hi", "hai"]):
        return "Halo 👋! Ketik 'model' untuk lihat daftar mobil atau tanya harga."
    return gemini_estimate_price(user_message)

# =========================
# MUAT MODEL & DATA
# =========================
MODEL_PATH = "mo.pkl.gz"
DATASET_PATH = "toyota.csv"
EXAMPLE_PATH = "example_schema.json"

model = load_model_if_exists(MODEL_PATH)
df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else pd.DataFrame()
try:
    with open(EXAMPLE_PATH, "r") as f:
        example_schema = json.load(f)
except:
    example_schema = {"model": "Avanza", "year": 2020, "transmission": "Manual",
                      "mileage": 15000, "fuelType": "Bensin", "tax": 1500000,
                      "mpg": 14.5, "engineSize": 1.3}

# =========================
# ANTARMUKA (2 TAB TERPISAH)
# =========================
tab1, tab2 = st.tabs(["🧮 Prediksi Manual", "💬 Chat Prediksi"])

with tab1:
    st.subheader("🚘 Form Prediksi Harga Mobil")
    with st.form("form_prediksi"):
        inputs = {}
        models = df["model"].dropna().unique().tolist() if not df.empty else ["Avanza", "Yaris", "Rush"]
        inputs["model"] = st.selectbox("Model Mobil", models)
        inputs["year"] = st.number_input("Tahun Produksi", 1990, 2025, 2020)
        inputs["transmission"] = st.selectbox("Transmisi", ["Manual", "Automatic"])
        inputs["mileage"] = st.number_input("Jarak Tempuh (km)", 0, 500000, 15000)
        inputs["fuelType"] = st.selectbox("Jenis Bahan Bakar", ["Bensin", "Diesel", "Hybrid"])
        inputs["tax"] = st.number_input("Pajak Tahunan (Rp)", 0, 10000000, 1500000)
        inputs["mpg"] = st.number_input("Efisiensi BBM (km/l)", 0.0, 100.0, 14.5)
        inputs["engineSize"] = st.number_input("Kapasitas Mesin (L)", 0.0, 5.0, 1.3)
        submit = st.form_submit_button("🚀 Prediksi")

    if submit and model is not None:
        X = prepare_input(inputs, example_schema)
        pred = model.predict(X)[0]
        harga_rp = pred * 20000
        st.success(f"💰 Estimasi harga {inputs['model']}: **Rp {harga_rp:,.0f}** (≈ £{pred:,.2f})")

with tab2:
    st.subheader("💬 Chat Asisten Gemini")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_box = st.container()
    with chat_box:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for role, msg in st.session_state.chat_history:
            bubble = "chat-bubble-user" if role == "user" else "chat-bubble-ai"
            st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    user_msg = st.text_input("Ketik pesan...")
    if st.button("Kirim 💬"):
        if user_msg.strip():
            st.session_state.chat_history.append(("user", user_msg))
            reply = local_chat_response(user_msg, df)
            st.session_state.chat_history.append(("ai", reply))
            st.rerun()

st.markdown("---")
st.caption("✨ Aplikasi Prediksi Harga Mobil + Chat Gemini AI — versi Neon Futuristik ✨")



