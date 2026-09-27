import streamlit as st
import pandas as pd
import joblib

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Stress Level Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- Base ---------- */
    .stApp {
        background-color: #f4f6fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* ---------- Header ---------- */
    .title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 4px;
        color: #1f2937;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 8px;
    }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #374151;
        margin-top: 28px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e5e7eb;
    }

    /* ---------- Cards ---------- */
    .info-card {
        padding: 20px 22px;
        border-radius: 14px;
        background-color: #ffffff;
        border: 1px solid #eef0f4;
        box-shadow: 0px 2px 8px rgba(17, 24, 39, 0.04);
        margin-bottom: 16px;
    }

    .result-card {
        padding: 32px;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #eef0f4;
        box-shadow: 0px 4px 16px rgba(17, 24, 39, 0.06);
        text-align: center;
        margin-top: 12px;
    }

    .result-title {
        font-size: 15px;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    .result-value {
        font-size: 40px;
        font-weight: 700;
        margin: 6px 0 8px 0;
    }

    .result-desc {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 4px;
    }

    .badge-low   { color: #059669; }
    .badge-mid   { color: #d97706; }
    .badge-high  { color: #dc2626; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eef0f4;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #4b5563;
        font-size: 14px;
        line-height: 1.5;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 14px;
        color: #374151;
    }

    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        background-color: #4f46e5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 15px;
        transition: background-color 0.15s ease;
    }

    div.stButton > button:hover {
        background-color: #4338ca;
        color: white;
        border: none;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #eef0f4;
        border-radius: 12px;
        padding: 14px 10px;
        box-shadow: 0px 2px 6px rgba(17, 24, 39, 0.03);
    }

    /* ---------- Divider ---------- */
    hr {
        margin: 1.2rem 0;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("kelompok5/stress_level.pkl")


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="title">🧠 Stress Level Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Prediksi tingkat stres mahasiswa menggunakan Random Forest Classification</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("### 📌 Tentang Aplikasi")
    st.write(
        "Aplikasi ini menggunakan model **Random Forest Classification** "
        "untuk memprediksi tingkat stres berdasarkan faktor psikologis, "
        "akademik, sosial, dan lingkungan."
    )

    st.markdown("---")
    st.markdown("### 📊 Kategori Stress Level")

    st.markdown(
        """
        <div class="legend-item"><span class="legend-dot" style="background-color:#059669;"></span> 0 — Rendah</div>
        <div class="legend-item"><span class="legend-dot" style="background-color:#d97706;"></span> 1 — Sedang</div>
        <div class="legend-item"><span class="legend-dot" style="background-color:#dc2626;"></span> 2 — Tinggi</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("Machine Learning Project")
    st.caption("Random Forest Classification")


# =========================================================
# INPUT DATA
# =========================================================

st.markdown('<div class="section-header">📝 Masukkan Data Mahasiswa</div>', unsafe_allow_html=True)
st.caption("Silakan masukkan nilai sesuai dengan kondisi mahasiswa.")

# ---------------------------------------------------------
# Psychological factors
# ---------------------------------------------------------

st.markdown('<div class="section-header">🧠 Faktor Psikologis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    anxiety_level = st.slider("Anxiety Level", 0, 21, 10)
with col2:
    self_esteem = st.slider("Self Esteem", 0, 30, 15)
with col3:
    depression = st.slider("Depression", 0, 27, 10)

col1, col2, col3 = st.columns(3)
with col1:
    mental_health_history = st.selectbox(
        "Mental Health History",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )
with col2:
    headache = st.slider("Headache", 0, 5, 2)
with col3:
    breathing_problem = st.slider("Breathing Problem", 0, 5, 2)

# ---------------------------------------------------------
# Physical & environment
# ---------------------------------------------------------

st.markdown('<div class="section-header">🏠 Kondisi Fisik & Lingkungan</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    blood_pressure = st.slider("Blood Pressure", 1, 3, 2)
with col2:
    sleep_quality = st.slider("Sleep Quality", 0, 5, 3)
with col3:
    noise_level = st.slider("Noise Level", 0, 5, 2)

col1, col2, col3 = st.columns(3)
with col1:
    living_conditions = st.slider("Living Conditions", 0, 5, 3)
with col2:
    safety = st.slider("Safety", 0, 5, 3)
with col3:
    basic_needs = st.slider("Basic Needs", 0, 5, 3)

# ---------------------------------------------------------
# Academic factors
# ---------------------------------------------------------

st.markdown('<div class="section-header">🎓 Faktor Akademik</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    academic_performance = st.slider("Academic Performance", 0, 5, 3)
with col2:
    study_load = st.slider("Study Load", 0, 5, 3)
with col3:
    teacher_student_relationship = st.slider("Teacher Student Relationship", 0, 5, 3)

col1, col2 = st.columns(2)
with col1:
    future_career_concerns = st.slider("Future Career Concerns", 0, 5, 3)
with col2:
    extracurricular_activities = st.slider("Extracurricular Activities", 0, 5, 3)

# ---------------------------------------------------------
# Social factors
# ---------------------------------------------------------

st.markdown('<div class="section-header">👥 Faktor Sosial</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    social_support = st.slider("Social Support", 0, 3, 2)
with col2:
    peer_pressure = st.slider("Peer Pressure", 0, 5, 2)
with col3:
    bullying = st.slider("Bullying", 0, 5, 1)


# =========================================================
# PREDICTION
# =========================================================

st.write("")
predict_button = st.button("🔮 Prediksi Stress Level", use_container_width=True)

if predict_button:

    input_data = pd.DataFrame([{
        "anxiety_level": anxiety_level,
        "self_esteem": self_esteem,
        "mental_health_history": mental_health_history,
        "depression": depression,
        "headache": headache,
        "blood_pressure": blood_pressure,
        "sleep_quality": sleep_quality,
        "breathing_problem": breathing_problem,
        "noise_level": noise_level,
        "living_conditions": living_conditions,
        "safety": safety,
        "basic_needs": basic_needs,
        "academic_performance": academic_performance,
        "study_load": study_load,
        "teacher_student_relationship": teacher_student_relationship,
        "future_career_concerns": future_career_concerns,
        "social_support": social_support,
        "peer_pressure": peer_pressure,
        "extracurricular_activities": extracurricular_activities,
        "bullying": bullying
    }])

    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100
    else:
        confidence = None

    if prediction == 0:
        stress_label = "Rendah"
        badge_class = "badge-low"
        description = "Tingkat stres berada pada kategori rendah. Kondisi mahasiswa relatif stabil."
    elif prediction == 1:
        stress_label = "Sedang"
        badge_class = "badge-mid"
        description = "Tingkat stres berada pada kategori sedang. Perlu perhatian lebih lanjut."
    else:
        stress_label = "Tinggi"
        badge_class = "badge-high"
        description = "Tingkat stres berada pada kategori tinggi. Disarankan mencari dukungan tambahan."

    # -------------------------------------------------
    # Hasil
    # -------------------------------------------------

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Hasil Prediksi</div>
            <div class="result-value {badge_class}">{stress_label}</div>
            <div class="result-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if confidence is not None:
        st.write("")
        st.progress(int(confidence), text=f"Confidence Model: {confidence:.2f}%")

    # -------------------------------------------------
    # Detail
    # -------------------------------------------------

    st.write("")
    st.markdown('<div class="section-header">📊 Detail Hasil</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Class Prediksi", str(prediction))
    with col2:
        st.metric("Kategori", stress_label)
    with col3:
        st.metric("Confidence", f"{confidence:.2f}%" if confidence is not None else "N/A")

    with st.expander("🔍 Lihat Data Input"):
        st.dataframe(input_data, use_container_width=True)
