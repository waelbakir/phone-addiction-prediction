"""
app.py — Smartphone Addiction Prediction
Streamlit deployment app
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Phone Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
MODEL_PATH   = BASE / "models" / "best_model.pkl"
SCALER_PATH  = BASE / "outputs" / "scaler.pkl"
FEATURE_PATH = BASE / "outputs" / "feature_names.pkl"


@st.cache_resource
def load_artifacts():
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, scaler, features


try:
    model, scaler, feature_names = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    missing_file = str(e)

# ── Sidebar: About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/smartphone.png", width=80)
    st.title("About")
    st.markdown(
        """
        This app predicts **smartphone addiction risk** based on a user's
        daily phone usage habits.

        **Pipeline:**
        - 📊 EDA → `01_EDA.ipynb`
        - ⚙️ Preprocessing → `02_Preprocessing_and_Balancing.ipynb`
        - 🤖 Modeling → `03_Modeling.ipynb`

        **Model:** Best-performing classifier from GridSearchCV  
        **Balancing:** SMOTE applied during training
        """
    )
    st.divider()
    st.caption("Data Mining Project — 2025")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📱 Smartphone Addiction Prediction")
st.markdown(
    "Fill in your smartphone usage habits below and click **Predict** to "
    "see if you're at risk of phone addiction."
)

if not artifacts_ok:
    st.error(
        f"⚠️ Could not load model artifacts. "
        f"Please run the notebooks first to generate them.\n\nMissing: `{missing_file}`"
    )
    st.stop()

st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    age = st.slider("Age", 10, 70, 22)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    st.subheader("📱 Usage Patterns")
    daily_screen_time = st.slider("Daily Screen Time (hours)", 0.0, 20.0, 5.0, step=0.5)
    social_media_hours = st.slider("Social Media Hours / day", 0.0, 12.0, 2.0, step=0.5)
    num_apps_used = st.number_input("Number of Apps Used Daily", 1, 50, 10)
    notifications_per_day = st.number_input("Notifications per Day", 0, 500, 50)

with col3:
    st.subheader("🧠 Health & Wellbeing")
    sleep_hours = st.slider("Sleep Hours / night", 2.0, 12.0, 7.0, step=0.5)
    stress_level = st.select_slider("Stress Level", ["Low", "Medium", "High"])
    academic_work_impact = st.radio(
        "Does phone use impact your academic / work performance?", ["No", "Yes"]
    )
    physical_activity_hours = st.slider("Physical Activity (hours / week)", 0.0, 20.0, 3.0, step=0.5)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Addiction Risk", type="primary", use_container_width=True):

    # Build raw input dict matching training encoding
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    impact_map = {"No": 0, "Yes": 1}

    raw_input = {
        "age": age,
        "gender": gender_map[gender],
        "daily_screen_time_hours": daily_screen_time,
        "social_media_hours": social_media_hours,
        "num_apps_used": num_apps_used,
        "notifications_per_day": notifications_per_day,
        "sleep_hours": sleep_hours,
        "stress_level": stress_map[stress_level],
        "academic_work_impact": impact_map[academic_work_impact],
        "physical_activity_hours": physical_activity_hours,
    }

    # Align to training feature columns
    input_df = pd.DataFrame([raw_input])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    # ── Results ──────────────────────────────────────────────────────────────
    st.subheader("🎯 Prediction Result")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if pred == 1:
            st.error("⚠️ **Addicted**\nYou may be at risk of phone addiction.")
        else:
            st.success("✅ **Not Addicted**\nYour usage appears healthy.")

    with res_col2:
        st.metric("Addiction Probability", f"{prob[1]*100:.1f}%")

    with res_col3:
        st.metric("Not Addicted Probability", f"{prob[0]*100:.1f}%")

    # Progress bar
    st.progress(float(prob[1]), text=f"Risk Level: {prob[1]*100:.1f}%")

    # Tips
    st.divider()
    st.subheader("💡 Recommendations")
    tips_col1, tips_col2 = st.columns(2)

    with tips_col1:
        st.info(
            "**Healthy Habits:**\n"
            "- Limit screen time to < 4 hours/day\n"
            "- Use Do Not Disturb mode at night\n"
            "- Take regular breaks (20-20-20 rule)\n"
            "- Keep phones out of the bedroom"
        )
    with tips_col2:
        if pred == 1:
            st.warning(
                "**Consider:**\n"
                "- Setting app time limits\n"
                "- Scheduling phone-free hours\n"
                "- Increasing physical activity\n"
                "- Consulting a digital wellness professional"
            )
        else:
            st.success(
                "**Keep it up!**\n"
                "- Maintain your current screen habits\n"
                "- Stay physically active\n"
                "- Continue prioritizing sleep\n"
                "- Review your habits periodically"
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only and does not constitute "
    "medical advice. Consult a professional for clinical assessment."
)
