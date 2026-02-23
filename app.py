import streamlit as st
import joblib
import numpy as np
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Fraud Intelligence",
    page_icon="💳",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#00f5d4,#00bbf9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
}

.stButton>button {
    background: linear-gradient(90deg,#00f5d4,#00bbf9);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= HEADER =================
st.markdown('<div class="title">AI Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ================= HERO SECTION =================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🔐 Smart Transaction Monitoring")
    st.write("""
    This AI-powered system analyzes credit card transactions 
    in real time and detects fraudulent activities using 
    machine learning algorithms.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.image(
        "https://images.unsplash.com/photo-1588776814546-ec7e87c02a2f",
        use_column_width=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ================= INPUT SECTION =================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("💳 Enter Transaction Details")

colA, colB = st.columns(2)

with colA:
    time_input = st.number_input("Transaction Time", value=0.0)
    amount_input = st.number_input("Transaction Amount", value=0.0)

with colB:
    st.image(
        "https://images.unsplash.com/photo-1605902711622-cfb43c4437d1",
        use_column_width=True
    )

st.markdown("### PCA Feature Inputs")

pca_values = []
cols = st.columns(4)

for i in range(28):
    with cols[i % 4]:
        val = st.number_input(f"V{i+1}", value=0.0)
        pca_values.append(val)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= PREDICTION =================
if st.button("🚀 Analyze Transaction"):

    input_data = np.array([[time_input] + pca_values + [amount_input]])
    input_scaled = scaler.transform(input_data)

    with st.spinner("Analyzing transaction with AI engine..."):
        time.sleep(1.5)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Fraud Risk Analysis")
    st.progress(float(probability))

    if prediction == 1:
        st.markdown(
            '<div class="result-box" style="background-color:#ff4b4b;">🚨 Fraudulent Transaction Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box" style="background-color:#00c853;">✅ Legitimate Transaction</div>',
            unsafe_allow_html=True
        )