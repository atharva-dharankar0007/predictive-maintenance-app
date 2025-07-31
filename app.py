import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Predictive Maintenance App", layout="wide")

# Netflix-like Dark Theme CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #141414;
        color: #E5E5E5;
    }

    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: background-color 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #B20710;
    }

    .stDataFrame th {
        background-color: #333333 !important;
        color: #FFFFFF !important;
    }

    .stMarkdown h1, h2, h3, h4 {
        color: #E50914;
        font-weight: bold;
    }

    .css-1aumxhk {
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }

    .custom-subheader {
        color: #E50914;
        font-size: 22px;
        font-weight: bold;
    }

    .stSidebar {
        background-color: #000000;
    }

    a {
        color: #E50914;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    hr {
        border: 1px solid #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and encoders
model_binary = joblib.load("model_binary.pkl")
model_multi = joblib.load("model_multi.pkl")
scaler = joblib.load("scaler.pkl")
le_type = joblib.load("label_encoder_type.pkl")
le_target = joblib.load("label_encoder_target.pkl")

# Header
st.markdown("""
<div style='background-color:#000000; padding: 20px; border-radius: 10px;'>
  <h1 style='color: #E50914;'>🔧 Easy Machine Failure Predictor</h1>
  <p style='color: white;'>Welcome to the <b>Predictive Maintenance Dashboard</b> — powered by machine learning!</p>
</div>
<br/>
<div>
 🎯 This tool helps you:
 <ul>
 <li>Predict if a machine will fail based on sensor data</li>
 <li>Know what kind of failure is likely</li>
 <li>See visual insights about the data</li>
 </ul>
 <hr/>
</div>
""", unsafe_allow_html=True)

# Sidebar input
st.sidebar.markdown("<h3 class='custom-subheader'>🔢 Enter Sensor Readings</h3>", unsafe_allow_html=True)

def user_input_features():
    type_input = st.sidebar.selectbox("Machine Type", le_type.classes_)
    air_temp = st.sidebar.number_input("Air Temperature [K]", min_value=290.0, max_value=315.0, value=300.0)
    proc_temp = st.sidebar.number_input("Process Temperature [K]", min_value=290.0, max_value=320.0, value=305.0)
    rpm = st.sidebar.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500)
    torque = st.sidebar.number_input("Torque [Nm]", min_value=20.0, max_value=80.0, value=50.0)
    tool_wear = st.sidebar.number_input("Tool Wear [min]", min_value=0, max_value=300, value=100)

    data = {
        "Type": le_type.transform([type_input])[0],
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.markdown("<h3 class='custom-subheader'>📋 Input Summary</h3>", unsafe_allow_html=True)
st.dataframe(input_df.style.format("{:.2f}"))

# Prediction
if st.button("🚀 Predict Machine Status"):
    input_scaled = scaler.transform(input_df)
    will_fail = model_binary.predict(input_scaled)[0]

    if will_fail == 0:
        st.success("✅ The machine is working fine. No failure expected.")
    else:
        predicted_type = model_multi.predict(input_scaled)
        failure_label = le_target.inverse_transform(predicted_type)[0]
        st.error(f"❌ Warning: Failure expected! Type: **{failure_label}**")

# Dashboard Visuals
st.markdown("<h3 class='custom-subheader'>📊 Sensor Insights from Real Data</h3>", unsafe_allow_html=True)
if st.checkbox("Show Sensor Data Charts"):
    try:
        sample_data = pd.read_csv("predictive_maintenance.csv")

        st.markdown("<h4 class='custom-subheader'>📌 Failure Type Distribution </h4>", unsafe_allow_html=True)
        failure_counts = sample_data["Failure Type"].value_counts().reset_index()
        failure_counts.columns = ["Failure Type", "Count"]
        fig1 = px.bar(failure_counts, x="Failure Type", y="Count", color="Failure Type", text="Count", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("<h4 class='custom-subheader'>📌 Temperature vs Torque by Failure</h4>", unsafe_allow_html=True)
        fig2 = px.scatter(sample_data, x="Air temperature [K]", y="Torque [Nm]",
                          color="Failure Type", hover_data=["Rotational speed [rpm]"], opacity=0.6, template="plotly_dark")
        fig2.add_scatter(x=input_df["Air temperature [K]"],
                         y=input_df["Torque [Nm]"],
                         mode='markers',
                         marker=dict(color='red', size=12),
                         name='Your Input')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<h4 class='custom-subheader'>📌 Input vs Average RPM</h4>", unsafe_allow_html=True)
        avg_rpm = sample_data["Rotational speed [rpm]"].mean()
        st.write(f"🔧 Your RPM: **{input_df['Rotational speed [rpm]'].values[0]}**")
        st.write(f"📊 Average RPM: **{avg_rpm:.2f}**")

    except FileNotFoundError:
        st.warning("Sample CSV not found. Charts are not available.")

# Footer
st.markdown("""
<br/>
<hr/>
<div style='text-align: center;'>
  <p>👨‍💻 <strong>Developed by Atharva Dharankar</strong></p>
  <p>📧 <a href='mailto:atharvadharankar0007@gmail.com'>atharvadharankar0007@gmail.com</a> | 🌍 Akola, Maharashtra, India</p>
  <p>💻 <a href='https://github.com/atharva-dharankar0007' target='_blank'>GitHub</a> | 🔗 <a href='https://www.linkedin.com/in/atharva-dharankar/' target='_blank'>LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)


