import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load models and scaler
random_forest = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature names used during training
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Set Streamlit page layout
st.set_page_config(
    page_title="Heart Disease Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.markdown(
    "<h1 style='text-align: center; color: #ff6f61;'>Heart Disease Prediction App</h1>",
    unsafe_allow_html=True,
)
st.write("""
Welcome to the **Heart Disease Prediction App**!  
Use the interactive sliders and dropdowns on the left sidebar to input your health features.  
The app will predict your likelihood of heart disease and provide precautionary measures.
""")

# Sidebar for user input
st.sidebar.header("Input Features")
st.sidebar.markdown(
    "<h3 style='color: #4caf50;'>Customize Your Health Data</h3>",
    unsafe_allow_html=True,
)

age = st.sidebar.slider("Age", 20, 80, 50)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
cp = st.sidebar.slider("Chest Pain Type (cp)", 0, 4, 2)
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (chol)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=["Yes", "No"])
restecg = st.sidebar.slider("Resting ECG Results (restecg)", 0, 2, 1)
thalach = st.sidebar.slider("Max Heart Rate Achieved (thalach)", 60, 200, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=["Yes", "No"])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 5.0, 1.0)
slope = st.sidebar.slider("Slope of the Peak Exercise (slope)", 0, 2, 1)
ca = st.sidebar.slider("Number of Major Vessels (ca)", 0, 4, 1)
thal = st.sidebar.slider("Thalassemia (thal)", 0, 3, 2)

# Convert user inputs into a dataframe
input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "Male" else 0],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [1 if fbs == "Yes" else 0],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [1 if exang == "Yes" else 0],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal],
}, columns=feature_names)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict
if st.sidebar.button("💓 Predict"):
    prediction = random_forest.predict(input_data_scaled)
    probabilities = random_forest.predict_proba(input_data_scaled)

    # Display results
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.markdown(
        f"<h2 style='text-align: center; color: #4caf50;'>{result}</h2>",
        unsafe_allow_html=True,
    )

    # Probability visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probabilities[0][1] * 100,
        title={'text': "Probability of Heart Disease (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if prediction[0] == 1 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "pink"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Precautionary measures
    if prediction[0] == 1:
        st.markdown(
            "<h3 style='color: #ff6f61;'>Precautionary Measures</h3>"
            "<ul>"
            "<li>Schedule an appointment with a cardiologist immediately.</li>"
            "<li>Adopt a heart-healthy diet (low in saturated fats, high in fiber).</li>"
            "<li>Engage in moderate exercise under medical supervision.</li>"
            "<li>Manage stress through relaxation techniques or therapy.</li>"
            "<li>Monitor and control blood pressure and cholesterol levels regularly.</li>"
            "</ul>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h3 style='color: #4caf50;'>Health Tips</h3>"
            "<ul>"
            "<li>Maintain a balanced diet rich in vegetables, fruits, and lean proteins.</li>"
            "<li>Exercise regularly to keep your heart healthy.</li>"
            "<li>Avoid smoking and excessive alcohol consumption.</li>"
            "<li>Schedule routine health check-ups to stay informed.</li>"
            "</ul>",
            unsafe_allow_html=True,
        )

    # Visualize input data
    st.markdown("<h3 style='color: #4caf50;'>Input Data Summary</h3>", unsafe_allow_html=True)
    st.dataframe(input_data)

    # Show feature importance
    feature_importances = random_forest.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    fig_importance = px.bar(
        feature_importance_df,
        x="Importance",
        y="Feature",
        orientation='h',
        title="Feature Importance",
        color="Importance",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig_importance, use_container_width=True)
