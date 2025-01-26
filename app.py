import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
# from pycaret.classification import load_model, predict_model, setup
import joblib
import google.generativeai as genai
import os
from dotenv import load_dotenv
# from pycaret import setup

load_dotenv()

API_KEY = os.getenv("API_KEY")
# Configure GenAI API
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load saved model and encoders
model_path = 'final_model'
encoders_path = 'label_encoders.pkl'
# model = load_model(model_path)
model_log = joblib.load('log_reg_model_sklearn.pkl')
label_encoders = joblib.load(encoders_path)

# Configure Gemini API
# genai.configure(api_key="AIzaSyAa6rMAFwAeG6UXkBBkkcfC5QM5kYzc3Q")
model = genai.GenerativeModel('gemini-1.5-flash')

def campaign_prediction_page():
    st.title("\U0001F697 EV Marketing Campaign Prediction")

    st.header("Enter User Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income (INR)", min_value=0, value=500000)
    vehicle_ownership = st.selectbox("Vehicle Ownership", ["Yes", "No"])
    preferred_features = st.selectbox("Preferred Features", label_encoders['Preferred Features'].classes_)
    financing_options = st.selectbox("Financing Options", label_encoders['Financing Options'].classes_)
    region = st.selectbox("Region", label_encoders['Region'].classes_)
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)

    if st.button("Predict Campaign Targeting"):
        inputs = {
            "Age": age,
            "Annual Income (INR)": annual_income,
            "Vehicle Ownership": 1 if vehicle_ownership == "Yes" else 0,
            "Preferred Features": label_encoders['Preferred Features'].transform([preferred_features])[0],
            "Financing Options": label_encoders['Financing Options'].transform([financing_options])[0],
            "Region": label_encoders['Region'].transform([region])[0],
            "Gender": label_encoders['Gender'].transform([gender])[0]
        }

        input_df = pd.DataFrame([inputs])
        # df = pd.read_csv("dataset_individual.csv")
        
        # exp = setup(data=df, target='Interest in EVs')
        
        prediction = model_log.predict(input_df)
        campaign_target = "High Interest" if prediction == 1 else "Low Interest"

        st.subheader("Prediction Result")
        st.write(f"Predicted Campaign Target: {campaign_target}")

        feedback_prompt = f"This is a Targeted Marketing Campaign task. Provide business recommendations based on the following data: {inputs} with the output being Predicted Campaign Target: {campaign_target}."
        feedback_response = gemini_model.generate_content(feedback_prompt)
        feedback = feedback_response.text

        st.subheader("AI-enabled Feedback & Recommendation")
        st.write(feedback)

def customer_clustering_page():
    st.title("\U0001F9ED Customer Clustering Analysis")

    st.header("Customer Clustering and Insights")

    df = pd.read_csv("dataset_individual.csv")
    

    categorical_columns = ['Gender', 'Region', 'Vehicle Ownership', 'Interest in EVs', 'Preferred Features', 'Financing Options']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    numerical_columns = ['Age', 'Annual Income (INR)']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    features = numerical_columns + categorical_columns

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features])

    for cluster in range(4):
        cluster_data = df[df['Cluster'] == cluster]
        # actual_means = cluster_data[['Age', 'Annual Income (INR)'] + categorical_columns].mean()

        # st.subheader(f"Cluster {cluster} Summary")
        # st.write("Mean values for each feature:")
        # st.table(actual_means)

        summary_prompt = f"Provide a business summary and recommendations for a customer segment with the following profile: Cluster {cluster} {cluster_data.to_dict()}"
        summary_response = gemini_model.generate_content(summary_prompt)
        summary = summary_response.text

        st.write(summary)

def ev_adoption_page():
    st.title("EV Adoption Rate Prediction with Gemini")

    st.header("Regional EV Adoption Analysis")

    df = pd.read_csv("dataset_regional.csv")

    st.write("Dataset Preview:", df.head())

    if "EV Adoption Rate (%)" in df.columns:
        features = df.drop(columns=["EV Adoption Rate (%)"])
    else:
        features = df

    # st.write("Features for Prediction:", features.columns.tolist())

    categorical_cols = ["Region", "Top Competitor", "Second Competitor"]
    categorical_cols = [col for col in categorical_cols if col in features.columns]
    numerical_cols = [col for col in features.columns if col not in categorical_cols]

    st.header("Provide Input for EV Adoption Rate Prediction")
    user_input = {}

    for col in categorical_cols:
        unique_values = features[col].dropna().unique()
        user_input[col] = st.selectbox(f"Select {col}:", options=unique_values)

    for col in numerical_cols:
        user_input[col] = st.text_input(f"Enter {col}:", value=str(features[col].mean()))

    input_data = {col: user_input[col] for col in features.columns}

    st.write("**Gemini Prediction**")
    if st.button("Predict with Gemini"):
        prompt = f"Based on the following regional data, predict the EV Adoption Rate (%):\n{input_data}"

        try:
            response = model.generate_content(prompt)
            feedback = response.text
            st.success(f"Predicted EV Adoption Rate: {feedback}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Campaign Prediction", "Customer Clustering", "EV Adoption Prediction"])

    if page == "Campaign Prediction":
        campaign_prediction_page()
    elif page == "Customer Clustering":
        customer_clustering_page()
    elif page == "EV Adoption Prediction":
        ev_adoption_page()
