import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("titanic_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# --- STYLING ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f4f8;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .css-18e3th9 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            color: white;
            background-color: #228be6;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🚢 Titanic Survival Prediction")
st.caption("Built with Machine Learning + Streamlit")

st.markdown("Enter passenger details below to predict their survival chance:")

# --- FORM LAYOUT ---
with st.form(key="prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 0, 80, 30)
        fare = st.number_input("Ticket Fare ($)", 0.0, 600.0, 50.0)
        sex = st.selectbox("Sex", ["male", "female"])
        sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
    
    with col2:
        parch = st.number_input("Parents/Children aboard", 0, 10, 0)
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        embarked_1 = st.checkbox("Embarked from Queenstown (Embarked_1.0)")
        embarked_2 = st.checkbox("Embarked from Southampton (Embarked_2.0)")
    
    submit = st.form_submit_button(label="Predict")

# --- PROCESS ---
if submit:
    sex_bin = 1 if sex == "male" else 0
    
    input_df = pd.DataFrame([{
        'Age': age,
        'Fare': fare,
        'Sex': sex_bin,
        'SibSp': sibsp,
        'Parch': parch,
        'Pclass': pclass,
        'Embarked_1.0': int(embarked_1),
        'Embarked_2.0': int(embarked_2)
    }])
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ This passenger **survived**! (Probability: {probability:.2%})")
    else:
        st.error(f"❌ This passenger **did not survive**. (Probability: {probability:.2%})")

