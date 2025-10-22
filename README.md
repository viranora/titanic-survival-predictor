# 🚢 Titanic Survival Predictor App

This is an interactive machine learning web application built with **Streamlit** that predicts whether a passenger would survive the Titanic disaster, based on key features such as age, fare, class, and embarkation point.

## ✨ Features

- Logistic Regression + GridSearchCV for best model selection
- Scikit-learn-powered predictions
- Streamlit UI with live input fields
- Deployment-ready architecture

## 📦 Project Structure

titanic/

├── titanic_app.py              # Streamlit application (main Python script)

├── titanic_best_model.pkl      # Trained machine learning model (Logistic Regression)

├── scaler.pkl                  # StandardScaler object to scale user input

├── requirements.txt            # List of Python packages needed to run the app

├── README.md                   # Project description and usage instructions

└── data/

    └── titanic.csv             # Original dataset (optional for deployment)


## 🚀 Live Demo

👉 https://titanic-survival-predictor-app.streamlit.app

## 🔧 Run Locally

Make sure you have Python installed, then:

```bash
# Clone the repo
git clone https://github.com/<your-username>/titanic-survival-predictor.git
cd titanic-survival-predictor

# Install required packages
pip install -r requirements.txt

# Start the Streamlit app
streamlit run titanic_app.py
```

## 📊 Model Overview

Features used: Age, Fare, Sex, SibSp, Parch, Pclass, Embarked_1.0, Embarked_2.0
Best Estimator: Logistic Regression with tuned hyperparameters
Evaluation: Accuracy, ROC curve, classification report, cross-validation

## 👩‍💻 Author
Built by Sema Nur Özyılmaz If you're passionate about ML, NLP, or tech communities — let's connect!

## 📚 More About This Project
For a step-by-step explanation of data preprocessing, model tuning, Streamlit interface design, and deployment, check out the full Medium article:

https://medium.com/@ssozylmz/from-csv-to-web-app-building-and-deploying-my-titanic-ml-model-with-streamlit-619e44c4f184

...by Vira

