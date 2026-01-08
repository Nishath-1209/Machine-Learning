import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="centered"
)

# =====================================
# App Title
# =====================================
st.title("📊 Telco Customer Churn Prediction")
st.subheader("Logistic Regression Model")
st.write(
    "This app predicts whether a telecom customer is **likely to churn** "
    "based on customer usage and service information."
)

# =====================================
# Load Dataset
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

# =====================================
# Data Understanding
# =====================================
st.write("### Dataset Shape:", df.shape)
st.write("### Churn Distribution")
st.bar_chart(df["Churn"].value_counts())

# =====================================
# Data Preprocessing
# =====================================
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop("customerID", axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# Train Model
# =====================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =====================================
# Model Evaluation
# =====================================
y_pred = model.predict(X_test)

st.write("## 📈 Model Performance")

st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 3))

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, cmap="Blues", values_format="d", ax=ax
)
st.pyplot(fig)

# =====================================
# Predict Unseen Customer
# =====================================
st.write("## 🔍 Predict Churn for an Unseen Customer")

customer_index = st.number_input(
    "Select customer index from dataset",
    min_value=0,
    max_value=len(X) - 1,
    value=0
)

new_customer = X.iloc[[customer_index]]
new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)
probability = model.predict_proba(new_customer_scaled)

if prediction[0] == 1:
    st.error("⚠️ Customer is likely to churn")
else:
    st.success("✅ Customer is likely to stay")

st.write("**Churn Probability:**", round(probability[0][1], 3))

# =====================================
# Business Interpretation
# =====================================
st.write("## 💼 Business Insight")
st.write(
    "Customers predicted with a high churn probability should be "
    "targeted with proactive retention offers such as discounts or "
    "personalized plans."
)
