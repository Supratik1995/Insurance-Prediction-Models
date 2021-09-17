import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

Regressor = data["model"]
le_policytype = data["le_policytype"]
le_gender = data["le_gender"]
le_premiumpayment = data["le_premiumpayment"]
le_occupation = data["le_occupation"]
le_smoker = data["le_smoker"]

def show_predict_page():
    st.title("Life Insurance Premium Prediction")

    st.sidebar.write("""### We would like to know more about you""")

    Policy_Type = (
        "Term",
        "Endowment",
    )

    Policy_Term = (
        "5",
        "10",
        "15",
        "20",
        "25",
        "30",
        "35",
        "40",
    )

    Gender = (
        "Male",
        "Female",
    )

    Age = ("20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
           "36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51",
           "52","53","54","55","56","57","58","59","60",
           )

    Sum_Assured = ("20 Lakhs","25 Lakhs","30 Lakhs","35 Lakhs","40 Lakhs","45 Lakhs","50 Lakhs",
                   "55 Lakhs", "60 Lakhs", "65 Lakhs", "70 Lakhs", "75 Lakhs", "80 Lakhs", "85 Lakhs",
                   "90 Lakhs", "95 Lakhs", "1 Crore",
    )

    Premium_Payment = (
        "Yearly",
        "Monthly",
    )

    Occupation = (
        "Salaried",
        "Self-Employed",
        "Retired",
        "Student",
        "Unemployed",
        "Agriculturist",
    )

    Smoker = (
        "Yes",
        "No",
    )

    Policy_Type = st.selectbox("Policy Type", Policy_Type)
    Policy_Term = st.slider("Policy Term", 5, 40, 5, 5)
    Gender = st.selectbox("Gender", Gender)
    Age = st.selectbox("Age", Age)
    Sum_Assured = st.slider("Sum Assured", 2000000, 10000000, 2000000, 500000)
    Premium_Payment = st.selectbox("Premium Payment", Premium_Payment)
    Occupation = st.selectbox("Occupation", Occupation)
    Smoker = st.selectbox("Smoker", Smoker)

    ok = st.button("Calculate Premium")
    if ok:
        X = np.array([[Policy_Type, Policy_Term, Gender,Age , Premium_Payment, Sum_Assured, Occupation, Smoker]])
        X[:, 0] = le_policytype.transform(X[:, 0])
        X[:, 2] = le_gender.transform(X[:, 2])
        X[:, 4] = le_premiumpayment.transform(X[:, 4])
        X[:, 6] = le_occupation.transform(X[:, 6])
        X[:, 7] = le_smoker.transform(X[:, 7])
        X = X.astype(float)

        Premium = Regressor.predict(X)
        st.subheader(f"Now secure your life with a premium of only Rs {Premium[0]:.2f}")

