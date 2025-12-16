import streamlit as st
import pandas as pd
import joblib
import numpy as np

ARTIFACTS_PATH = "artifacts/"


scaler = joblib.load(ARTIFACTS_PATH + "scaler.pkl")
kmeans = joblib.load(ARTIFACTS_PATH + "kmeans_model.pkl")

VARIABLES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AGE",
    "YEARS_EMPLOYED"
]


st.set_page_config(page_title="Home Credit Clustering", layout="centered")

st.title("Segmentación de Clientes – Home Credit")
st.write(
    """
    Esta aplicación utiliza un modelo **K-Means** entrenado sobre datos de Home Credit
    para asignar a un solicitante de crédito a un **cluster de comportamiento**.
    """
)

st.sidebar.header("🔢 Ingreso de datos del cliente")

income = st.sidebar.number_input(
    "Ingreso total",
    min_value=0.0,
    value=150000.0,
    step=1000.0
)

credit = st.sidebar.number_input(
    "Monto del crédito",
    min_value=0.0,
    value=500000.0,
    step=5000.0
)

annuity = st.sidebar.number_input(
    "Monto de anualidad",
    min_value=0.0,
    value=25000.0,
    step=500.0
)

age = st.sidebar.number_input(
    "Edad",
    min_value=18,
    max_value=100,
    value=35
)

years_employed = st.sidebar.number_input(
    "Años trabajados",
    min_value=0.0,
    max_value=60.0,
    value=8.0,
    step=0.5
)


if st.sidebar.button("Calcular Cluster"):


    df_input = pd.DataFrame(
        [[income, credit, annuity, age, years_employed]],
        columns=VARIABLES
    )


    X_scaled = scaler.transform(df_input)


    cluster = int(kmeans.predict(X_scaled)[0])

    st.subheader("Resultado")
    st.success(f"El cliente pertenece al **Cluster {cluster}**")


    explanations = {
        0: "Clientes con ingresos y montos de crédito bajos.",
        1: "Clientes con ingresos medios y estabilidad laboral.",
        2: "Clientes con montos de crédito elevados.",
        3: "Clientes con ingresos altos y mayor capacidad financiera."
    }

    st.info(explanations.get(cluster, "Cluster no definido"))

    st.subheader("Datos ingresados")
    st.dataframe(df_input)