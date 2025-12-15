from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Credit Risk API")

model = joblib.load("../artifacts/model.pkl")
scaler = joblib.load("../artifacts/scaler.pkl")

@app.post("/evaluate_risk")
def evaluate_risk(data: dict):
    X = np.array(list(data.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]

    decision = (
        "APROBAR" if prob < 0.3 else
        "REVISION_MANUAL" if prob < 0.6 else
        "RECHAZAR"
    )

    return {
        "default_probability": round(prob, 4),
        "decision": decision
    }
