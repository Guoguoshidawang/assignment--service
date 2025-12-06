import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from flask import Flask, request, jsonify

# -----------------------------
# Add model class
# -----------------------------
class DeliveryTimeRegressor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(DeliveryTimeRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Service is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        distance = float(data["distance"])
        prep_time = float(data["prep_time"])
        experience = float(data["experience"])
    except:
        return {"error": "Invalid or missing fields"}, 400

    X = np.array([[distance, prep_time, experience]], dtype=np.float32)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_tensor).item()

    return {"prediction_minutes": float(pred)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
