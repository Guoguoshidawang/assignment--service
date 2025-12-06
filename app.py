import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from flask import Flask, request, jsonify

# -------------------------
# Define the model class
# Must match the class used when saving model.pkl
# -------------------------
class DeliveryTimeRegressor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Load model and scaler
# -------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Put model into eval mode
model.eval()

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Service is running!"

@app.route("/predict", methods=["POST"])
def predict():

    # try reading JSON
    data = request.get_json(silent=True)

    # if no JSON, try form data
    if not data:
        data = request.form.to_dict()

    if not data:
        return jsonify({"error": "No input received"}), 400

    try:
        distance = float(data.get("distance"))
        prep_time = float(data.get("prep_time"))
        experience = float(data.get("experience"))
    except:
        return jsonify({"error": "Invalid or missing input fields"}), 400

    # Prepare input for model
    X = np.array([[distance, prep_time, experience]], dtype=np.float32)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Model prediction
    with torch.no_grad():
        prediction = model(X_tensor).item()

    return jsonify({"prediction_minutes": float(prediction)})


# -------------------------
# Run App for Render
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
