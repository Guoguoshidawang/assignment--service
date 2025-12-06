{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import torch\
import joblib\
import numpy as np\
from flask import Flask, request, jsonify\
\
app = Flask(__name__)\
\
# Load model and scaler\
model = joblib.load("model.pkl")\
scaler = joblib.load("scaler.pkl")\
\
@app.route("/", methods=["GET"])\
def home():\
    return "Service is running!"\
\
@app.route("/predict", methods=["POST"])\
def predict():\
    data = request.get_json()\
\
    try:\
        distance = float(data["distance"])\
        prep_time = float(data["prep_time"])\
        experience = float(data["experience"])\
    except:\
        return \{"error": "Input fields invalid or missing"\}, 400\
\
    X = np.array([[distance, prep_time, experience]], dtype=np.float32)\
\
    X_scaled = scaler.transform(X)\
\
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)\
\
    with torch.no_grad():\
        pred = model(X_tensor).item()\
\
    return \{"prediction_minutes": float(pred)\}\
\
if __name__ == "__main__":\
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))\
}