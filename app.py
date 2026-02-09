from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# --- LOAD MODEL AND COLUMNS ---
try:
    loaded_model = joblib.load('AQI_Forecasting.pkl')
    feature_names = joblib.load('model_columns.pkl')
    print("🚀 Model and columns loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")

def get_aqi_category(prediction):
    if prediction <= 50: return "Good", "#00e400"
    elif prediction <= 100: return "Satisfactory", "#ffff00"
    elif prediction <= 200: return "Moderate", "#ff7e00"
    elif prediction <= 300: return "Poor", "#ff0000"
    elif prediction <= 400: return "Very Poor", "#8f3f97"
    else: return "Severe", "#7e0023"

# --- ADDED THIS SECTION ---
@app.route('/', methods=['GET'])
def home():
    return "<h1>🌍 AQI Prediction API is Running!</h1><p>Use the frontend (index.html) to make predictions.</p>"
# -------------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_city = data.get('city')
        month = int(data.get('month'))
        day = int(data.get('day'))
        year = int(data.get('year'))

        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        input_df['Month'] = month
        input_df['Date_'] = day
        input_df['Year'] = year

        city_col = f"City_{user_city}"
        if city_col in input_df.columns:
            input_df[city_col] = 1
        else:
            return jsonify({"error": f"City '{user_city}' not supported"}), 400

        prediction = float(loaded_model.predict(input_df)[0])
        category, color = get_aqi_category(prediction)

        return jsonify({
            "aqi": round(prediction, 2),
            "category": category,
            "color": color,
            "city": user_city
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)