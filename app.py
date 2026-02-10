from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# --- LOAD MODEL AND COLUMNS ---
try:
    # mmap_mode='r' is essential to save RAM on Render
    loaded_model = joblib.load('AQI_Forecasting.pkl', mmap_mode='r')
    feature_names = joblib.load('model_columns.pkl')
    # Create a mapping of column names to their index position
    col_map = {name: i for i, name in enumerate(feature_names)}
    print("🚀 Model loaded with Memory Mapping!")
except Exception as e:
    print(f"❌ Error: {e}")

def get_aqi_category(prediction):
    if prediction <= 50: return "Good", "#10b981"
    elif prediction <= 100: return "Satisfactory", "#9cd84e"
    elif prediction <= 200: return "Moderate", "#f59e0b"
    elif prediction <= 300: return "Poor", "#ff8c42"
    elif prediction <= 400: return "Very Poor", "#ef4444"
    else: return "Severe", "#7f1d1d"

@app.route('/', methods=['GET'])
def health():
    return "API is Online"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create a light NumPy array instead of a heavy Pandas DataFrame
        input_data = np.zeros(len(feature_names))
        
        # Fill the indices based on the col_map
        input_data[col_map['Month']] = int(data.get('month'))
        input_data[col_map['Date_']] = int(data.get('day'))
        input_data[col_map['Year']] = int(data.get('year'))
        
        city_col = f"City_{data.get('city')}"
        if city_col in col_map:
            input_data[col_map[city_col]] = 1
        else:
            return jsonify({"error": "City not supported"}), 400

        # Predict using reshaped array
        prediction = float(loaded_model.predict(input_data.reshape(1, -1))[0])
        category, color = get_aqi_category(prediction)

        return jsonify({
            "aqi": round(prediction, 2),
            "category": category,
            "color": color,
            "city": data.get('city')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)