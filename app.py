from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import gc

app = Flask(__name__)
CORS(app)

# --- GLOBAL MODEL LOADING ---
# mmap_mode='r' reads from disk. We load columns as a list for speed.
try:
    loaded_model = joblib.load('AQI_Forecasting.pkl', mmap_mode='r')
    feature_names = joblib.load('model_columns.pkl')
    # Pre-calculate index mapping to avoid searching strings in every request
    col_map = {name: i for i, name in enumerate(feature_names)}
    print("🚀 Model loaded. Memory optimization active.")
except Exception as e:
    print(f"❌ Critical Error: {e}")

def get_aqi_category(prediction):
    if prediction <= 50: return "Good", "#10b981"
    elif prediction <= 100: return "Satisfactory", "#9cd84e"
    elif prediction <= 200: return "Moderate", "#f59e0b"
    elif prediction <= 300: return "Poor", "#ff8c42"
    elif prediction <= 400: return "Very Poor", "#ef4444"
    else: return "Severe", "#7f1d1d"

@app.route('/', methods=['GET'])
def health():
    return "Atmosphere API Online"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Initialize a light NumPy array instead of a heavy Pandas DataFrame
        input_array = np.zeros(len(feature_names))
        
        # Map values directly to indices
        if 'Month' in col_map: input_array[col_map['Month']] = int(data.get('month'))
        if 'Date_' in col_map: input_array[col_map['Date_']] = int(data.get('day'))
        if 'Year' in col_map: input_array[col_map['Year']] = int(data.get('year'))
        
        city_col = f"City_{data.get('city')}"
        if city_col in col_map:
            input_array[col_map[city_col]] = 1
        else:
            return jsonify({"error": "City not supported"}), 400

        # Reshape for scikit-learn (1 sample, N features)
        prediction = float(loaded_model.predict(input_array.reshape(1, -1))[0])
        category, color = get_aqi_category(prediction)

        # Explicitly trigger garbage collection to free memory
        gc.collect()

        return jsonify({
            "aqi": round(prediction, 2),
            "category": category,
            "color": color,
            "city": data.get('city')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment or default to 10000 for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)