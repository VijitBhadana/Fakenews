from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load model and vectorizer
model_dir = 'model'
model_path = os.path.join(model_dir, 'model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or vectorizer: {e}")

@app.route('/')
def home():
    return "✅ Fake News Detection API is Running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400

        input_text = data['text']

        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]

        # Try to also extract probability if model supports it
        if hasattr(model, "predict_proba"):
            confidence = round(float(max(model.predict_proba(input_vector)[0])), 2)
        else:
            confidence = None

        label = 'Real' if prediction == 1 else 'Fake'

        return jsonify({
            'prediction': label,
            'confidence': confidence
        })

    except Exception as e:
        print("❌ Prediction error:", traceback.format_exc())
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Run on port 5001 to avoid clash with Node.js backend
    app.run(host='0.0.0.0', port=5001, debug=True)
