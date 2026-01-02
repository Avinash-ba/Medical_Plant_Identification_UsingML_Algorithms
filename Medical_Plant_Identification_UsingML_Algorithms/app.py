from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import os
import random

app = Flask(__name__)

# Demo mode - simulate plant identification without requiring TensorFlow
DEMO_MODE = True

# Sample medicinal plants for demo
SAMPLE_PLANTS = [
    'AMLA', 'BASIL', 'NEEM', 'GINGER', 'TURMERIC', 'ALOEVERA', 'TULASI',
    'CORIANDER', 'FENNEL', 'THYME', 'ROSEMARY', 'PEPPERMINT', 'SAGE'
]

# Plant information database
PLANT_INFO = {
    'AMLA': {
        'scientific': 'Phyllanthus emblica',
        'benefits': 'Rich in Vitamin C, antioxidants, supports immunity, digestion, and hair health.',
        'uses': 'Used in Ayurvedic medicine for various health conditions.'
    },
    'BASIL': {
        'scientific': 'Ocimum basilicum',
        'benefits': 'Antimicrobial, anti-inflammatory, stress relief, digestion aid.',
        'uses': 'Used in cooking and traditional medicine for respiratory and digestive issues.'
    },
    'NEEM': {
        'scientific': 'Azadirachta indica',
        'benefits': 'Antibacterial, antiviral, antifungal, blood purifier.',
        'uses': 'Used in Ayurvedic medicine for skin conditions, immunity, and dental care.'
    },
    'GINGER': {
        'scientific': 'Zingiber officinale',
        'benefits': 'Anti-nausea, anti-inflammatory, digestion aid, pain relief.',
        'uses': 'Used for motion sickness, digestion, and inflammatory conditions.'
    },
    'TULASI': {
        'scientific': 'Ocimum sanctum',
        'benefits': 'Adaptogenic, anti-stress, immune booster, respiratory health.',
        'uses': 'Sacred in Hinduism, used for stress, immunity, and respiratory issues.'
    },
    'TURMERIC': {
        'scientific': 'Curcuma longa',
        'benefits': 'Anti-inflammatory, antioxidant, immune support, joint health.',
        'uses': 'Used in cooking and medicine for inflammation and various health conditions.'
    },
    'ALOEVERA': {
        'scientific': 'Aloe barbadensis miller',
        'benefits': 'Skin healing, anti-inflammatory, digestive health, immune support.',
        'uses': 'Used topically for burns and wounds, internally for digestion.'
    }
}

def simulate_prediction():
    """Simulate plant prediction for demo purposes"""
    predicted_plant = random.choice(SAMPLE_PLANTS)
    confidence = random.uniform(0.75, 0.95)  # Random confidence between 75-95%

    # Generate mock predictions for all plants
    all_predictions = {}
    remaining_confidence = 1.0 - confidence
    other_plants = [p for p in SAMPLE_PLANTS if p != predicted_plant]

    # Distribute remaining confidence among other plants
    for plant in other_plants[:-1]:
        conf = random.uniform(0.01, remaining_confidence * 0.5)
        all_predictions[plant] = conf
        remaining_confidence -= conf

    # Last plant gets remaining confidence
    all_predictions[other_plants[-1]] = remaining_confidence

    # Add the predicted plant
    all_predictions[predicted_plant] = confidence

    return predicted_plant, confidence, all_predictions

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Please upload a valid image file'}), 400

        # Read the image (basic validation)
        try:
            image = Image.open(io.BytesIO(file.read()))
            image.verify()  # Verify it's a valid image
        except Exception:
            return jsonify({'error': 'Invalid image file'}), 400

        if DEMO_MODE:
            # Use demo prediction
            predicted_class, confidence, all_predictions = simulate_prediction()

            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'note': 'This is a demo prediction. Train a model with real data for accurate results.'
            })
        else:
            # Real prediction would go here (when TensorFlow is available)
            return jsonify({'error': 'Real model not available. Please train the model first.'}), 503

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mode': 'demo' if DEMO_MODE else 'production',
        'supported_plants': SAMPLE_PLANTS
    })

if __name__ == '__main__':
    print("Starting Medicinal Plant Identification Web App")
    print(f"Mode: {'Demo' if DEMO_MODE else 'Production'}")
    if DEMO_MODE:
        print("Demo mode: Using simulated predictions")
        print("To use real ML model, install TensorFlow and train the model")
    app.run(debug=True, host='0.0.0.0', port=5000)
