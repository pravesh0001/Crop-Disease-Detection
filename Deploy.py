from flask import Flask, request, jsonify, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Ensure the 'uploads' directory exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Define the expected image size based on the model's requirements
IMAGE_SIZE = 256  # Update to match the input shape of your model

# Load the trained model
model = tf.keras.models.load_model('crop_disease_model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict_disease(image_path):
    """Predict the disease from the given image."""
    image = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to match the model's input
    img_array = tf.keras.preprocessing.image.img_to_array(image)  # Convert image to array
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    """Handle the home page and predictions."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        file_path = os.path.join('static/uploads', file.filename)  # Save the file in static/uploads
        file.save(file_path)

        # Predict the disease
        predicted_class, confidence = predict_disease(file_path)

        # Fertilizer suggestions based on predictions
        fertilizer_suggestions = {
            'Potato___Early_blight': 'Use a fungicide containing mancozeb or chlorothalonil.',
            'Potato___Late_blight': 'Use a fungicide with metalaxyl or fosetyl-Al.',
            'Potato___healthy': 'No fertilizer needed. Maintain healthy practices.'
        }
        fertilizer = fertilizer_suggestions.get(predicted_class, "No suggestion available")

        return render_template('index.html', 
                               predicted_class=predicted_class, 
                               confidence=f"{confidence:.2f}", 
                               fertilizer=fertilizer, 
                               image_url=url_for('static', filename=f'uploads/{file.filename}'))
    
    return render_template('index.html', predicted_class=None, image_url=None)

if __name__ == '__main__':
    app.run(debug=True)