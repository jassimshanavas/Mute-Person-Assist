# app.py
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import base64
import os
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('Sign_classifier.h5')

# Class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                'del', 'nothing', 'space']

def preprocess_image(image_data, target_size=(224, 224)):
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB and resize
    image = image.convert('RGB')
    image = image.resize(target_size)
    
    # Convert to numpy array and preprocess
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class_index]
        
        return jsonify({'status': 'success', 'label': predicted_label})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
