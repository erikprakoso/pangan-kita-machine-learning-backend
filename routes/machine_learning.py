# routes/ml.py

from flask import request
from tensorflow.keras.models import load_model
from utils.image import load_image
import numpy as np
import json
from . import ml_bp
from utils.response import standard_response

# Define the DepthwiseConv2D layer as a custom layer
from tensorflow.keras.layers import DepthwiseConv2D


class MyDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' argument
        super().__init__(**kwargs)


MODEL_PATH = 'mobilenet3.h5'
model = load_model(MODEL_PATH, custom_objects={
                   'DepthwiseConv2D': MyDepthwiseConv2D}, compile=False)

with open('labels.json', 'r') as file:
    class_labels = json.load(file)
class_labels = {int(k): v for k, v in class_labels.items()}


@ml_bp.route('/predictions', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return standard_response("ERROR", "No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return standard_response("ERROR", "No selected file"), 400

    if file:
        img_array = load_image(file, target_size=(224, 224))
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        predicted_label = class_labels[predicted_class]
        data = {
            "filename": file.filename,
            'prediction': str(predicted_label),
            'probability': str(predictions[0].tolist())
        }
        return standard_response("SUCCESS", "Prediction successful", data)
