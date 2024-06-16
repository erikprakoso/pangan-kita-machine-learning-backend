from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import io
from tensorflow.keras.layers import DepthwiseConv2D

app = Flask(__name__)

# Define the DepthwiseConv2D layer as a custom layer


class MyDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' argument
        super().__init__(**kwargs)


# Load the model with the custom object
MODEL_PATH = 'mobilenet3.h5'
model = load_model(MODEL_PATH, custom_objects={
                   'DepthwiseConv2D': MyDepthwiseConv2D}, compile=False)

with open('labels.json', 'r') as file:
    class_labels = json.load(file)
class_labels = {int(k): v for k, v in class_labels.items()}


def load_image(file, target_size=(224, 224)):
    # Open the file stream
    file_stream = io.BytesIO(file.read())

    # Load the image from the file stream
    img = load_img(file_stream, target_size=target_size)

    # Convert the image to array
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        img_array = load_image(file, target_size=(224, 224))
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        predicted_label = class_labels[predicted_class]
        return jsonify({
            "filename": file.filename,
            'prediction': str(predicted_label),
            'probability': str(predictions[0].tolist())
        }), 200


if __name__ == '__main__':
    app.run(debug=True)
