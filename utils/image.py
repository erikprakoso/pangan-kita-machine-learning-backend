from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io

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
