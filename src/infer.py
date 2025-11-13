"""
Simple inference utility: load model, preprocess single image, return probability and label.
"""
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


def load_image(path, target_size=(224,224)):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img)/255.0
    return arr


def predict_image(model_path, image_path, threshold=0.5):
    model = tf.keras.models.load_model(model_path)
    arr = load_image(image_path)
    inp = np.expand_dims(arr, 0)
    prob = float(model.predict(inp)[0][0])
    label = 'Fracture' if prob >= threshold else 'Normal'
    return {'probability': prob, 'label': label}


if __name__ == '__main__':
    # Example (adjust paths)
    mp = 'models/best_model.h5'
    imgp = 'example.jpg'
    print('Run predict_image with a valid model and image path')
