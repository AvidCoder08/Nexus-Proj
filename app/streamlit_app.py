"""
Streamlit demo skeleton for inference.
Run with:
    streamlit run app/streamlit_app.py

This placeholder loads a saved model and runs prediction on uploaded images.
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title('Fracture Detection Demo')

uploaded_file = st.file_uploader('Upload an X-ray image', type=['png','jpg','jpeg'])

MODEL_PATH = 'models/best_model.h5'

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.warning(f'Could not load model at {path}. Train a model first or adjust MODEL_PATH.\n{e}')
        return None


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = load_model()
    if model is not None:
        img = image.resize((224,224))
        arr = np.array(img)/255.0
        inp = np.expand_dims(arr, 0)
        pred = model.predict(inp)[0][0]
        st.write('Prediction (probability of fracture):', float(pred))
        label = 'Fracture' if pred >= 0.5 else 'Normal'
        st.subheader(label)
