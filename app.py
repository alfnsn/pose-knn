import streamlit as st
import pickle
import numpy as np
from PIL import Image

with open('pose.pkl', 'rb') as f:
    pose_encoder = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('best_knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

def resize1(file_path):
    img = Image.open(file_path).convert('L')  
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, -1)
    return img

st.title("Prediksi Pose")

uploaded_file = st.file_uploader("Upload gambar", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption='Gambar yang diupload', use_container_width=True)

    with col2:
        img_array = resize1(uploaded_file)
        img_pca = pca.transform(img_array)
        prediction = model.predict(img_pca)
        pred_label = pose_encoder.inverse_transform(prediction)[0]

        st.markdown("<h1>Hasil Prediksi</h1>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding: 20px; background-color: #1f1f2e; border-radius: 10px;">
            <h3 style="color: #f9c74f;">Prediksi:</h3>
            <p style="font-size: 28px; color: #90be6d;"><b>{pred_label.upper()}</b></p>
        </div>
        """, unsafe_allow_html=True)
