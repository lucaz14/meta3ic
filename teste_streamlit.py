# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:54:31 2024

@author: guerr
"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
import pickle
from PIL import Image

# Configuração do Streamlit
st.title("Classificador de Flores")
st.text("Faça upload de uma imagem de flor para classificação.")

# Carregar o modelo salvo
model = load_model("flower_model_simple.h5")

with open("label_encoder.pkl", "rb") as f:
    nomes_codif = pickle.load(f)

# Classes de flores
flower_classes = list(nomes_codif.inverse_transform(range(len(nomes_codif.classes_))))
IMG_SIZE = 224

# Função de predição
def predict_image(image):
    try:
        # Pré-processamento com Pillow
        img = image.resize((IMG_SIZE, IMG_SIZE))  # Redimensionar a imagem
        img = np.array(img) / 255.0  # Normalizar
        img = np.expand_dims(img, axis=0)  # Adicionar dimensão para lotes (batch)

        # Predição
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        predicted_class = flower_classes[class_index]
        confidence = float(predictions[0][class_index])

        return predicted_class, confidence
    except Exception as e:
        return None, str(e)

# Interface de upload no Streamlit
uploaded_file = st.file_uploader("Escolha uma imagem de flor...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir a imagem com Pillow
    image = Image.open(uploaded_file)

    if image is not None:
        # Mostrar a imagem carregada
        st.image(image, channels="BGR", caption="Imagem carregada", use_container_width=True)

        # Fazer a predição
        predicted_class, confidence = predict_image(image)

        if predicted_class:
            st.success(f"Classe Predita: {predicted_class}")
            st.info(f"Confiança: {confidence:.2f}")
        else:
            st.error(f"Erro: {confidence}")
    else:
        st.error("Não foi possível processar a imagem. Certifique-se de que é uma imagem válida.")
