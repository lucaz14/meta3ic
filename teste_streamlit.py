# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:54:31 2024

@author: guerr
"""

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
import pickle

# Configuração do Streamlit
st.title("Classificador de Flores")
st.text("Faça upload de uma imagem de flor para classificação.")

# Carregar o modelo salvo
model = load_model("flower_model.h5")
model.compile(optimizer=Adam(learning_rate=0.00122195324402186004), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

with open("label_encoder.pkl", "rb") as f:
    nomes_codif = pickle.load(f)

# Classes de flores
flower_classes = list(nomes_codif.inverse_transform(range(len(nomes_codif.classes_))))
IMG_SIZE = 224

# Função de predição
def predict_image(image):
    try:
        # Pré-processamento
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

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
    # Converter o arquivo carregado em uma imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        # Mostrar a imagem carregada
        st.image(image, channels="BGR", caption="Imagem carregada", use_column_width=True)

        # Fazer a predição
        predicted_class, confidence = predict_image(image)

        if predicted_class:
            st.success(f"Classe Predita: {predicted_class}")
            st.info(f"Confiança: {confidence:.2f}")
        else:
            st.error(f"Erro: {confidence}")
    else:
        st.error("Não foi possível processar a imagem. Certifique-se de que é uma imagem válida.")
