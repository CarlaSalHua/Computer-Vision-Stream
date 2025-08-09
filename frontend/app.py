import streamlit as st
import requests
from PIL import Image
import io
import os
import base64

st.set_page_config(page_title="Box Detector", page_icon="📦", layout="centered")
st.title("📦 Empty and Full Box Detector with AI")
st.markdown("Upload an image to let the model detect empty and full boxes.")

imagen_subida = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida:
    st.image(imagen_subida, caption="📷 Imagen Original", use_container_width=True)

if st.button("🚀 Procesar imagen"):
    if imagen_subida is not None:
        try:
            with st.spinner("Procesando imagen... ⏳"):
                files = {"file": (imagen_subida.name, imagen_subida.getvalue(), imagen_subida.type)}
                BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
                response = requests.post(f"{BACKEND_URL}/model/predict", files=files)

                if response.status_code == 200:
                    data = response.json()
                    img_bytes = base64.b64decode(data["image_base64"])
                    imagen_procesada = Image.open(io.BytesIO(img_bytes))
                    st.success(f"✅ Procesamiento completado – Objetos detectados: {data['num_objects']}")
                    st.image(imagen_procesada, caption="📦 Imagen Procesada", use_container_width=True)
                else:
                    st.error(f"❌ Error al procesar la imagen. Código: {response.status_code}")
        except Exception as e:
            st.error(f"⚠️ Error al conectarse con el backend: {str(e)}")
    else:
        st.warning("Por favor, sube una imagen antes de procesar.")