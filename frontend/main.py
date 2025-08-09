import streamlit as st
from PIL import Image
import io
import random
import os

st.set_page_config(
    page_title="Computer Vision Project",
    page_icon="👋",
)

st.write("# Computer Vision Project! 👋")

# Configuración de la app
st.set_page_config(page_title="Box Detector", page_icon="📦", layout="centered")
st.title("📦 Empty and Full Box Detector with AI")
st.markdown("Upload an image to let the model detect empty and full boxes.")

# Ruta del dataset simulado
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# Subida de imagen
imagen_subida = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if imagen_subida:
    st.image(imagen_subida, caption="📷 Original Image", use_container_width=True)

if st.button("🚀 Procesar imagen"):
    if imagen_subida:
        with st.spinner("Procesando imagen (modo demo)... ⏳"):
            try:
                # Selecciona una imagen aleatoria del dataset
                ejemplos = [f for f in os.listdir(DATASET_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
                if not ejemplos:
                    st.error("No se encontraron imágenes de prueba en la carpeta 'dataset'.")
                else:
                    imagen_demo_path = os.path.join(DATASET_DIR, random.choice(ejemplos))
                    imagen_demo = Image.open(imagen_demo_path)
                    st.success("✅ Procesamiento simulado completado")
                    st.image(imagen_demo, caption="📦 Imagen Procesada (demo)", use_container_width=True)
            except Exception as e:
                st.error(f"Error al cargar imagen de demo: {str(e)}")
    else:
        st.warning("Por favor, sube una imagen antes de procesar.")