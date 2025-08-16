import streamlit as st
import requests
from PIL import Image
import io
import base64
import av
import os
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. CONFIGURACIÓN ÚNICA DE LA PÁGINA (debe ser el primer comando st)
st.set_page_config(page_title="Box Detector AI", page_icon="📦", layout="wide")

# 2. TÍTULO PRINCIPAL DE LA APLICACIÓN
st.title("📦 Detector de Cajas Llenas y Vacías con IA")

# 3. URL BASE DEL BACKEND (más fácil de mantener)
BASE_URL = os.getenv("BACKEND_URL_BASE", "http://localhost:8000")
ENDPOINT_PREDICT = f"{BASE_URL}/model/predict"
ENDPOINT_STREAM = f"{BASE_URL}/model/predict_stream"


# 4. CREACIÓN DE PESTAÑAS PARA CADA FUNCIONALIDAD
tab1, tab2 = st.tabs(["Procesar Imagen Estática", "Detección en Tiempo Real"])


# --- PESTAÑA 1: LÓGICA PARA SUBIR IMÁGENES ---
with tab1:
    st.header("Sube una imagen para analizar")
    imagen_subida = st.file_uploader("📤 Elige una imagen", type=["jpg", "jpeg", "png"], key="uploader")

    col1, col2 = st.columns(2)

    with col1:
        if imagen_subida:
            st.image(imagen_subida, caption="📷 Imagen Original", use_container_width=True)

    if st.button("🚀 Procesar imagen", key="process_button"):
        if imagen_subida is not None:
            try:
                with st.spinner("Procesando imagen... ⏳"):
                    files = {"file": (imagen_subida.name, imagen_subida.getvalue(), imagen_subida.type)}
                    response = requests.post(ENDPOINT_PREDICT, files=files)
                    
                    with col2:
                        if response.status_code == 200:
                            data = response.json()
                            img_bytes = base64.b64decode(data["image_base64"])
                            imagen_procesada = Image.open(io.BytesIO(img_bytes))
                            st.success(f"✅ Objetos detectados: {data['num_objects']}")
                            st.image(imagen_procesada, caption="📦 Imagen Procesada", use_container_width=True)
                        else:
                            st.error(f"❌ Error del servidor: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Error de conexión con el backend: {e}")
        else:
            st.warning("Por favor, sube una imagen antes de procesar.")


# --- PESTAÑA 2: LÓGICA PARA DETECCIÓN EN TIEMPO REAL ---
with tab2:
    st.header("Activa tu cámara para detectar en tiempo real")
    st.info("ℹ️ Permite el acceso a la cámara y presiona 'START'. Puedes usar la opción de 'Compartir Pantalla' de tu navegador si lo prefieres.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # """ Esta función se ejecuta por cada fotograma del stream de video. """
    # Convertir el fotograma de video a una imagen que podamos usar
        img = frame.to_ndarray(format="bgr24")
        # 1. Codificar la imagen para enviarla al backend
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 2. Enviar la imagen al backend
        try:
            response = requests.post(ENDPOINT_STREAM, json={"image_base64": img_base64})
            response.raise_for_status() # Lanza un error si la respuesta no es 2xx
            data = response.json()

            # 3. Decodificar la imagen procesada recibida del backend
            img_processed_bytes = base64.b64decode(data["image_base64"])
            img_processed = cv2.imdecode(np.frombuffer(img_processed_bytes, np.uint8), cv2.IMREAD_COLOR)

            # 4. Mostrar el número de objetos detectados sobre la imagen
            num_objects = data["num_objects"]
            text = f"Items detectados: {num_objects}"
            # Parámetros para el texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 0) # Verde
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = 10
            text_y = text_size[1] + 10

            # Dibujar el texto en la imagen procesada
            cv2.putText(img_processed, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Devolver el fotograma procesado para mostrarlo en el frontend
            return av.VideoFrame.from_ndarray(img_processed, format="bgr24")

        except requests.exceptions.RequestException as e:
            # Si hay un error de conexión, no hacemos nada y devolvemos el frame original
            st.warning(f"No se pudo conectar al backend: {e}")
            return frame # Devuelve el frame original sin procesar

    # --- Componente de Streamlit ---
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

    # Configuración para servidores STUN (necesario para la conexión P2P)
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="box-detector",
        mode=WebRtcMode.SENDRECV, # Enviar nuestro video y recibir el video procesado
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}, # Pedir acceso a video, no audio
        async_processing=True, # Procesamiento asíncrono para mejor rendimiento
    )

    st.info("ℹ️ Puedes elegir tu cámara web o la opción 'Compartir Pantalla' en la ventana emergente de tu navegador.")