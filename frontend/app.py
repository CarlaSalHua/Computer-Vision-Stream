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

# 1. UNIQUE PAGE CONFIGURATION (must be the first st command)
st.set_page_config(page_title="Box Detector AI", page_icon="📦", layout="wide")

# 2. MAIN APPLICATION TITLE
st.title("📦 Full and Empty Box Detector with AI")

# 3. BACKEND BASE URL (easier to maintain)
BASE_URL = os.getenv("BACKEND_URL_BASE", "http://localhost:8000")
ENDPOINT_PREDICT = f"{BASE_URL}/model/predict"
ENDPOINT_STREAM = f"{BASE_URL}/model/predict_stream"


# 4. CREATE TABS FOR EACH FEATURE
tab1, tab2 = st.tabs(["Process Static Image", "Real-Time Detection"])


# --- TAB 1: LOGIC FOR UPLOADING IMAGES ---
with tab1:
    st.header("Upload an image to analyze")
    imagen_subida = st.file_uploader("📤  Choose an image", type=["jpg", "jpeg", "png"], key="uploader")

    col1, col2 = st.columns(2)

    with col1:
        if imagen_subida:
            st.image(imagen_subida, caption="📷 Original Image", use_container_width=True)

    if st.button("🚀 Process image", key="process_button"):
        if imagen_subida is not None:
            try:
                with st.spinner("Processing image... ⏳"):
                    files = {"file": (imagen_subida.name, imagen_subida.getvalue(), imagen_subida.type)}
                    response = requests.post(ENDPOINT_PREDICT, files=files)
                    
                    with col2:
                        if response.status_code == 200:
                            data = response.json()
                            img_bytes = base64.b64decode(data["image_base64"])
                            imagen_procesada = Image.open(io.BytesIO(img_bytes))
                            st.success(f"✅ Objects detected: {data['num_objects']}")
                            st.image(imagen_procesada, caption="📦 Processed Image", use_container_width=True)
                        else:
                            st.error(f"❌ Server error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Connection error with backend: {e}")
        else:
            st.warning("Please upload an image before processing.")


# --- TAB 2: LOGIC FOR REAL-TIME DETECTION ---
with tab2:
    st.header("Activate your camera for real-time detection")
    st.info("ℹ️ Allow camera access and press 'START'. You can also use your browser’s 'Share Screen' option if you prefer.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        # """ This function runs for each video frame in the stream. """
        # Convert the video frame to an image we can use
        img = frame.to_ndarray(format="bgr24")
        # 1. Encode the image to send it to the backend
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 2. Send the image to the backend
        try:
            response = requests.post(ENDPOINT_STREAM, json={"image_base64": img_base64})
            response.raise_for_status() # Raises error if response is not 2xx
            data = response.json()

            # 3. Decode the processed image received from the backend
            img_processed_bytes = base64.b64decode(data["image_base64"])
            img_processed = cv2.imdecode(np.frombuffer(img_processed_bytes, np.uint8), cv2.IMREAD_COLOR)

            # 4. Display the number of detected objects on the image
            num_objects = data["num_objects"]
            text = f"Items detected: {num_objects}"
            # Text parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 0) # Green
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = 10
            text_y = text_size[1] + 10

            # Draw the text on the processed image
            cv2.putText(img_processed, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Return the processed frame to be displayed in the frontend
            return av.VideoFrame.from_ndarray(img_processed, format="bgr24")

        except requests.exceptions.RequestException as e:
            # If there’s a connection error, do nothing and return the original frame
            st.warning(f"No se pudo conectar al backend: {e}")
            return frame # Return the original unprocessed frame

    # --- Streamlit Component ---
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

    # STUN server configuration (needed for P2P connection)
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="box-detector",
        mode=WebRtcMode.SENDRECV,  # Send our video and receive the processed video
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}, # Request video, not audio
        async_processing=True, # Asynchronous processing for better performance
    )

    st.info("ℹ️ You can select your webcam or the 'Share Screen' option in your browser’s popup window.")