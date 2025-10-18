#streamlit run 12Unit12_5NST5.py
import streamlit as st
import cv2
import time
import numpy as np
st.title("Unit12_5|StudentID|Real-time Style Transfer")
run = st.checkbox('Run')
models = ["la_muse.t7", "the_scream.t7", "composition_vii.t7", "starry_night.t7",
          "la_muse_eccv16.t7", "udnie.t7", "mosaic.t7", "candy.t7", "feathers.t7", "the_wave.t7"]
selected_model = st.sidebar.selectbox("Select a style transfer model", models)
model_path = 'models/' + selected_model
net = cv2.dnn.readNetFromTorch(model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("https://cctv.klcg.gov.tw/facd4662")
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
placeholder_original = st.empty()
placeholder_styled = st.empty()

while run and cap.isOpened():
    prev_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (500, 280))
    # Preprocessing: create a blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (500, 280), (103.939, 116.779, 123.680))
    net.setInput(blob)
    out = net.forward()[0]
    # Postprocessing: add back mean values, normalize, and reorder axes
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    # 計算 FPS
    fps = round(1 / (time.time() - prev_time), 3)
    frame_text = cv2.putText(frame.copy(), f"fps: {fps}", (20, 20),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    frame_rgb = cv2.cvtColor(frame_text, cv2.COLOR_BGR2RGB)
    styled = np.clip(out * 255, 0, 255).astype(np.uint8)
    styled_rgb = cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)
    placeholder_original.image(frame_rgb, channels="RGB", caption="Original Image")
    placeholder_styled.image(styled_rgb, channels="RGB", caption="Stylized Image")
cap.release()