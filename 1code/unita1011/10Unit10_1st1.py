# pip install streamlit
# streamlit run 10Unit10_1st1.py
import cv2
import streamlit as st

cap = cv2.VideoCapture(0)

st.title("Streamlit + CV2 Unit10_1 | 322022 | st1")
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])

while run:
    success, frame = cap.read()
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
