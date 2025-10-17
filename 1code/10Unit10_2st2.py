# pip install streamlit
# streamlit run 10Unit10_2st2.py
import cv2
import streamlit as st
import yt_dlp
import mediapipe as mp

base_options = mp.tasks.BaseOptions("models/efficientdet_lite0.tflite")
options = mp.tasks.vision.ObjectDetectorOptions(base_options, score_threshold=0.2)
detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

video_url = "https://www.youtube.com/watch?v=v9rQqa_VTEY"
# video_url = "https://www.youtube.com/watch?v=XUWjAsajKXg"
ydl_opts = {"format": "best", "quiet": True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
stream_url = info_dict["url"]

cap = cv2.VideoCapture(stream_url)

st.title("Streamlit + CV2 Unit10_2 | 322022 | st2")
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])

while run:
    success, image = cap.read()
    image = cv2.resize(image, (600, 360))
    image_mp = mp.Image(mp.ImageFormat.SRGB, image)  # prepare image for mediapipe
    detection_result = detector.detect(image_mp)  # send image_mp to detector
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        cv2.rectangle(
            image,
            (bbox.origin_x, bbox.origin_y),
            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
            (100, 200, 0),
            1,
        )
        category = detection.categories[0]
        result_text = (
            category.category_name + " (" + str(round(category.score, 2)) + ")"
        )
        cv2.putText(
            image,
            result_text,
            (10 + bbox.origin_x, 20 + bbox.origin_y),
            1,
            1,
            (255, 255, 255),
            1,
        )
    # cv2.imshow('M11505005_obj_ytb', image)
    FRAME_WINDOW.image(image, channels="BGR")

cap.release()
