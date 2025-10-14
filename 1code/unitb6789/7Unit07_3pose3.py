#7Unit07_3Face3.py
import cv2
import mediapipe as mp
import numpy as np
import yt_dlp
conn = mp.solutions.pose.POSE_CONNECTIONS
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpd = mp.solutions.drawing_utils
spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

video_url = "https://www.youtube.com/shorts/8pc9ETW0UAA"
ydl_opts = {'format': 'best',  'quiet': True }
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
stream_url = info_dict['url']

cap = cv2.VideoCapture(stream_url)
while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (450, 600))
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bkb = np.zeros(image.shape, dtype=np.uint8)
    results = pose.process(imgrgb)
    if results.pose_landmarks:
        mpd.draw_landmarks(image, results.pose_landmarks, conn, spec)
        mpd.draw_landmarks(bkb, results.pose_landmarks, conn, spec)
    cv2.imshow('Unit07_3 | StudentID | Pose3', image)
    cv2.imshow('Unit07_3 | StudentID | Pose3 Black', bkb)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()