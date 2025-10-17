#7Unit07_2Face2.py
import cv2
import mediapipe as mp
import numpy as np
conn = mp.solutions.pose.POSE_CONNECTIONS
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpd = mp.solutions.drawing_utils
spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

cap = cv2.VideoCapture('video/pose001.mp4')
while cap.isOpened():
    success, image = cap.read()
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bkb = np.zeros(image.shape, dtype=np.uint8)
    results = pose.process(imgrgb)
    if results.pose_landmarks:
        mpd.draw_landmarks(image, results.pose_landmarks, conn, spec)
        mpd.draw_landmarks(bkb, results.pose_landmarks, conn, spec)
    cv2.imshow('Unit07_2 | StudentID | Pose2', image)
    cv2.imshow('Unit07_2 | StudentID | Pose2 Black', bkb)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()