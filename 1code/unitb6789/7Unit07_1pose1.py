#7Unit07_1Face1.py
import cv2
import mediapipe as mp

conn = mp.solutions.pose.POSE_CONNECTIONS
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpd = mp.solutions.drawing_utils
spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
print(spec)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (800, 500))
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imgrgb)

    if results.pose_landmarks:
        mpd.draw_landmarks(image, results.pose_landmarks, conn, spec)
    cv2.imshow('Unit07_1 | StudentID | Pose1', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()