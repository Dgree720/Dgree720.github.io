# 5Unit05_1face.py
import cv2
import mediapipe as mp

face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.2
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgrgb)
    if results.detections:
        for detection in results.detections:
            mp.solutions.drawing_utils.draw_detection(image, detection)
    cv2.imshow("Unit05_1 | 322022 |face", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
