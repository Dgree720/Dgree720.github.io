#6Unit06_1Face1.py
import cv2
import mediapipe as mp

mpd = mp.solutions.drawing_utils
mpfm = mp.solutions.face_mesh
dspec = mpd.DrawingSpec((0, 255, 0), 1, 1)
cspec = mpd.DrawingSpec((128, 128, 128), 1, 1)
cpoint = mpfm.FACEMESH_TESSELATION
fm = mpfm.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (800, 500))
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = fm.process(imgrgb)                      # process the imgrgb
    if results.multi_face_landmarks:
        for f_landmarks in results.multi_face_landmarks:
            mpd.draw_landmarks(image, landmark_list=f_landmarks, connections=cpoint,
                               landmark_drawing_spec=dspec, connection_drawing_spec=cspec)
    cv2.imshow('Unit06_1 | StudentID | faceLM1', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()