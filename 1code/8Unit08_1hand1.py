#8Unit08_1Face1.py
import cv2
import mediapipe as mp

mpd = mp.solutions.drawing_utils
lm_style=mp.solutions.drawing_styles.get_default_hand_landmarks_style()
conn_style=mp.solutions.drawing_styles.get_default_hand_connections_style()
mphc = mp.solutions.hands.HAND_CONNECTIONS
hands = mp.solutions.hands.Hands(model_complexity=0,  max_num_hands=2)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgrgb)
    if results.multi_hand_landmarks:
        for h_landmarks in results.multi_hand_landmarks:
            mpd.draw_landmarks(image, h_landmarks, mphc, lm_style, conn_style)
    cv2.imshow('Unit08_1 | StudentID | hand1', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()