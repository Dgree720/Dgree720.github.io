#8Unit08_3hand3.py
import cv2
import mediapipe as mp
import random

mpd = mp.solutions.drawing_utils
lm_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
conn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()
mphc = mp.solutions.hands.HAND_CONNECTIONS
hands = mp.solutions.hands.Hands(model_complexity=0, max_num_hands=2)

cap = cv2.VideoCapture(0)
run = True
rx, ry, count = 0, 0, 0
color_box = (0, 0, 255)
while cap.isOpened():
    success, image = cap.read()
    img = cv2.resize(image, (640, 420))
    w, h = (img.shape[1], img.shape[0])
    if run:
        run = False
        rx = random.randint(10, w - 80)
        ry = random.randint(10, h - 80)
        color_box = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        print("New box:", rx, ry, "Color:", color_box)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgrgb)
    if results.multi_hand_landmarks:
        for h_landmarks in results.multi_hand_landmarks:
            mpd.draw_landmarks(img, h_landmarks, mphc, lm_style, conn_style)
            x = h_landmarks.landmark[20].x * w
            y = h_landmarks.landmark[20].y * h
            if (rx < x < rx + 80) and (ry < y < ry + 80):
                count += 1
                run = True
    cv2.rectangle(img, (rx, ry), (rx + 80, ry + 80), color_box, 5)
    img = cv2.flip(img, 1)
    cv2.putText(img, 'Score:'+str(count), (30, 80), 2, 2, (0, 0, 255), 2)
    cv2.imshow('Unit08_3 | StudentID | hand3', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
