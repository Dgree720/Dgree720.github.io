import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(num_hands=2, base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgrgb)
    recognition_result = recognizer.recognize(image_mp)
    handedness_arr = recognition_result.handedness
    gesture_arr = recognition_result.gestures
    num_hands_detected = len(handedness_arr)  # or len(gesture_arr)
    for i in range(num_hands_detected):
        lr_text = "Unknown"
        if len(handedness_arr[i]) > 0:
            top_hand_category = handedness_arr[i][0]
            lr_text = top_hand_category.display_name
        gesture_text = "NoGesture"
        gesture_score = 0.0
        if len(gesture_arr[i]) > 0:
            top_gesture = gesture_arr[i][0]
            gesture_text = top_gesture.category_name
            gesture_score = top_gesture.score
        display_str = f"Hand #{i+1}: {lr_text}, {gesture_text} ({gesture_score:.2f})"
        cv2.putText(image, display_str,(30, 50 + 40*i), 2,1.0,(0, 255, 255),2 )
    cv2.imshow('Unit09_2 | StudentID | gesture26', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
