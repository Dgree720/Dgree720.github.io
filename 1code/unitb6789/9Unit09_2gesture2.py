# 9Unit09_2Face2.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
base_options = python.BaseOptions(
    model_asset_path="/home/andreas/Documents/Python/Dgree720.github.io/1code/models/gesture_recognizer.task"
)
options = vision.GestureRecognizerOptions(num_hands=2, base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgrgb)
    recognition_result = recognizer.recognize(image_mp)
    # print(recognition_result)
    for i, gesture in enumerate(recognition_result.gestures):
        top_gesture = gesture[0]
        gesture_name = top_gesture.category_name
        score = top_gesture.score

        side = recognition_result.handedness[i][0].category_name

        if side == "Right":
            cv2.putText(
                image,
                f"Right: {gesture_name} ({score:.2f})",
                (30, 100),
                1,
                1,
                (0, 255, 255),
                2,
            )

        if side == "Left":
            cv2.putText(
                image,
                f"Left: {gesture_name} ({score:.2f})",
                (30, 50),
                1,
                1,
                (0, 255, 255),
                2,
            )
    cv2.imshow("Unit09_2 | 322022 | gesture2", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
# 👍, 👎, ✌️, ☝️, ✊, 👋, 🤟
# 0 - Unrecognized gesture, label: Unknown
# 1 - Closed fist, label: Closed_Fist
# 2 - Open palm, label: Open_Palm
# 3 - Pointing up, label: Pointing_Up
# 4 - Thumbs down, label: Thumb_Down
# 5 - Thumbs up, label: Thumb_Up
# 6 - Victory, label: Victory
# 7 - Love, label: ILoveYou
