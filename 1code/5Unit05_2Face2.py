# 5Unit05_2face2.py
import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection  # setup face detection
mp_drawing = mp.solutions.drawing_utils  # setup drawing utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgrgb)
    w, h = (image.shape[1], image.shape[0])
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
            s = detection.location_data.relative_bounding_box  # get face bounding box
            eye = int(s.width * w * 0.05)  # eye size = 1/10 facesize
            a = detection.location_data.relative_keypoints[0]  # left eye
            b = detection.location_data.relative_keypoints[1]  # right eye
            c = detection.location_data.relative_keypoints[2]  # nose
            d = detection.location_data.relative_keypoints[3]  # mouth
            e = detection.location_data.relative_keypoints[4]  # ear right
            f = detection.location_data.relative_keypoints[5]  # ear left

            ax, ay = int(a.x * w), int(a.y * h)  # real value
            bx, by = int(b.x * w), int(b.y * h)
            cx, cy = int(c.x * w), int(c.y * h)  # real value
            dx, dy = int(d.x * w), int(d.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            fx, fy = int(f.x * w), int(f.y * h)

            cv2.circle(
                image, (ax, ay), (eye + 10), (255, 255, 255), -1
            )  # draw left eye (white)
            cv2.circle(
                image, (bx, by), (eye + 10), (255, 255, 0), -1
            )  # draw right eye (white)
            cv2.circle(image, (ax, ay), eye, (0, 0, 255), -1)  # draw left eye (black)
            cv2.circle(image, (bx, by), eye, (0, 0, 255), -1)  # draw right eye (black)

            cv2.circle(image, (cx, cy), (eye + 1), (255, 0, 255), -1)

            pts1 = np.array(
                [
                    [ex, ey + 12],
                    [ex - 12, ey + 8],
                    [ex - 8, ey - 12],
                    [ex + 8, ey - 12],
                    [ex + 12, ey + 8],
                ],
                np.int32,
            )

            pts1 = pts1.reshape((-1, 1, 2))

            pts2 = np.array(
                [
                    [fx, fy + 12],
                    [fx - 12, fy + 8],
                    [fx - 8, fy - 12],
                    [fx + 8, fy - 12],
                    [fx + 12, fy + 8],
                ],
                np.int32,
            )

            pts2 = pts2.reshape((-1, 1, 2))

            cv2.polylines(image, [pts1], True, (0, 255, 0), 5)
            cv2.polylines(image, [pts2], True, (0, 255, 0), 5)

            cv2.rectangle(image, (dx - 30, dy - 5), (dx + 30, dy + 5), (255, 0, 0), -1)

    cv2.imshow("Unit05_2 | 322022 |face2", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
