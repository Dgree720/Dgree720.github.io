import cv2
import mediapipe as mp
import numpy as np
import math

pose = mp.solutions.pose.Pose()
conn = mp.solutions.pose.POSE_CONNECTIONS
mpd = mp.solutions.drawing_utils
spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
switch, count, count2 = 0, 0, 0
color = (0, 0, 255)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imgrgb)
    h, w, c = image.shape
    xx1 = int(w * 0.1)
    poslist = []
    if results.pose_landmarks:
        mpd.draw_landmarks(image, results.pose_landmarks, conn, spec)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            poslist.append([id, cx, cy])
    try:
        # The angle of the right elbow
        x1, y1 = poslist[12][1], poslist[12][2]
        x2, y2 = poslist[14][1], poslist[14][2]
        x3, y3 = poslist[16][1], poslist[16][2]
        right_angle = abs(
            int(
                math.degrees(
                    math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)
                )
            )
        )
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.line(image, (x3, y3), (x2, y2), (0, 255, 255), 3)
        cv2.circle(image, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(image, (x1, y1), 15, (0, 255, 255), 2)
        cv2.circle(image, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 15, (0, 0, 255), 2)
        cv2.circle(image, (x3, y3), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(image, (x3, y3), 15, (0, 255, 255), 2)
        # right hand bending on a scale of 10 to 170 degrees, maximum 100% and minimum 0%
        right_per = np.interp(right_angle, (10, 170), (100, 0))
        # the height of the bar on the Y-axis based on the degree of right hand bending, 200~400
        right_bar = int(np.interp(right_angle, (10, 170), (200, 400)))
        # rectangle represent the bar's height and also display the numerical value
        cv2.rectangle(image, (xx1, int(right_bar)), (xx1 + 30, 400), color, cv2.FILLED)
        cv2.putText(
            image,
            str(int(right_per)) + "%",
            (xx1 - 10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        # hand raise to 95% or lowerto 5%, it is considered halfway
        color = (0, 0, 255)
        if right_per >= 95:
            color = (0, 255, 0)
            if switch == 0:
                count += 0.5
                switch = 1
        if right_per <= 5:
            color = (0, 255, 0)
            if switch == 1:
                count += 0.5
                switch = 0

        # The angle of the left elbow
        x4, y4 = poslist[11][1], poslist[11][2]
        x5, y5 = poslist[13][1], poslist[13][2]
        x6, y6 = poslist[15][1], poslist[15][2]
        left_angle = abs(
            int(
                math.degrees(
                    math.atan2(y4 - y5, x4 - x5) - math.atan2(y6 - y5, x6 - x5)
                )
            )
        )
        cv2.line(image, (x4, y4), (x5, y5), (0, 255, 255), 3)
        cv2.line(image, (x6, y6), (x5, y5), (0, 255, 255), 3)
        cv2.circle(image, (x4, y4), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(image, (x4, y4), 15, (0, 255, 255), 2)
        cv2.circle(image, (x5, y5), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x5, y5), 15, (0, 0, 255), 2)
        cv2.circle(image, (x6, y6), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(image, (x6, y6), 15, (0, 255, 255), 2)
        # right hand bending on a scale of 10 to 170 degrees, maximum 100% and minimum 0%
        left_per = np.interp(left_angle, (10, 170), (100, 0))
        # the height of the bar on the Y-axis based on the degree of right hand bending, 200~400
        left_bar = int(np.interp(left_angle, (10, 170), (200, 400)))
        # rectangle represent the bar's height and also display the numerical value
        yy1 = int(w * 0.9)

        cv2.rectangle(image, (yy1, int(left_bar)), (yy1 + 30, 400), color, cv2.FILLED)
        cv2.putText(
            image,
            str(int(left_per)) + "%",
            (yy1 - 10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        # hand raise to 95% or lowerto 5%, it is considered halfway
        color = (0, 0, 255)
        if left_per >= 90:
            color = (0, 255, 0)
            if switch == 0:
                count2 += 0.5
                switch = 1
        if left_per <= 10:
            color = (0, 255, 0)
            if switch == 1:
                count2 += 0.5
                switch = 0
    except:
        pass
    cv2.putText(
        image, str(count), (xx1 - 40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6
    )

    cv2.putText(
        image, str(count2), (yy1 - 40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6
    )

    cv2.imshow("Unit07_4 | StudentID | Pose4", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
