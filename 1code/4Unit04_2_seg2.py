# 4Unit04_2_seg2.py
import cv2
import numpy as np
import mediapipe as mp

selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(0)
bgb = np.zeros([300, 520, 3], np.uint8)
bgc = cv2.imread("pic/bgc.jpg")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    img = cv2.resize(image, (520, 300))
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(imgrgb)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    output_image = np.where(condition, img, bgb)
    output_image2 = np.where(condition, img, bgc)
    cv2.imshow("Unit04_2| 322022 |original", img)
    cv2.imshow("Unit04_2| 322022 |originalblack", output_image)
    cv2.imshow("Unit04_2| 322022 |originalcolor", output_image2)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
