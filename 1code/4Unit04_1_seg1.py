#4Unit04_1_seg1.py
import cv2
import numpy as np
import mediapipe as mp

selfie_segmentation = (
    mp.solutions.selfie_segmentation.SelfieSegmentation(0))        # 0 general-purpose, 1 landscape images
bgb = np.zeros([300,520,3], np.uint8)                        #black background
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    img = cv2.resize(image,(520,300))                        # resize image to 520x300
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(imgrgb)                  # process selfie_segmentation and get result
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # In summary, this code's purpose is to create a condition or
    # boolean mask (condition) that checks whether the values in results.segmentation_mask
    # are greater than 0.1. This condition can be used for various purposes, such as filtering
    # or selecting elements that meet this specific threshold in your data.
    output_image = np.where(condition, img, bgb)
    # create an output_image that combines elements from two input arrays (img and bgb)
    # based on the condition. Where the condition is met, you get the corresponding pixel from img,
    # and where it's not met, you get the corresponding pixel from bgb.
    cv2.imshow('Unit04_1 | StudentID |original', img)
    cv2.imshow('Unit04_1 | StudentID |selfie_segmentation1', output_image)
    if cv2.waitKey(5)  & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()