#15Unit15_2yolo2.py
import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')#,force_reload=True)
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("video/jump.mp4")
# cap = cv2.VideoCapture("https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=13380")
# cap = cv2.VideoCapture("https://cctv1.kctmc.nat.gov.tw/f75bb280?t=0.9768836315531848")
cap = cv2.VideoCapture("https://cctv.klcg.gov.tw/facd4662")

while cap.isOpened():
    success, image = cap.read()
    results = model(image)
    results.print()
    print(results.xyxy)
    cv2.imshow('Unit15_2 | StudentID |YOLO02', np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == 27:
       break
cap.release()
cv2.destroyAllWindows()