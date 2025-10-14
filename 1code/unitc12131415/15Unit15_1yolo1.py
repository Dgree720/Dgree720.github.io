# pip install -U ultralytics tqdm seaborn
#15Unit15_1yolo1.py
import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')#,force_reload=True)
# print(model)
img=cv2.imread('pic/IMG_2997.JPG')
results = model(img)
results.print()
print(results.xyxy)
cv2.imshow('Unit15_1 | StudentID | YOLO01', np.squeeze(results.render()))
cv2.waitKey(0)