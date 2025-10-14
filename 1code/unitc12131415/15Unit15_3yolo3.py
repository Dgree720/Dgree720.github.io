#15Unit15_3yolo3.py
import torch
import cv2
import numpy as np
import yt_dlp

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# video_url = "https://www.youtube.com/watch?v=v9rQqa_VTEY"
video_url = "https://www.youtube.com/watch?v=XUWjAsajKXg"

ydl_opts = {'format': 'best',  'quiet': True }
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
stream_url = info_dict['url']
cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (600, 400))
    results = model(image)
    results.print()
    # print(results.xyxy)
    cv2.imshow('Unit15_3 | StudentID | YOLO03', np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == 27:
       break
cap.release()
cv2.destroyAllWindows()