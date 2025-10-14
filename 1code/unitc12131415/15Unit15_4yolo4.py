#15Unit15_4yolo4.py
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
model.track(source="video/jump.mp4",
              show =True, save = False ,line_width= 2)
# model.predict(source="video/jump.mp4",
#               show =True, save = False ,line_width= 2)
# model.track(source="https://cctv.klcg.gov.tw/facd4662",
#               show =True, save = False ,line_width= 2)
# model.predict(source="https://cctv.klcg.gov.tw/facd4662",
#               show =True, save = False , line_width= 2)
