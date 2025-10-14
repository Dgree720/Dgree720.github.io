#15Unit15_5yolo5.py
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
# model.predict(source="https://www.youtube.com/watch?v=XUWjAsajKXg",
#               show =True, save = False ,line_width= 2)
# model.predict(source="https://www.youtube.com/watch?v=-EG5E7SLVGo",
#               show =True, save = False ,line_width= 2)
model.predict(source="https://www.youtube.com/watch?v=VvC-gWcHhrI",
              show =True, save = False ,line_width= 2)

