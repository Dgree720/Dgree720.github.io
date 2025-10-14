#12Unit12_3NST3.py
import cv2
import time
models = ["la_muse.t7","the_scream.t7","composition_vii.t7","starry_night.t7","la_muse_eccv16.t7"
          ,"udnie.t7","mosaic.t7","candy.t7","feathers.t7","the_wave.t7"]
outs=[]
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromTorch('models/'+models[3])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

while cap.isOpened():
    prev_time = time.time()
    success, image = cap.read()
    image = cv2.resize(image, (500, 280))
    blob = cv2.dnn.blobFromImage(image, 1.0, (500, 280), (103.939, 116.779, 123.680))
    net.setInput(blob)
    out = net.forward()
    out = out[0]
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    fps='fps: '+str(round(1 / (time.time() - prev_time),3))
    cv2.putText(image,fps, (20, 20), 2, 0.6, ( 0,255,255))
    cv2.imshow('Unit12_3 | StudentID | styled video', out)
    cv2.imshow('Unit12_3 | StudentID | original video', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()