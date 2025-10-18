#12Unit12_4NST4.py
import cv2
import time
models = ["la_muse.t7","the_scream.t7","composition_vii.t7","starry_night.t7","la_muse_eccv16.t7","udnie.t7","mosaic.t7","candy.t7","feathers.t7","the_wave.t7"]
outs=[]
nets = []
cap = cv2.VideoCapture(0)
while cap.isOpened():
    prev_time = time.time()
    success, image = cap.read()
    image = cv2.resize(image,(0, 0), None, 0.15, 0.15)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=True, crop=False)
    for i in range(0, len(models)):
        net = cv2.dnn.readNetFromTorch('models/'+models[i])
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setInput(blob)
        out = net.forward()
        out = out[0]
        out[0] += 103.939
        out[1] += 116.779
        out[2] += 123.68
        out /= 255
        out = out.transpose(1, 2, 0)
        outs.append(out)
    imgStack = cv2.hconcat(outs[0:5])
    imgStack2 = cv2.hconcat(outs[5:10])
    imgStack3 = cv2.vconcat([imgStack, imgStack2])
    fps = 'fps: ' + str(round(1 / (time.time() - prev_time), 3))
    cv2.putText(image, fps, (20, 20), 2, 0.6, (0, 255, 255))
    cv2.imshow('Unit12_4 | StudentID | output', imgStack3)
    cv2.imshow("Unit12_4 | StudentID | original", image)
    outs =[]
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()