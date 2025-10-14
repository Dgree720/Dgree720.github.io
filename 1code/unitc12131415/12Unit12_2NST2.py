#12Unit12_2NST2.py
import cv2
models = ["la_muse.t7","the_scream.t7","composition_vii.t7","starry_night.t7","la_muse_eccv16.t7"
          ,"udnie.t7","mosaic.t7","candy.t7","feathers.t7","the_wave.t7"]# Define a list of style models
outs=[]                    # Used to store all the stylized output images
image = cv2.imread('pic/IMG_2997.png')
image = cv2.resize(image,(0, 0), None, 0.3,0.3)
(h, w) = image.shape[:2]    # Get the height and width of the image
for i in range(0,len(models)):
    net = cv2.dnn.readNetFromTorch('models/'+ models[i])
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680))
    net.setInput(blob)
    out = net.forward()
    out = out[0]              # Change the shape from (1, C, H, W) to (C, H, W)
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    outs.append(out)       # Save the stylized result
imgStack = cv2.hconcat(outs[0:5])
imgStack2 = cv2.hconcat(outs[5:10])
imgStack3 = cv2.vconcat([imgStack, imgStack2])
for i in range(0,len(models)//2):
    cv2.putText(imgStack3, models[i], ((10 + w * i), 20), 2, 0.6, (255,255, 255))
    cv2.putText(imgStack3, models[i+5], ((10 + w * i), 20+h), 2, 0.6, (0, 0, 255))
cv2.imshow('Unit12_2 | StudentID | All styles', imgStack3)
cv2.imshow('Unit12_2 | StudentID | original image', image)
cv2.waitKey(0)
