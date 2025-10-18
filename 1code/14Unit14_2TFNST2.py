#14Unit14_2TFNST2.py
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
models = ["andy.jpg", "candy.jpg", "composition.jpg", "la_muse.jpg", "mosaic.jpg"
          , "starry_night.jpg", "the_wave.jpg", "ss.jpg", "ss2.jpg", "ss3.jpg"]
outs=[]
content_image=cv2.imread('pic/img_2997.png')
content_image = cv2.resize(content_image, (0, 0), None, 0.3, 0.3)
cv2.imshow("original", content_image)
h,w,c = content_image.shape
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
for i in range(0,len(models)):
    style_image=cv2.imread('models/'+ models[i])
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))
    model = hub.load('models/')
    out_image = model(tf.constant(content_image), tf.constant(style_image))
    outs.append(np.squeeze(out_image))
    print('thinking... model#', i)
imgStack = cv2.hconcat(outs[0:5])
imgStack2 = cv2.hconcat(outs[5:10])
imgStack3 = cv2.vconcat([imgStack, imgStack2])
for i in range(0,len(models)//2):
    cv2.putText(imgStack3, models[i], ((10 + w * i), 20), 4, 0.6, (0, 255, 255), 1)
    cv2.putText(imgStack3, models[i+5], ((10 + w * i), 20+h), 4, 0.6, (0, 0, 255), 1)
cv2.imshow("Unit14_2 | StudentID | TFNST02", np.squeeze(imgStack3))
cv2.waitKey(0)
