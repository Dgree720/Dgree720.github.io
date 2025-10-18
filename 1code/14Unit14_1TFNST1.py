# pip install tensorflow   MAC pip install tensorflow-macos
# pip install tensorflow_hub
#14Unit14_1TFNST1.py
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
# Read and process the image
content_image = cv2.imread('pic/img_2997.png')
content_image = cv2.resize(content_image, (480, 280))
cv2.imshow("original", content_image)
content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
style_image=cv2.imread('models/ss.jpg')
cv2.imshow("style", style_image)
style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
# Use numpy to convert to a float32 array, add a batch dimension, and normalize to [0,1]
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
# Resize style_image to 256 pixels (training size), content_image can have any size# Add a batch dimension（batch dimension）
# Change the shape from (height, width, channels) to (1, height, width, channels).
# Neural networks usually expect the input to contain one or more images, so this batch dimension is required.
style_image = tf.image.resize(style_image, (256, 256))
print('thinking... ')
model = hub.load('models/')
out_image = model(tf.constant(content_image), tf.constant(style_image))
img = cv2.cvtColor(np.squeeze(out_image), cv2.COLOR_RGB2BGR)
cv2.imshow("Unit14_1 | StudentID | TFNST1", img)
cv2.waitKey(0)
