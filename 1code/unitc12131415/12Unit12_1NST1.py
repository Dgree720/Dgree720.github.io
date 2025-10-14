# File: Unit12_1NST1.py
import cv2
#1️⃣ Load and Resize the Input Image
input_path = 'pic/IMG_2997.png'
model_path = 'models/starry_night.t7'
image = cv2.imread(input_path) # Read and resize the image
image = cv2.resize(image, (600, 400))
# 2️⃣ Preprocess: Convert Image to a Blob required by the DNN model
# The mean values (103.939, 116.779, 123.680) are subtracted from each channel (BGR order).
blob = cv2.dnn.blobFromImage(
    image, 1.0, (600, 400),(103.939, 116.779, 123.680),swapRB=False, crop=False)
# 3️⃣ Load the Pretrained .t7 Model
print("🧠 Loading Torch model...")
net = cv2.dnn.readNetFromTorch(model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
print("✅ Model loaded successfully!")
# 4️⃣ Forward Pass: Apply Style Transfer
net.setInput(blob)
out = net.forward()  # Output shape: (1, 3, H, W)
out = out[0]
# 5️⃣ Postprocess the Output Image Add back the mean values
out[0] += 103.939
out[1] += 116.779
out[2] += 123.680
out /= 255.0   # Normalize pixel values to [0, 1]
# Reorder dimensions from (C, H, W) to (H, W, C) for OpenCV display
out = out.transpose(1, 2, 0)
cv2.imshow('Unit12_1 | Original Image', image)
cv2.imshow('Unit12_1 | Styled Image (Starry Night)', out)
cv2.waitKey(0)
cv2.destroyAllWindows()