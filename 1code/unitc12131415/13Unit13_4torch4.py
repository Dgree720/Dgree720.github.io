#streamlit run 13Unit13_4torch4.py
import re
import cv2
import time
import torch
import streamlit as st
from torchvision import transforms
from transformer_net import TransformerNet

st.title("Unit13_4 | StudentID | Real-time Neural Style Transfer on Video")
models = ["candy.pth", "mosaic.pth", "rain_princess.pth", "udnie.pth"]
# use "rain_princess.pth"
selected_model = models[2]
state_dict = torch.load('models/' + selected_model, weights_only=True)
content_transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.mul(255))])
## Remove unnecessary batch norm keys to avoid version mismatch
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
style_model = TransformerNet()
style_model.load_state_dict(state_dict)
style_model.to('cpu')
style_model.eval()
original_placeholder = st.empty()   # original image
styled_placeholder = st.empty()     # styled image
cap = cv2.VideoCapture("https://cctv.klcg.gov.tw/facd4662")
# cap = cv2.VideoCapture(0)
while cap.isOpened():
    prev_time = time.time()
    success, image = cap.read()
    if not success:
        st.write("Unable to capture image from the camera")
        break
    image = cv2.resize(image, (500, 280))
    content_image = content_transform(image)  # Preprocess the image & convert to tensor format
    content_image = content_image.unsqueeze(0).to('cpu')
    # Perform inference using the style model
    with torch.no_grad():
        output = style_model(content_image).cpu()
    # Postprocessing: extract the first image, clamp pixel values, and convert to uint8 format
    img = output[0].clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")
    fps = 1 / (time.time() - prev_time)
    original_placeholder.image(image, caption="original image", channels="BGR")
    styled_placeholder.image(img, caption=f"styled image (FPS: {fps:.2f})", channels="RGB")
cap.release()
