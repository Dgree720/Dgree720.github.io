#13Unit13_3torch3.py
import re
import cv2
import time
import torch
from torchvision import transforms
from transformer_net import TransformerNet
models = ["candy.pth", "mosaic.pth", "rain_princess.pth", "udnie.pth"]
content_transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.mul(255))])
cap = cv2.VideoCapture(0)
style_model = TransformerNet()
state_dict = torch.load('models/' + models[2], weights_only=True)
# Remove unnecessary weight keys to avoid version mismatch issues
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
style_model.load_state_dict(state_dict)
style_model.to('cpu')
style_model.eval()

while cap.isOpened():
    prev_time = time.time()
    success, image = cap.read()
    image = cv2.resize(image, (500, 280))
    content_image = content_transform(image) # Preprocess the image & convert to tensor format
    content_image = content_image.unsqueeze(0).to('cpu')
    with torch.no_grad():                    # Perform inference using the style model
        output = style_model(content_image).cpu()
    img = output[0].clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print(1 / (time.time() - prev_time))
    cv2.imshow("Unit13_3 | StudentID | output", img)
    cv2.imshow("Unit13_3 | StudentID | original", image)
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()