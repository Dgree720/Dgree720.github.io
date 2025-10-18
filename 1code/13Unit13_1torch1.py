# pip install torch torchvision torchaudio
# File: Unit13_1torch1.py
import re
import cv2
import torch
from torchvision import transforms
from transformer_net import TransformerNet  # Make sure this file is available
# 1️⃣ Load and Prepare Input Image
content_imageO = cv2.imread('pic/IMG_2997.png')  # original image path
content_image = cv2.cvtColor(content_imageO, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
# 2️⃣ Initialize and Load Style Model
print("🧠 Loading style model...")
style_model = TransformerNet()
state_dict = torch.load('models/candy.pth', map_location='cpu', weights_only=True)
# Remove unnecessary batch norm keys to avoid version mismatch
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
# Load weights and set model to evaluation mode
style_model.load_state_dict(state_dict)
style_model.to('cpu')
style_model.eval()
print("✅ Model loaded successfully!")
# 3️⃣ Preprocess Input Image
content_transform = transforms.Compose([
    transforms.ToTensor(),                      # Convert to Tensor
    transforms.Lambda(lambda x: x.mul(255))  ]) # Scale to 0-255
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to('cpu')  # Add batch dimension
# 4️⃣ Apply Style Transfer
print("🎨 Applying style transfer...")
with torch.no_grad():
    output = style_model(content_image).cpu()
# 5️⃣ Postprocess Output Image
img = output[0].clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
# 6️⃣ Display and Save Results
cv2.imshow("Unit13_1 | Original Image", content_imageO)
cv2.imshow("Unit13_1 | Styled Image", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
