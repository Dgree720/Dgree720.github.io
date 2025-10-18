#13Unit13_2torch2.py
import re
import cv2
import torch
from torchvision import transforms
from transformer_net import TransformerNet
models = ["candy.pth", "mosaic.pth", "rain_princess.pth", "udnie.pth"]
outs=[]
content_image = cv2.imread('pic/IMG_2997.png')
content_imageO = cv2.resize(content_image, (0, 0), None, 0.5, 0.5)
content_image = cv2.cvtColor(content_imageO, cv2.COLOR_BGR2RGB)
content_transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.mul(255))])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to('cpu')
style_model = TransformerNet()

for i in range(0,len(models)):
    state_dict = torch.load('models/'+models[i], weights_only=True)
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to('cpu')
    style_model.eval()
    with torch.no_grad():
        output = style_model(content_image).cpu()
    img = output[0].clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    outs.append(img)
imgStack = cv2.hconcat(outs[0:4])
cv2.imshow("Unit13_2 | StudentID | Original images", content_imageO)
cv2.imshow("Unit13_2 | StudentID | Styled images", imgStack)
cv2.waitKey(0)