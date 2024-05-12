import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision.transforms import v2

class ModelSegmentation:
    def __init__(self, model_path, device, encoder_name="mobilenet_v2", encoder_weigths="imagenet", in_channels=3, classes=1):
        self.model_path=model_path
        self.device = device
        self.model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weigths,    
            in_channels=in_channels,              
            classes=classes
        ).to(self.device)
        self.load()
    
    def load(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, start_image):
        img = cv2.cvtColor(start_image, cv2.COLOR_BGR2RGB) / 255
        transform = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        img = transform(img)
        img = torch.tensor(img, dtype=torch.float).permute(2,0,1)

        pred = self.model(img.to(self.device, dtype=torch.float32).unsqueeze(0))
        pred = torch.sigmoid(pred).squeeze(0).squeeze(0)
        pred = pred.data.cpu().numpy()
        pred = np.where(pred<0.5, 1, 0).astype(np.int16)

        return pred