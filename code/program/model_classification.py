import torch
from torchvision import transforms
from PIL import Image


class ModelClassification:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.load()
        self.size = None
        
    def load(self):
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()

    def set_size(self, width, height):
        self.size = 380 # width if width < height else height
    
    def predict(self, start_image):
        val_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = val_transforms(Image.fromarray(start_image))

        with torch.set_grad_enabled(False):
            preds = self.model(img.unsqueeze(0).to(self.device))
            result = torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy()

        return float(result[0])