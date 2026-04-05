import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

class ReIDExtractor:
    def __init__(self, device=None):
        """
        Initializes a ResNet-18 model for feature extraction.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load ResNet-18 and remove the final classification layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Standard Re-ID preprocessing: Resize to 128x64 and Normalize
        self.transform = T.Compose([
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, frame, bboxes):
        """
        Extracts features for a list of bounding boxes in a frame.
        bboxes: List of [x1, y1, x2, y2]
        Returns: Numpy array of shape (N, 512)
        """
        if not bboxes:
            return np.array([])

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Clip coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                # Fallback for invalid crops
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops.append(self.transform(crop_pil))

        # Batch process crops for speed
        batch = torch.stack(crops).to(self.device)
        features = self.model(batch)
        features = features.view(features.size(0), -1)
        
        # L2 Normalization (Crucial for Cosine Distance)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()