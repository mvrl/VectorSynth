import torch
import os
import json
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

class OSMDataset(Dataset):
    def __init__(self, 
                 metadata_path: str,
                 image_dir: str, 
                 pixel_tensor_dir: str, # pixel -> taglist idx
                 mode: str ='train'): # or 'val', 'test'
        
        self.image_dir = image_dir
        self.pixel_tensors = pixel_tensor_dir
        self.metadata = pd.read_csv(metadata_path)
        self.mode = mode
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        point_id = row['point_id']

        sat_img = Image.open(os.path.join(self.image_dir, f"patch_{point_id}.jpeg")).convert("RGB")
        pixel_tensor = torch.load(os.path.join(self.pixel_tensors, f"bbox_{point_id}.pt")) 

        # simulataneous transformations for pixel alignment (assume both 512 x 512, no resizing done)
        if self.mode == 'train':
            if torch.rand(1) < 0.3:
                sat_img = F.hflip(sat_img)
                pixel_tensor = F.hflip(pixel_tensor)
            if torch.rand(1) < 0.3:
                sat_img = F.vflip(sat_img)
                pixel_tensor = F.vflip(pixel_tensor)

        sat_img = F.to_tensor(sat_img)  # normalizes RGB to [0, 1]

        return sat_img, pixel_tensor