from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import json
import re

from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import time

class VecSatNetDataset(Dataset):
    def __init__(
        self,
        csv_path,
        data_dir,
        embedding_tensor,
        tag_list=None,
        captions=None,
        mode='train',
        tensor_path='pixel_tensors'
    ):
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path
        else:
            if csv_path.endswith('.csv'):
                self.df = pd.read_csv(csv_path)
            else:
                raise ValueError("csv_path must be a path to a CSV file or a pandas DataFrame")
        
        if mode=='train':
            self.df = self.df[self.df['split'].isin(['train', 'val'])]
            self.df = self.df[self.df['coverage'] >= 0.7]
        
        elif mode=='test':
            self.df = self.df[self.df['split'] == 'test']

        self.data_dir = data_dir
        self.tensor_path = os.path.join(data_dir, tensor_path)
        self.mode = mode

        if captions is not None:
            #read in the json
            with open(captions, 'r') as f:
                self.captions = json.load(f)
        else:
            self.captions = None
        
        
        self.embedding_tensor = embedding_tensor
        
        if tag_list.endswith('.json'):
            with open(tag_list, 'r') as f:
                tag_list_index = json.load(f)
        elif tag_list.endswith('.pt'):
            tag_list_index = torch.load(tag_list, weights_only=False)

        self.valid_tag_indices = set(range(len(tag_list_index)))

    def _calculate_current_coverage(self, pixel_grid):
        """Calculate current coverage (non -1 pixels)"""
        valid_pixels = (pixel_grid != -1).sum().item()
        total_pixels = pixel_grid.numel()
        return valid_pixels / total_pixels if total_pixels > 0 else 0.0

    def __len__(self):
        return len(self.df)
    
    def _load_pixel_grid(self, point_id):
        # Load pixel grid tensor from the specified path unless it is already provided in df
        if 'pixel_tensor' in self.df.columns:
            tensor_path = self.df.loc[self.df['point_id'] == point_id, 'pixel_tensor'].values[0]
            if not os.path.exists(tensor_path):
                raise FileNotFoundError(f"Pixel grid file missing: {tensor_path}")
        else:
            tensor_path = os.path.join(self.tensor_path, f'bbox_{point_id}.pt')
        # tensor_path = os.path.join(self.tensor_path, f'bbox_{point_id}.pt')
        pixel_grid = torch.load(tensor_path,  weights_only=False)

        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"Pixel grid file missing: {tensor_path}")
        return pixel_grid
    
    def _compute_pixel_embeddings(self, pixel_grid):
        emb_dim = self.embedding_tensor.shape[1]
        H, W = pixel_grid.shape
        
        # Create a mask of valid tags
        valid_mask = torch.zeros(self.embedding_tensor.shape[0], dtype=torch.bool, device=pixel_grid.device)
        valid_mask[np.array(list(self.valid_tag_indices))] = True
        
        # Flatten pixel_grid to 1D
        flat_grid = pixel_grid.flatten()
        
        # Create a mask for valid pixels
        pixel_valid_mask = valid_mask[flat_grid]
        
        # Initialize embeddings tensor (emb_dim, H*W) with zeros
        embeddings_flat = torch.zeros((emb_dim, H*W), device=pixel_grid.device)
        
        # For valid pixels, fetch embeddings from embedding_tensor
        valid_indices = flat_grid[pixel_valid_mask]
        embeddings_flat[:, pixel_valid_mask] = self.embedding_tensor[valid_indices].T
        
        # Reshape to (emb_dim, H, W)
        embeddings_grid = embeddings_flat.reshape(emb_dim, H, W)
        
        # Permute to (H, W, emb_dim)
        embeddings_grid = embeddings_grid.permute(1, 2, 0)
        
        return embeddings_grid

    def _get_embeddings(self, point_id):
        
        pixel_grid = self._load_pixel_grid(point_id)
        pix_embeddings = self._compute_pixel_embeddings(pixel_grid)

        return pix_embeddings, 1
    
    def _read_json(self, idx):
        item = self.df.iloc[idx]
        point_id = item["point_id"]

        caption = ""
        if self.captions is not None:
            # Get caption from the JSON file
            caption = self.captions.get(f'patch_{point_id}', "")

        # Build prompt randomly 
        rnd_int = np.random.randint(0, 1)
        if self.mode != 'test':
            if rnd_int == 0:
                prompt = "An aerial image"
            else:
                rnd_int = np.random.randint(0, 1)
                if rnd_int == 0:
                    prompt = f'An aerial image of {item["NAME_2"]}, {item["NAME_1"]}.'
                else:
                    prompt = f'An aerial image of {caption.replace("The image shows", "")}'
        else:
            prompt = f'An aerial image of {item["NAME_2"]}, {item["NAME_1"]}. {caption}'
        
        # Get per pixel embeddings tensor (768, 512, 512)
        pix_embeddings, flag = self._get_embeddings(point_id)

        # Load satellite image (target)
        target_path = os.path.join(self.data_dir, f'sat_images/patch_{point_id}.jpeg')
        target = cv2.imread(target_path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = torch.tensor((target.astype(np.float32) / 127.5) - 1.0)

        # Load osm image (source)
        source_path = os.path.join(self.data_dir, f'osm_images/patch_{point_id}.jpeg')
        source = cv2.imread(source_path)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = (source.astype(np.float32) / 255.0)
        source = torch.from_numpy(source).permute(2, 0, 1)

        return dict(jpg=target, txt=prompt, hint=pix_embeddings, flag=flag, osm=source, point_id=point_id)
    
    def __getitem__(self, idx):
        d = self._read_json(idx)
        if d['flag']:
            return d
        else:
            # In case embeddings missing, try new random idx until found
            while not d['flag']:
                idx = np.random.randint(0, len(self.df))
                d = self._read_json(idx)
            return d
        

def collate_fn(batch):

    jpg = []
    txt = []
    flag = []
    hint = []
    osm = []

    jpg = torch.stack([item['jpg'] for item in batch], dim=0) 
    txt = [item['txt'] for item in batch]                  
    hint = torch.stack([item['hint'] for item in batch])   
    flag = torch.tensor([item['flag'] for item in batch])  
    osm = torch.stack([item['osm'] for item in batch], dim=0)  

    return dict(jpg=jpg, txt=txt, hint=hint, flag=flag, osm=osm)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = VecSatNetDataset(
        csv_path = '/VectorSynth/data/metadata/final_points.csv', 
        data_dir = '/VectorSynth/data/', 
        embedding_tensor = torch.load('/VectorSynth/data/embeddings/clip.pt', weights_only=False), 
        tag_list="/VectorSynth/data/metadata/taglist_vocab.pt",
        captions = '/VectorSynth/data/metadata/captions.json',
        mode='train')

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    for i, data in enumerate(tqdm(loader, desc="Loading batches")):
        import code; code.interact(local=locals())