import torch
import einops
import numpy as np
from PIL import Image
import os


def process_hint(hint_tensor, render_encoder, device):
    """
    Process hint tensor through the render encoder.
    
    Args:
        hint_tensor: Tensor of shape (H, W, embedding_dim) - e.g., (512, 512, 768)
        render_encoder: Render encoder model
        device: torch device
        
    Returns:
        control: Processed control tensor ready for ControlNet (1, 3, H, W)
    """
    hint = hint_tensor.to(device)
    hint = hint.unsqueeze(0)  # Add batch dimension: (1, H, W, embedding_dim)
    hint = einops.rearrange(hint, "b h w c -> b c h w")  # (1, embedding_dim, H, W)
    
    with torch.no_grad():
        control = render_encoder(hint).sigmoid()
    
    return control


def create_sample_hint(
    point_id,
    pixel_tensor_path,
    embedding_tensor_path,
    taglist_vocab_path,
    device='cpu'
):
    """
    Create a hint tensor from pixel grid and embeddings (mimics VecSatNetDataset).
    
    Args:
        point_id: ID of the point/bbox
        pixel_tensor_path: Path to directory containing pixel tensors (e.g., 'pixel_tags/')
        embedding_tensor_path: Path to embedding tensor file (e.g., 'osm-clip.pt')
        taglist_vocab_path: Path to taglist vocab file
        device: Device to load tensors on
        
    Returns:
        hint: Tensor of shape (H, W, embedding_dim)
    """
    
    # Load pixel grid
    pixel_grid_path = os.path.join(pixel_tensor_path, f'bbox_{point_id}.pt')
    pixel_grid = torch.load(pixel_grid_path, weights_only=False, map_location=device)
    
    # Load embeddings
    embedding_tensor = torch.load(embedding_tensor_path, weights_only=False, map_location=device)
    
    # Load taglist vocab to get valid indices
    taglist_vocab = torch.load(taglist_vocab_path, weights_only=False)
    valid_tag_indices = set(range(len(taglist_vocab)))
    
    # Compute pixel embeddings
    hint = _compute_pixel_embeddings(pixel_grid, embedding_tensor, valid_tag_indices, device)
    
    return hint


def _compute_pixel_embeddings(pixel_grid, embedding_tensor, valid_tag_indices, device):
    """
    Compute per-pixel embeddings from pixel grid (internal helper).
    """
    emb_dim = embedding_tensor.shape[1]
    H, W = pixel_grid.shape
    
    # Create a mask of valid tags
    valid_mask = torch.zeros(embedding_tensor.shape[0], dtype=torch.bool, device=device)
    valid_mask[np.array(list(valid_tag_indices))] = True
    
    # Flatten pixel_grid to 1D
    flat_grid = pixel_grid.flatten()
    
    # Create a mask for valid pixels
    pixel_valid_mask = valid_mask[flat_grid]
    
    # Initialize embeddings tensor (emb_dim, H*W) with zeros
    embeddings_flat = torch.zeros((emb_dim, H*W), device=device)
    
    # For valid pixels, fetch embeddings from embedding_tensor
    valid_indices = flat_grid[pixel_valid_mask]
    embeddings_flat[:, pixel_valid_mask] = embedding_tensor[valid_indices].T
    
    # Reshape to (emb_dim, H, W)
    embeddings_grid = embeddings_flat.reshape(emb_dim, H, W)
    
    # Permute to (H, W, emb_dim)
    embeddings_grid = embeddings_grid.permute(1, 2, 0)
    
    return embeddings_grid

def find_tag_index(taglist, target):
    """
    Find the index of a tuple in taglist:
    - If (target,) exists, return its index.
    - Otherwise, return the index of the tuple containing target
      with the fewest extra numbers.
    - If no tuple contains target, return None.
    """
    # Try exact match first
    try:
        return taglist.index((target,))
    except ValueError:
        # Find all tuples that contain target
        candidates = [(i, t) for i, t in enumerate(taglist) if target in t]
        
        if candidates:
            # Pick the one with the shortest length
            idx, _ = min(candidates, key=lambda x: len(x[1]))
            return idx
        else:
            return None