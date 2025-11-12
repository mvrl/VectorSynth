import torch
import random
from typing import Tuple
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

# 
def generate_tag_poly_pairs(pixel_tensor: torch.Tensor, 
                            sat_image: torch.Tensor, 
                            K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate K (tag_id, pooled satellite feature) pairs using downsampled pixel-to-feature alignment.
    
    Args:
        pixel_tensor: [B, 512, 512] LongTensor of taglist IDs (high-res)
        sat_image:    [B, D, 128, 128] FloatTensor of satellite features (low-res)
        K:            Number of samples to generate
    
    Returns:
        sampled_tag_tensor: [K] LongTensor of sampled taglist IDs
        avg_embeddings:     [K, D] FloatTensor of pooled satellite embeddings (with gradients)
    """
    B, D, H_lr, W_lr = sat_image.shape

    # Map from tag_id to image indices where it occurs
    tag_to_images = {}
    for b in range(B):
        unique_tags = torch.unique(pixel_tensor[b])
        for tag in unique_tags.tolist():
            if tag == -1: #ignore case where taglist is -1
                continue
            tag_to_images.setdefault(tag, []).append(b)

    all_tags = list(tag_to_images.keys())
    if len(all_tags) < K:
        print(f"[Info] Only {len(all_tags)} unique taglist IDs found, but K={K} was requested. Setting K = {len(all_tags)}.")
        K = len(all_tags)

    sampled_tags = random.sample(all_tags, K)
    sampled_tag_tensor = torch.tensor(sampled_tags, dtype=torch.long, device=pixel_tensor.device)

    avg_embeddings = []

    for tag in sampled_tags:
        img_idx = random.choice(tag_to_images[tag])

        # High-res binary mask for this tag
        mask_hr = (pixel_tensor[img_idx] == tag).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]

        # Downsample mask to match sat_image resolution
        mask_lr = F.interpolate(mask_hr, size=(H_lr, W_lr), mode='bilinear', align_corners=False).squeeze()  # [128, 128]

        # Threshold to get binary mask (keep low to ensure we retain thing polygons like highways)
        binary_mask = (mask_lr > 0.2)  # BoolTensor of shape [128, 128]
        mask_reshaped = binary_mask.unsqueeze(0)  # [1, 128, 128]

        feat = sat_image[img_idx]  # [D, 128, 128]

        if binary_mask.sum() == 0:
            # Soft fallback: average over the whole image
            pooled = feat.view(D, -1).mean(dim=1)  # [D]
        else:
            selected_pixels = torch.masked_select(feat, mask_reshaped).view(D, -1)  # [D, N] where N is number of selected pixels
            pooled = selected_pixels.mean(dim=1)  # [D] avg pooling among pixels

        avg_embeddings.append(pooled)

    avg_embeddings = torch.stack(avg_embeddings, dim=0)  # [K, D]
    return sampled_tag_tensor, avg_embeddings


# for evaluation
def prepare_ground_truth_masks(pixel_tensor: torch.Tensor,
                                taglist_vocab: dict = None,
                                tag_vocab: dict = None) -> dict:
    """
    For a given pixel tensor, extract binary masks for each taglist index (except -1).

    Args:
        pixel_tensor: [512, 512] LongTensor
        taglist_vocab: Optional dict mapping taglist index to list of tag_ids
        tag_vocab: Optional dict mapping tag_id to tag string

    Returns:
        taglist_to_mask: Dict[int, torch.BoolTensor] → binary mask per taglist index
        taglist_to_text: Dict[int, str] → corresponding taglist text (if vocab is given)
    """
    taglist_to_mask = {}
    taglist_to_text = {}
    assert pixel_tensor.shape == (512, 512)

    unique_ids = torch.unique(pixel_tensor)
    for taglist_id in unique_ids.tolist():
        if taglist_id == -1:
            continue
        binary_mask = (pixel_tensor == taglist_id)  # [512, 512], bool tensor
        taglist_to_mask[taglist_id] = binary_mask

        if taglist_vocab is not None and tag_vocab is not None:
            tag_ids = taglist_vocab[taglist_id]
            tag_strings = [tag_vocab[tid].lower().replace("=", " ") for tid in tag_ids]
            taglist_to_text[taglist_id] = " ".join(tag_strings)

    return taglist_to_mask, taglist_to_text


def compute_ap(saliency_map: torch.Tensor, ground_truth_mask: torch.Tensor) -> float:
    """
    Compute average precision (AP) between saliency map and binary ground truth mask.

    Args:
        saliency_map: [512, 512] float tensor (unnormalized scores)
        ground_truth_mask: [512, 512] bool or int tensor (binary mask)

    Returns:
        ap: float – average precision score
    """
    # Flatten both
    y_score = saliency_map.flatten().cpu().numpy()
    y_true = ground_truth_mask.flatten().cpu().numpy().astype(int)

    if y_true.sum() == 0:
        return float('nan')  # Avoid scoring empty masks

    ap = average_precision_score(y_true, y_score)
    return ap


def visualize_similarity(sim_map):
    sim_np = sim_map.detach().cpu().numpy()
    sim_np = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-6)

    plt.figure(figsize=(6, 6))
    plt.imshow(sim_np, cmap='hot')
    plt.colorbar()
    plt.title("OSM-CLIP Similarity Map")
    plt.axis('off')
    plt.show()