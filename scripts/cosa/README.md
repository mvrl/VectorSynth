# COSA: Contrastive OSM-Satellite Alignment

COSA is a contrastive learning model that aligns OpenStreetMap (OSM) text with satellite imagery features. It learns a shared embedding space where semantically similar OSM tags and satellite image regions are close together.

## Model Overview

COSA uses a dual-encoder architecture:
- **Image Encoder**: Satlas-pretrained Swin-B with FPN 
- **Text Encoder**: CLIP ViT-L/14 text encoder 

The model is trained with InfoNCE contrastive loss to align per-polygon image features with their corresponding OSM text embeddings.

## Files

```
cosa/
├── cosa.ckpt              # Trained model checkpoint
├── model.py               # OSMBind model class
├── text_encoder.py        # Text encoder implementations
├── compute_embeddings.py  # Script to generate tag embeddings
└── README.md              # This file
```

## Usage

### Generate Tag Embeddings

To create COSA embeddings for your tag vocabulary:

```bash
python compute_embeddings.py \
    --taglist_path /path/to/taglist_vocab.pt \
    --tag_vocab_path /path/to/tag_vocab.pt \
    --output_path cosa-embeddings.pt \
    --encoder_type osm-clip \
    --checkpoint_path cosa.ckpt
```

### Encode Custom Text

```python
import torch
from model import OSMBind

# Load model
checkpoint = torch.load("cosa.ckpt", map_location="cuda")
model = OSMBind(
    taglist_path="/path/to/taglist_vocab.pt",
    tagvocab_path="/path/to/tag_vocab.pt",
    text_backbone="clip"
)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval().cuda()

# Encode text
embedding = model.encode_text("building residential")

# Encode batch
embeddings = model.text_encoder.encode_batch(["building", "highway residential"])
```

### Create Hint Tensor for VectorSynth

```python
import torch

# Load pre-computed embeddings
embeddings = torch.load("cosa-embeddings.pt")

# Load pixel tensor (from dataset)
pixel_tensor = torch.load("pixel_tensors/bbox_123.pt")

# Create hint: look up embedding for each pixel
hint = embeddings[pixel_tensor]
```