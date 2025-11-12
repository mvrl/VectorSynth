import torch
import os
from utils import create_sample_hint

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths (update these to your data paths)
data_dir = './data/'
pixel_tensor_path = os.path.join(data_dir, 'pixel_tags')
embedding_tensor_path = os.path.join(data_dir, 'embeddings/clip.pt')
taglist_vocab_path = os.path.join(data_dir, 'metadata/taglist_vocab.pt')
output_dir = os.path.join(data_dir, 'examples')

# Example point IDs (update with actual point IDs from your dataset)
example_point_ids = [
    "your_point_id_1",
    "your_point_id_2",
    "your_point_id_3",
]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Generate and save hints for each example
for i, point_id in enumerate(example_point_ids):
    try:
        print(f"\nProcessing point_id: {point_id}")
        
        # Create hint tensor
        hint = create_sample_hint(
            point_id=point_id,
            pixel_tensor_path=pixel_tensor_path,
            embedding_tensor_path=embedding_tensor_path,
            taglist_vocab_path=taglist_vocab_path,
            device=device
        )
        
        # Save hint
        sample_dir = os.path.join(output_dir, f'sample_{i}')
        os.makedirs(sample_dir, exist_ok=True)
        
        hint_path = os.path.join(sample_dir, 'hint.pt')
        torch.save(hint, hint_path)
        
        print(f"✅ Saved hint to {hint_path}")
        print(f"   Shape: {hint.shape}")
        
    except Exception as e:
        print(f"❌ Error processing {point_id}: {e}")

print(f"\n✅ Done! Saved {len(example_point_ids)} sample hints to {output_dir}")