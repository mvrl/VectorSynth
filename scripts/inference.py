import torch
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
from utils import process_hint, create_sample_hint

# =============================================================================
# Configuration
# =============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Choose model: "MVRL/VectorSynth" (CLIP) or "MVRL/VectorSynth-COSA" (COSA)
MODEL_REPO = "MVRL/VectorSynth"
RENDER_ENCODER_FILE = "render_encoder/clip-render_encoder.pth"

# For COSA variant:
# MODEL_REPO = "MVRL/VectorSynth-COSA"
# RENDER_ENCODER_FILE = "render_encoder/cosa-render_encoder.pth"


# =============================================================================
# Load Models
# =============================================================================
class RenderEncoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


print(f"Loading from {MODEL_REPO}...")

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_REPO, torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Load RenderEncoder
render_path = hf_hub_download(MODEL_REPO, RENDER_ENCODER_FILE)
ckpt = torch.load(render_path, map_location=device, weights_only=False)
render_encoder = RenderEncoder(
    in_channels=ckpt['config']['in_channels'],
    out_channels=ckpt['config']['out_channels']
)
render_encoder.load_state_dict(ckpt['state_dict'])
render_encoder = render_encoder.to(device).eval()


# =============================================================================
# Generate
# =============================================================================

# For creating hint tensors: see data/dataset.md
# For COSA version: see cosa/README.md

# Option 1: Load pre-saved hint
hint = torch.load("path/to/hint.pt", map_location=device, weights_only=False)

# Option 2: Create hint from data (see data/dataset.md for generating these files)
# hint = create_sample_hint(
#     point_id="123",
#     pixel_tensor_path="path/to/pixel_tensors/",
#     embedding_tensor_path="path/to/embeddings.pt",  # CLIP or COSA embeddings
#     taglist_vocab_path="path/to/taglist_vocab.pt",
#     device=device
# )

control = process_hint(hint, render_encoder, device)

output = pipe(
    prompt="An aerial image of a city neighborhood",
    image=control,
    num_inference_steps=40,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=torch.manual_seed(42)
)

output.images[0].save("generated_satellite.png")
print("Saved: generated_satellite.png")
