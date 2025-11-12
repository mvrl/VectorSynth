import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from render import RenderEncoder
from utils import process_hint, create_sample_hint

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
render_encoder_path = "path/to/render_encoder.pth"
controlnet_path = "path/to/controlnet"

# Load render encoder from state_dict
checkpoint = torch.load(render_encoder_path, map_location=device, weights_only=False)
config = checkpoint['config']

render_encoder = RenderEncoder(
    encoder_type=config['encoder_type'],
    in_channels=config['in_channels'],
    out_channels=config['out_channels']
)

render_encoder.load_state_dict(checkpoint['model_state_dict'])
render_encoder = render_encoder.to(device).eval()

# Load diffusion pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Option 1: Load pre-saved hint tensor
hint = torch.load("path/to/hint.pt", weights_only=False, map_location=device)

# Option 2: Create hint from scratch (uncomment to use)
# hint = create_sample_hint(
#     point_id="your_point_id",
#     pixel_tensor_path="/path/to/pixel_tags",
#     embedding_tensor_path="/path/to/osm-clip.pt",
#     taglist_vocab_path="/path/to/taglist_vocab.pt",
#     device=device
# )

# Process hint through render encoder
control = process_hint(hint, render_encoder, device)

# Generate image
output = pipe(
    prompt="An aerial image of a city neighborhood",
    num_inference_steps=40,
    guidance_scale=7.5,
    image=control,
    controlnet_conditioning_scale=1.0,
    generator=torch.manual_seed(42)
)

output.images[0].save("generated_satellite.png")
print("Generated image saved!")