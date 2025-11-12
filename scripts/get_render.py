import os
import torch
from ..ControlNet.cldm.model import create_model, load_state_dict

def extract_render_encoder(model_path, config_path, output_path):
    """
    Extract RenderEncoder from ControlNet checkpoint and save it
    """
    device = torch.device("cpu")
    model = create_model(config_path).to(device)
    model.load_state_dict(load_state_dict(model_path, location="cpu"), strict=False)
    
    # Extract the RenderEncoder
    render_encoder = model.render
    render_encoder.eval()
    
    # Save both the entire model AND the config/state_dict for portability
    torch.save({
        'model': render_encoder,  # Full model (needs cldm import)
        'state_dict': render_encoder.state_dict(),  # State dict only
        'config': {
            'encoder_type': getattr(render_encoder, 'encoder_type', '1d'),
            'in_channels': 768,
            'out_channels': 3
        }
    }, output_path)
    
    print(f"Saved RenderEncoder to {output_path}")
    return render_encoder

def test_extracted_model(checkpoint_path, test_input_shape=(1, 768, 512, 512)):
    """
    Test that the extracted RenderEncoder works correctly
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    render_encoder = checkpoint['model']  # Extract model from dict
    render_encoder.eval()
    
    test_input = torch.randn(test_input_shape).to(device)
    
    # Test forward pass
    with torch.no_grad():
        output = render_encoder(test_input)
    
    print(f"\nTest successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    return output

if __name__ == "__main__":
    # Extract RenderEncoder from your trained model
    extract_render_encoder(
        model_path="/VectorSynth/checkpoint/vectorsynth-clip/vectorsynth-clip.ckpt",
        config_path="/VectorSynth/scripts/models/cldm_v21.yaml",
        output_path="/VectorSynth/scripts/models/dump/clip-render_encoder.pth"
    )
    
    # Test the extracted model
    test_extracted_model("/VectorSynth/scripts/models/dump/clip-render_encoder.pth")

# python -m VectorSynth.scripts.get_render