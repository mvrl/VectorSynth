import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualRenderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class RenderEncoder(nn.Module):
    def __init__(self, encoder_type="1d", in_channels=768, out_channels=3):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "1d":
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        elif encoder_type == "residual":
            self.model = ResidualBlockRender(in_channels, out_channels)

        elif encoder_type == "expressive":
            mid_channels = 256
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, mid_channels),
                nn.SiLU(),
                ResidualRenderBlock(mid_channels),
                ResidualRenderBlock(mid_channels),
                ResidualRenderBlock(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        else:
            raise ValueError(f"Unknown encoder_type '{encoder_type}'. Use '1d', 'residual', or 'expressive'.")

    def forward(self, x):
        return self.model(x)

class ResidualBlockRender(nn.Module):
    def __init__(self, in_channels=768, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.out = nn.Sigmoid()

        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        h = self.relu1(self.conv1(x))
        h = self.relu2(self.conv2(h))
        h = self.conv3(h)
        h = h + residual
        return self.out(h)

def load_render_encoder(checkpoint_path, device='cpu'):
    """Load standalone RenderEncoder from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['model_config']
    model = RenderEncoder(
        encoder_type=config['encoder_type'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded RenderEncoder: {config}")
    return model