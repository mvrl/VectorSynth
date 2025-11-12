import torch
import torch.nn as nn
import pytorch_lightning as pl
from satlaspretrain_models import Weights
import torch.nn.functional as F

# class DINOv2FeatUpEncoder(pl.LightningModule):
#     raise NotImplementedError


class SatlasPretrainEncoder(pl.LightningModule):
    def __init__(self, 
                 fpn: bool = True, 
                 model_name: str = "Aerial_SwinB_SI", 
                 out_dim: int = 768, # must match w/ text encoder
                 num_extra_fpn_layers: int = 4): # layers 1 - 4
        super().__init__()
        self.weights_manager = Weights()
        self.backbone = self.weights_manager.get_pretrained_model(model_name, fpn=fpn) 
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.unfreeze_last_layers(n=2)
        
        # self.print_backbone_trainability()
        self.num_extra_fpn_layers = num_extra_fpn_layers
        self.out_dim = out_dim
        self.in_chans = 128 * (num_extra_fpn_layers+1)
        self.projection_head = nn.Sequential(
            nn.Conv2d(self.in_chans, self.out_dim, kernel_size=1), # 1x1 convs
            nn.ReLU(),
            nn.BatchNorm2d(self.out_dim, self.out_dim),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_dim),
        )

    def concat_fpn(self, fpn_features):
        # upsampled = [fpn_features[1]]
        upsampled=[]
        for f in fpn_features[0:self.num_extra_fpn_layers + 1]:
            up = F.interpolate(f, size=(128, 128), mode='bilinear', align_corners=False)
            upsampled.append(up)
        return torch.cat(upsampled, dim=1)
    
    def unfreeze_last_layers(self, n: int = 0):
        """
        Unfreezes the last `n` layers of the backbone (deepest layers).
        """
        # Collect all submodules of the backbone
        layers = list(self.backbone.children())
        if n > len(layers):
            n = len(layers)  # don't go out of bounds

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def print_backbone_trainability(self):
        print("\n=== Backbone Parameter Trainability Report ===")
        total_params = 0
        trainable_params = 0

        for name, param in self.backbone.named_parameters():
            status = "Trainable" if param.requires_grad else "Frozen"
            print(f"{status:<10} | {name}")
            total_params += 1
            if param.requires_grad:
                trainable_params += 1

        print("\n=== Summary ===")
        print(f"Total parameters:   {total_params}")
        print(f"Trainable:          {trainable_params}")
        print(f"Frozen:             {total_params - trainable_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fpn_features = self.backbone(x)
        concat_feats = self.concat_fpn(fpn_features)
        img_embed = self.projection_head(concat_feats)
        # img_embed = fpn_features[1] # [128, 128, 128] for SwinB backbone 
        return img_embed