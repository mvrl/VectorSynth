import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import OSMDataset
from torch.utils.data import DataLoader
import random
from typing import Optional, List, Tuple, Literal
from image_encoder import SatlasPretrainEncoder
from text_encoder import TextEncoder
from orthogonal_adamw import OrthogonalAdamW
from configs.config_e5 import config
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from utils import generate_tag_poly_pairs
import matplotlib.pyplot as plt
import io
import wandb
from PIL import Image


# This performs a typical InfoNCE loss
def contrastive_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(image_feats, text_feats.t()) * logit_scale
    labels = torch.arange(logits.size(0), device=logits.device)

    return F.cross_entropy(logits, labels), logits


class OSMBind(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.image_encoder = SatlasPretrainEncoder(fpn=True, model_name="Aerial_SwinB_SI",
                                                   out_dim=768, num_extra_fpn_layers=4)
        taglist_vocab = torch.load(kwargs.get("taglist_vocab_path"))
        tag_vocab_inverted = torch.load(kwargs.get("tag_vocab_path")) # str -> int
        tag_vocab = {v: k for k, v in tag_vocab_inverted.items()} # int -> str
        self.text_encoder = TextEncoder(taglist_vocab, tag_vocab, 
                                        model_name=kwargs.get("text_backbone"))
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # softer scale for misaligned encoders
    
        self.batch_size = kwargs.get("batch_size")
        self.num_workers = kwargs.get("num_workers")
        self.lr = kwargs.get("lr", 1e-4)
        self.num_samples = kwargs.get("num_samples") # number of OSM classes sampled
        self.ort_grad = kwargs.get("ort_grad")
    
    def forward(self, sat_img: torch.Tensor, pixel_tensor: torch.Tensor):
        full_image_feats = self.image_encoder(sat_img)  # [B, D, H', W']
        sampled_tag_tensor, image_poly_feats = generate_tag_poly_pairs(pixel_tensor, full_image_feats, K=self.num_samples) # [K], [K, D]
        text_sampled_feats = self.text_encoder(sampled_tag_tensor)  # [K, D]
        
        return image_poly_feats, text_sampled_feats # [K, D], [K, D]
    
    def shared_step(self, batch):
        sat_img, pixel_tensor = batch
        image_poly_feats, text_sampled_feats = self(sat_img, pixel_tensor)  # [K, D], [K, D]

        # contrastive loss for whole batch
        image_feats_norm = F.normalize(image_poly_feats, dim=1)
        text_feats_norm = F.normalize(text_sampled_feats, dim=1)
        logit_scale = self.logit_scale.exp()
        loss, logits = contrastive_loss(image_feats_norm, text_feats_norm,
                                            logit_scale=logit_scale)
        return loss, logits
    
    def log_similarity_matrix(self, logits):
        mat = logits.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6,6))
        cax = ax.matshow(mat, cmap="viridis")
        fig.colorbar(cax)
        ax.set_xlabel("Text samples")
        ax.set_ylabel("Image samples")
        ax.set_title("Similarity Matrix")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # ✅ Fix: Convert buffer to PIL Image
        image = Image.open(buf)

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                "similarity_matrix": wandb.Image(image),
                "global_step": self.global_step
            })

    def training_step(self, batch, batch_idx):
        loss, logits = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('temperature', self.logit_scale.exp().item(), prog_bar=True, on_epoch=True)
        if self.global_step % 500 == 0:
            self.log_similarity_matrix(logits)
        # Log histogram of similarity scores every step
        if self.logger and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({"logits_hist": wandb.Histogram(logits.detach().cpu().numpy())})

        # Optionally log mean and max of logits for monitoring
        self.log("logits_mean", logits.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("logits_max", logits.max(), on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        min_log_scale = np.log(1 / 1.0)    
        max_log_scale = np.log(1 / 0.01)   
        self.logit_scale.data.clamp_(min_log_scale, max_log_scale)

    def on_after_backward(self):
        if self.global_rank == 0 and self.current_epoch == 0:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"⚠️ Unused parameter: {name}")

    def validation_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("This model was initialized without a training dataset.")
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          persistent_workers=False)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("This model was initialized without a validation dataset.")
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          persistent_workers=False)

    def configure_optimizers(self):
        params = self.parameters()
        if self.ort_grad:
            self.optim = OrthogonalAdamW(
                params,
                lr=self.lr,
                betas=(0.9, 0.98),
                beta_ort=0.9,
                eps=1e-6,
                weight_decay=0.01
            )
        else:
            self.optim = torch.optim.AdamW(
                params,
                lr=self.lr,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optim,
            T_0=20
        )

        return [self.optim], [self.scheduler]
    
    def sim_map_inf(self, sat_image: torch.Tensor, raw_text: str) -> torch.Tensor:
        """
        Args:
            sat_image: [1, 3, 512, 512] tensor (already normalized)
            raw_text: str, e.g., "building"

        Returns:
            sim_map: [512, 512] similarity map between image and text embedding
        """
        assert sat_image.dim() == 4 and sat_image.size(0) == 1, "Expected input of shape [1, 3, H, W]"

        # Step 1: Extract spatial features
        with torch.no_grad():
            # image features
            feat_map = self.image_encoder(sat_image)  # [1, D, H', W']
            feat_map = feat_map.squeeze(0)            # [D, H', W']
            feat_map_upsampled = F.interpolate(feat_map.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)  # [D, 512, 512]
            feat_map_upsampled = F.normalize(feat_map_upsampled, dim=0)  # [D, 512, 512]

            # text features
            text_feat = self.text_encoder.encode_raw_text(raw_text)

            # cosine sim
            text_feat = F.normalize(text_feat, dim=0)
            feat_map_upsampled = F.normalize(feat_map_upsampled, dim=0)
            sim_map = torch.einsum('chw,c->hw', feat_map_upsampled, text_feat)  # [512, 512]

        return sim_map

    def encode_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            return self.text_encoder.encode_raw_text(text)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.image_encoder(image)

def seed_everything(seed=42):
    """
    seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__=='__main__':
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_warn_always(False)

    seed_everything()
    train_dataset = OSMDataset(metadata_path = config.train_csv,
                               image_dir=config.sat_img_dir, 
                               pixel_tensor_dir=config.pixel_tensors_dir,
                               mode='train')
    val_dataset = OSMDataset(metadata_path = config.val_csv,
                               image_dir=config.sat_img_dir, 
                               pixel_tensor_dir=config.pixel_tensors_dir,
                               mode='val')
    
    kwargs = {
        'batch_size':config.batch_size, 
        'num_workers': config.num_workers,
        'num_samples': config.num_contrastive_samples,
        'ort_grad': config.ort_grad,
        'lr': config.lr,
        'taglist_vocab_path': config.taglist_vocab_path,
        'tag_vocab_path': config.tag_vocab_path,
        'text_backbone': config.text_backbone
    }
    
    model = OSMBind(train_dataset, val_dataset, **kwargs)
    torch.cuda.empty_cache()

    checkpoint_path = ''  # specify path if resuming from checkpoint
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.save_dir,
        filename=config.filename,
        mode='min',
        save_top_k=1,
        every_n_epochs=1
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min'
    )

    logger = WandbLogger(project="osmclip", 
                         name=f"{config.experiment_name}")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config.devices,
        strategy='ddp',
        max_epochs=config.max_epochs,
        num_nodes=1,
        callbacks=[checkpoint, early_stop_callback],
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=5, 
        logger = logger #wandb logger
        )
    
    trainer.fit(model)