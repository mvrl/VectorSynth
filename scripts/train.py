from ..ControlNet.share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from vectordataset import VecSatNetDataset, collate_fn
from ..ControlNet.cldm.logger import ImageLogger
from ..ControlNet.cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Configs
resume_training = False
resume_checkpoint_path = '' # '/VectorSynth/checkpoint/vectorsynth-clip/vectorsynth-clip.ckpt'
initial_path = "/VectorSynth/ControlNet/models/control_sd21_ini.ckpt" 
batch_size = 8
logger_freq = 150
total_epochs = 6
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
render_encoder = "1d"

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model("/VectorSynth/scripts/models/cldm_v21.yaml").cpu()# './models/cldm_v21.yaml'
if not resume_training:
    model.load_state_dict(load_state_dict(initial_path, location="cpu"), strict=False)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
model.render_encoder = render_encoder

checkpoint = ModelCheckpoint(
    dirpath= os.path.join("/VectorSynth/checkpoint", "vectorsynth-clip"),
    filename="vectorsynth-clip",
    every_n_train_steps=150,
    verbose=True 
)

# Misc
dataset = VecSatNetDataset(
        csv_path = '/VectorSynth/data/metadata/final_points.csv', 
        data_dir = '/VectorSynth/data/', 
        embedding_tensor = torch.load('/VectorSynth/data/embeddings/clip.pt', weights_only=False), 
        tag_list="/VectorSynth/data/metadata/taglist_vocab.pt",
        captions = '/VectorSynth/data/metadata/captions.json',
        mode='train')

dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
logger = ImageLogger(batch_frequency=logger_freq)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=32,
    max_epochs=total_epochs,
    callbacks=[logger, checkpoint],
    resume_from_checkpoint=resume_checkpoint_path if resume_training else None
)

# Train!
trainer.fit(model, dataloader)

# export PYTHONPATH=/VectorSynth/ControlNet:$PYTHONPATH
# python -m rendersynth.scripts.train