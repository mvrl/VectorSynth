from easydict import EasyDict as edict

config = edict()

# params
config.batch_size = 6
config.num_contrastive_samples = 128
config.ort_grad = True
config.lr = 1e-4
config.text_backbone = 'clip' # 'clip' or 'bert' or 'e5' or 'gritlm'

# Data
config.sat_img_dir = '/data/b.j.wei/rendersynth/osm_clip/data/sat_images/'
config.pixel_tensors_dir = '/data/b.j.wei/rendersynth/osm_clip/data/pixel_tensors/'
config.final_csv = '/data/b.j.wei/rendersynth/osm_clip/data/metadata/final_points.csv' # all points combined csv
config.train_csv = '/data/b.j.wei/rendersynth/osm_clip/data/metadata/train_points.csv'
config.val_csv = '/data/b.j.wei/rendersynth/osm_clip/data/metadata/val_points.csv'
config.test_csv = '/data/b.j.wei/rendersynth/osm_clip/data/metadata/test_points.csv'

# Lookups
config.taglist_vocab_path = '/data/b.j.wei/rendersynth/osm_clip/data/metadata/taglist_vocab.pt'
config.tag_vocab_path = '/data/b.j.wei/rendersynth/osm_clip/data/metadata/tag_vocab.pt'

# training, validation, and testing
config.num_workers = 16
config.accumulate_grad_batches = 8
config.max_epochs = 100
config.devices = 2
config.val_check_interval = 0.5

# logging
config.save_dir = 'checkpoints/osmclip_clip'
config.experiment_name = "osmclip_config_clip"
config.filename = f'{config.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}'

# checkpoint (best one)
config.checkpoint_path = '/data/b.j.wei/rendersynth/osm_clip/checkpoints/osmclip_config_default-epoch=78-val_loss=3.17.ckpt'

 