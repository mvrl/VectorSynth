# VectorSynth: Fine-Grained Satellite Image Synthesis with Structured Semantics

<div align="center">
<img src="imgs/logo.png" width="1000">

<-- [![arXiv](https://img.shields.io/badge/arXiv-2404.06637-red?style=flat&label=arXiv)](https://arxiv.org/pdf/2511.07744) -->
<!-- [![Project Page](https://img.shields.io/badge/Project-Website-green)](https://example.com) -->
<!-- [![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow?style=flat&logo=hug)](https://huggingface.co/MVRL/VectorSynth) -->

[Daniel Cher*](https://dcher95.github.io/),
[Brian Wei*](),
[Srikumar Sastry](https://sites.wustl.edu/srikumarsastry/),
[Nathan Jacobs](https://jacobsn.github.io/)

(*Corresponding Author)
</div>

This repository is the official implementation of VectorSynth. VectorSynth is a suite of models for synthesizing satellite images with global style and text-driven layout control.

![](imgs/teaser.png)

## ⏭️ Next
- [ ] Release PyTorch `ckpt` files for all models
- [ ] Release dataset

## 🧑‍💻 Setup and Training

Create a conda environment:

```bash
conda env create -f environment.yaml
conda activate vectorsynth
```

See [train.md](scripts/train.md) for training details.

## 📊 Dataset Generation

See [dataset.md](scripts/data/dataset.md) for detailed instructions on generating the dataset(s) from OpenStreetMap data.

## 🌏 Inference

See [inference.md](scripts/inference.md) for a complete inference example.

**Requirements:**
- Trained ControlNet checkpoint (diffusers format)
- Render encoder checkpoint

We use 🤗 Diffusers for inference. VectorSynth requires special preprocessing through our render encoder to convert vector map embeddings into control signals for the diffusion model.

## 📑 Citation

```bibtex
@inproceedings{cher2025vectorsynth,
  title={VectorSynth: Fine-Grained Satellite Image Synthesis with Structured Semantics},
  author={Cher, Daniel and Wei, Brian and Sastry, Srikumar and Jacobs, Nathan},
  year={2025},
  eprint={arXiv:2511.07744},
  note={arXiv preprint}
}
```

## 🔍 Additional Links

Check out our lab website for other interesting works on geospatial understanding and mapping:
* Multi-Modal Vision Research Lab (MVRL) - [Link](https://mvrl.cse.wustl.edu/)
* Related Works from MVRL - [Link](https://mvrl.cse.wustl.edu/publications/)
* See our previous work - [Link](https://github.com/mvrl/GeoSynth)
