# Train VectorSynth

## 🧑‍💻 Setting up environment

Create a conda environment:

```bash
conda env create -f environment.yaml
conda activate vectorsynth
```

## 🗺️ Generating Vector Data

See `scripts/data/README.md` for detailed notes on the data generation pipeline.

## 🔥 Training

Setup all parameters of interest in `train.py`, then run:

```bash
python train.py
```