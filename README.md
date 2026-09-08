# SpectraLink-Net

A lightweight dual-branch network for 3D vessel segmentation that links
frequency-domain and spatial features through learnable band partitioning and
sparse cross-attention.

## Results

| Dataset | Structure | Dice | Params |
|---|---|---|---|
| MSD Task08 | Hepatic vessel | **74.47%** | 6.12M |
| TopCoW2024 | Cerebral artery | **73.33%** | 6.12M |

The same 6.12M-parameter model is used on both datasets — roughly one tenth
the size of comparable 3D segmentation networks — without per-organ
retuning of the architecture.

## Method

Vascular structures have a natural frequency hierarchy: large trunks live in
low-frequency components, fine peripheral branches in high-frequency ones.
Convolutional networks capture local detail but miss long-range continuity,
while transformers model global context at a cost that is prohibitive in 3D.

Two components address this:

**Learnable Band Partitioning** (`models/frequency_branch/band_partition.py`)
applies a 3D FFT and learns anatomy-adaptive frequency bands end to end
through differentiable soft masks, rather than fixing band boundaries by hand.

**Sparse Cross-Attention Bridge** (`models/fusion/cross_attention_sparse.py`)
establishes explicit frequency–spatial token correspondence via Top-K
relevance sampling, exploiting the sparsity of that correspondence to avoid
the cost of dense attention.

Training uses a combined Dice + BCE objective with deep supervision, plus a
clDice term (`losses/cldice_loss.py`) to preserve the topology of thin
tubular structures.

## Repository layout

```
config/          config.yaml, config_5fold.yaml
data/            preprocess.py, dataset.py, dataloader.py, crop.py, transforms.py
models/
  spectralink_net.py        top-level model
  frequency_branch/         band_partition.py, complex_block.py, frequency_branch.py
  spatial_branch/           multi_scale_conv.py, spatial_branch.py
  fusion/                   cross_attention_sparse.py, freq_guided_fusion*.py
  encoder/ decoder/ heads/  encoder_block.py, decoder.py, seg_head.py
  components/               complex_ops.py, conv_blocks.py
losses/          dice_loss.py, cldice_loss.py, combined_loss.py, deep_supervision.py
scripts/         train.py, train_5fold.py, val.py, inference.py,
                 evaluate_5fold.py, generate_5fold.py, visualization.py
utils/           checkpoint.py, logger.py, metrics.py
```

## Usage

Requires Python 3.10+, PyTorch, SimpleITK/nibabel, numpy, scipy, PyYAML.

**1. Preprocess** — resamples to uniform spacing, normalises intensities and
extracts foreground locations:

```bash
python data/preprocess.py
```

**2. Configure** — `config/config.yaml` holds the paths used for the original
runs on the cluster:

```yaml
data:
  data_dir:   '/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/data/preprocess/processed/data'
  props_dir:  '/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/data/preprocess/processed/properties'
  split_file: '/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/data/preprocess/5fold_splits.json'
  patch_size: [64, 128, 128]
  samples_per_volume: 4
  oversample_foreground: 0.7
  batch_size: 2
  num_workers: 8
```

Replace the three paths with your own. Training defaults: 250 epochs with 10
warm-up, lr 1e-3 decaying to 1e-6, weight decay 1e-5, gradient clipping 1.0,
AMP and gradient checkpointing enabled. Loss is 0.7·Dice + 0.3·BCE with
`pos_weight` 6.0 and deep supervision weights [1.0, 0.5, 0.25].

**3. Train**

```bash
python scripts/train.py --config config/config.yaml

python scripts/train.py --config config/config.yaml \
  --resume /home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/outputs/checkpoint.pth
```

Five-fold cross-validation:

```bash
python scripts/generate_5fold.py
python scripts/train_5fold.py --config config/config_5fold.yaml
python scripts/evaluate_5fold.py
```

**4. Inference**

```bash
python scripts/inference.py \
  --input      /home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/data/preprocess/processed/data/case_001.nii.gz \
  --output     /home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/outputs/case_001_pred.nii.gz \
  --checkpoint /home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/outputs/best.pth \
  --patch_size 64 128 128 \
  --overlap    0.5 \
  --threshold  0.5
```

Add `--save_prob` to write the probability map alongside the binary mask.

## Status

Research code accompanying an unpublished manuscript. Pretrained weights are
not currently released.
