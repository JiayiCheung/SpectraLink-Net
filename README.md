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
