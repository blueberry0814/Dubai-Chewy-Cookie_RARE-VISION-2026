
##ICPR 2026 RARE-VISION Competition — Team: Dubai Chewy Cookie

Anatomy-guided temporal multi-label classification for Video Capsule Endoscopy (VCE).
Classifies 17 classes per frame: 8 anatomical regions + 9 pathological findings.

![그림1](https://github.com/user-attachments/assets/7c6264a5-edd1-4fb0-aa42-0aba01d0c3dd)

---

## Repository Structure

```
GALAR-TemporalNet/
├── configs/
│   └── config.yaml          # All hyperparameters
├── data/
│   └── dataset.py           # Sliding-window dataset
├── models/
│   └── model.py             # GALAR-TemporalNet architecture
├── utils/
│   ├── layers.py            # GCN layers
│   ├── losses.py            # ASL + auxiliary losses
│   ├── make_json.py         # Submission JSON builder
│   └── viterbi.py           # Viterbi post-processing
├── extract_features.py      # Step 1: DINOv2 feature extraction
├── train.py                 # Step 2: Training
├── inference.py             # Step 3a: Inference (labeled data)
├── test_submit.py           # Step 3b: Inference for competition test data
├── checkpoints/             # Saved model weights (place best_model.pth here)
└── requirements.txt
```

---

## Expected Data Structure

```
galar_dataset/
├── 1/
│   ├── frame_000100.PNG
│   ├── frame_000105.PNG
│   └── ...
├── 2/
│   └── ...
└── Labels/
    ├── 1.csv
    ├── 2.csv
    └── ...
```

Each label CSV must contain columns: `frame`, and one column per class label.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1: Extract DINOv2 Features

```bash
python extract_features.py \
    --frames_root ./galar_dataset \
    --labels_dir  ./galar_dataset/Labels \
    --output_dir  ./features
```

Output: `features/{video_id}_features.npy`, `features/{video_id}_frames.npy`

---

## Step 2: Train

```bash
python train.py --config configs/config.yaml
```

Key config options (`configs/config.yaml`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.root_dir` | `./galar_dataset` | Dataset root |
| `data.features_dir` | `./features` | Extracted features |
| `training.val_ratio` | `0.2` | Validation split (0.0 = full data) |
| `training.num_epochs` | `200` | Total epochs |
| `training.device` | `cuda` | Device |

Best model saved to `checkpoints/best_model.pth`.

---

## Step 3a: Inference (Labeled Validation Data)

```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --test_features_dir ./features \
    --test_labels_dir ./galar_dataset/Labels \
    --output_dir ./results
```

---

## Step 3b: Inference for Competition Test Data

For unlabeled test videos (e.g., `ukdd_navi_00051`):

```bash
python test_submit.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --frames_root ./test_data \
    --feat_cache_dir ./features/test \
    --output_dir ./results/submission
```

Output: `results/submission/prediction.json` — ready for the competition evaluator.

---

## Results

| Metric | Score |
|--------|-------|
| Overall mAP@0.5 | 0.2644 |
| Overall mAP@0.95 | 0.2353 |

| Video | mAP@0.5 | mAP@0.95 |
|-------|---------|----------|
| ukdd_navi_00051 | 0.4153 | 0.4118 |
| ukdd_navi_00068 | 0.2392 | 0.1766 |
| ukdd_navi_00076 | 0.1388 | 0.1177 |
