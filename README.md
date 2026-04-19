# Multimodal Biosignal-Based Immersion Level Classification

Binary classification of user immersion states using heterogeneous biological signals — EEG, GSR, and PPG — with a cross-modal attention-based deep learning model.

---

## Architecture

```
EEG  (14ch) ──► EEGNet encoder ──────────────────┐
GSR  (1ch)  ──► 1D CNN encoder ──► Linear proj ──► Modality Attention ──► Classifier ──► logit
PPG  (1ch)  ──► 1D CNN encoder ──────────────────┘
```

- **EEG branch**: EEGNet (temporal conv → depthwise conv → separable conv)
- **GSR / PPG branches**: Shared 1D CNN architecture with adaptive average pooling
- **Fusion**: Each modality projected to a shared feature space (128-dim), then fused via learned modality attention weights (softmax over 3 scalars)
- **Output**: Single logit → BCEWithLogitsLoss during training

---

## Results

| Split | Samples | Accuracy |
|-------|---------|----------|
| Train | 79,968  | 77.6%    |
| Val   | 7,995   | 79.7%    |
| Test  | 15,996  | **79.1%**|

```
Classification Report (Test set):
              precision    recall  f1-score
         0.0     0.7663    0.8696    0.8147
         1.0     0.8288    0.7044    0.7615
    accuracy                         0.7914
   macro avg     0.7976    0.7870    0.7881
```

---

## Tech Stack

- Python 3.x
- PyTorch
- NumPy, scikit-learn, matplotlib

---

## Project Structure

```
.
├── model.py        # FusionModel: EEGNet + 1D CNN + Modality Attention
├── Dataset.py      # BiosignalDataset: loads sliced .npy windows
├── train.py        # Training loop with early stopping
├── requirements.txt
└── README.md
```

---

## Usage

```bash
pip install -r requirements.txt

python train.py
```

> **Note**: Update `data_dir` and `split_dir` paths in `train.py` to point to your local dataset before running.

---

## Dataset

Pre-processed and sliced biosignal windows from a multimodal immersion experiment.
- Signals: EEG (14 channels), GSR (1 channel), PPG (1 channel)
- Window size: 128 samples
- Total: ~104K samples across train/val/test splits
- Raw data not included in this repository.
