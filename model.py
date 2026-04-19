"""
FusionModel: Multimodal Biosignal-Based Immersion Level Classification
-----------------------------------------------------------------------
Architecture:
  - EEG branch  : EEGNet (depthwise + separable conv)
  - GSR branch  : 1D CNN
  - PPG branch  : 1D CNN
  - Fusion      : Concatenation → Modality Attention → Binary classifier

Input shapes (per batch):
  eeg : (B, 1, EEG_CHANNELS, T)   default EEG_CHANNELS=14
  gsr : (B, 1, T)
  ppg : (B, 1, T)

NOTE: Adjust EEG_CHANNELS if your dataset uses a different montage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1. EEG Encoder  (EEGNet)
# ──────────────────────────────────────────────
class EEGNet(nn.Module):
    """
    Compact EEGNet encoder.
    Ref: Lawhern et al., 2018 (simplified for binary classification pipeline).
    """
    def __init__(self, eeg_channels: int = 14, samples: int = 128,
                 F1: int = 8, D: int = 2, F2: int = 16, dropout: float = 0.5):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(eeg_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        # compute flattened feature size
        self._feature_dim = self._get_feature_dim(eeg_channels, samples, F1, D, F2)

    def _get_feature_dim(self, eeg_channels, samples, F1, D, F2):
        with torch.no_grad():
            x = torch.zeros(1, 1, eeg_channels, samples)
            x = self.temporal_conv(x)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        # x: (B, 1, eeg_channels, T)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        return x.view(x.size(0), -1)   # (B, feature_dim)

    @property
    def output_dim(self):
        return self._feature_dim


# ──────────────────────────────────────────────
# 2. Peripheral Signal Encoder  (1D CNN)
#    shared architecture for GSR and PPG
# ──────────────────────────────────────────────
class SignalEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (B, out_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self._out_dim = out_dim

    def forward(self, x):
        # x: (B, 1, T)
        x = self.net(x)
        x = x.squeeze(-1)           # (B, out_dim)
        return self.dropout(x)

    @property
    def output_dim(self):
        return self._out_dim


# ──────────────────────────────────────────────
# 3. Modality Attention
# ──────────────────────────────────────────────
class ModalityAttention(nn.Module):
    """
    Learns a scalar attention weight for each modality feature vector,
    then returns a weighted sum over the concatenated representation.
    """
    def __init__(self, eeg_dim: int, gsr_dim: int, ppg_dim: int):
        super().__init__()
        self.eeg_proj = nn.Linear(eeg_dim, 1)
        self.gsr_proj = nn.Linear(gsr_dim, 1)
        self.ppg_proj = nn.Linear(ppg_dim, 1)

    def forward(self, f_eeg, f_gsr, f_ppg):
        # scalar score per modality
        scores = torch.cat([
            self.eeg_proj(f_eeg),   # (B, 1)
            self.gsr_proj(f_gsr),   # (B, 1)
            self.ppg_proj(f_ppg),   # (B, 1)
        ], dim=1)                   # (B, 3)
        weights = F.softmax(scores, dim=1)  # (B, 3)

        # weighted features (broadcast over feature dim)
        attended = (
            weights[:, 0:1] * f_eeg +
            weights[:, 1:2] * f_gsr +   # requires same dim → projected below
            weights[:, 2:3] * f_ppg
        )
        return attended, weights        # return weights for interpretability


# ──────────────────────────────────────────────
# 4. FusionModel  (top-level)
# ──────────────────────────────────────────────
class FusionModel(nn.Module):
    """
    Full multimodal fusion model.

    Forward inputs:
        eeg : (B, 1, EEG_CHANNELS, T)
        gsr : (B, 1, T)
        ppg : (B, 1, T)

    Returns:
        logits  : (B, 1)   — use BCEWithLogitsLoss during training
        weights : (B, 3)   — modality attention weights [eeg, gsr, ppg]
    """
    def __init__(self,
                 eeg_channels: int = 14,   # ← adjust if needed
                 eeg_samples:  int = 128,
                 peripheral_dim: int = 64,
                 hidden_dim: int = 128,
                 dropout: float = 0.4):
        super().__init__()

        # encoders
        self.eeg_enc = EEGNet(eeg_channels=eeg_channels, samples=eeg_samples)
        self.gsr_enc = SignalEncoder1D(out_dim=peripheral_dim)
        self.ppg_enc = SignalEncoder1D(out_dim=peripheral_dim)

        eeg_dim = self.eeg_enc.output_dim

        # project all features to the same dim before attention weighted sum
        self.eeg_proj = nn.Linear(eeg_dim, hidden_dim)
        self.gsr_proj = nn.Linear(peripheral_dim, hidden_dim)
        self.ppg_proj = nn.Linear(peripheral_dim, hidden_dim)

        # modality attention (operates on projected features)
        self.attention = ModalityAttention(hidden_dim, hidden_dim, hidden_dim)

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # binary logit
        )

    def forward(self, eeg, gsr, ppg):
        # encode each modality
        f_eeg = self.eeg_enc(eeg)           # (B, eeg_dim)
        f_gsr = self.gsr_enc(gsr)           # (B, peripheral_dim)
        f_ppg = self.ppg_enc(ppg)           # (B, peripheral_dim)

        # project to shared space
        f_eeg = F.relu(self.eeg_proj(f_eeg))    # (B, hidden_dim)
        f_gsr = F.relu(self.gsr_proj(f_gsr))    # (B, hidden_dim)
        f_ppg = F.relu(self.ppg_proj(f_ppg))    # (B, hidden_dim)

        # modality attention → weighted sum
        fused, attn_weights = self.attention(f_eeg, f_gsr, f_ppg)  # (B, hidden_dim)

        # classify
        logits = self.classifier(fused)     # (B, 1)
        return logits, attn_weights


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────
if __name__ == "__main__":
    B, T = 4, 128
    eeg = torch.randn(B, 1, 14, T)
    gsr = torch.randn(B, 1, T)
    ppg = torch.randn(B, 1, T)

    model = FusionModel()
    logits, weights = model(eeg, gsr, ppg)

    print(f"logits  : {logits.shape}")    # (4, 1)
    print(f"weights : {weights.shape}")   # (4, 3)
    print(f"attention weights (sample 0): {weights[0].detach()}")
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
