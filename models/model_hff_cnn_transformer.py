import torch.nn as nn
from config import *
from models.model_cnn_transformer import sinusoidal_pos_enc


class HFFCNNTransformerOCR(nn.Module):
    """
    Hierarchical Feature Fusion OCR model.
    Takes feature maps with two CNN levels and fuses them with final level
    before they go to transformer, giving it both coarse and fine features.
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # ── Block 1+2: [B,1,64,288] → [B,128,16,144]
        self.cnn_shallow = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # → [B,64,32,144]

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)), # → [B,128,16,144]
            nn.Dropout2d(0.05),
        )

        # ── Block 3+4+5+6: [B,128,16,144] → [B,512,4,144] ─────────────────
        self.cnn_deep = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)), # → [B,256,8,144]
            nn.Dropout2d(0.05),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)), # → [B,512,4,144]
        )

        # ── Final height collapse: [B,512,4,144] → [B,512,1,144] ───
        self.cnn_collapse = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(4, 1), padding=0),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # ── Projections for scale1 and scale2 to d=512 ────────────────────
        # scale1: [B,128,16,144] → avg pool height → [B,128,1,144] → proj → [B,512,1,144]
        self.proj_shallow = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),             # collapses height
            nn.Conv2d(128, 512, 1), # 1×1 projection
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # scale2: [B,512,4,144] → avg pool height → [B,512,1,144]
        self.proj_deep = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # ── Fusion: concat 3×512 → 512 via conv1×1 ─────────────────
        self.fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 512, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, IMG_H, IMG_W)
            s = self.cnn_shallow(dummy)
            d = self.cnn_deep(s)
            c = self.cnn_collapse(d)
            self.max_T = c.shape[-1]  # 144

        self.register_buffer(
            "pos_enc", sinusoidal_pos_enc(self.max_T, 512), persistent=True
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=False,
            norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=6)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.cnn_shallow(x)
        s2 = self.cnn_deep(s1)
        s3 = self.cnn_collapse(s2)

        p1 = self.proj_shallow(s1)
        p2 = self.proj_deep(s2)

        fused = self.fusion(
            torch.cat([p1, p2, s3], dim=1)
        )

        f = fused.squeeze(2).permute(2, 0, 1)  # [144, B, 512]
        T = f.size(0)
        f = f + self.pos_enc[:T]

        mask = self.estimate_src_key_padding_mask(f)  # ← novo
        y = self.transformer(f, src_key_padding_mask=mask)  # ← added argument
        return self.classifier(y)           # [144, B, num_classes]

    def estimate_src_key_padding_mask(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: [T, B, 512] — CNN features after squeeze i permute
        Vraća: [B, T] bool mask — True = ignore that position
        """
        # energy by timestep: how "active" is each column
        energy = f.abs().mean(dim=-1)  # [T, B]
        energy = energy.permute(1, 0)  # [B, T]

        # limit: columns below X% max energy are getting masked
        threshold = energy.max(dim=1, keepdim=True).values * 0.05
        mask = energy < threshold  # [B, T] — True = empty column

        # safety: never mask everything — leave at least first 10
        mask[:, :10] = False
        return mask