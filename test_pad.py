import cv2
import numpy as np
import torch
from src.models.backbones import build_backbone

# create a dummy image HxWx3
img = np.ones((576, 1024, 3), dtype=np.uint8) * 128

# SAM
sam = build_backbone(backbone_type="sam", backbone_variant="vit_b", img_size=1024, trainable=False)
sam_feat = sam.encode([img])[0]
print(f"SAM feat: {sam_feat.shape}, top left corner: {sam_feat[:, 0, 0][:5]}")

# NanoSAM
nsam = build_backbone(backbone_type="nanosam", backbone_variant="resnet18", img_size=1024, trainable=False)
nsam_feat = nsam.encode([img])[0]
print(f"NanoSAM feat: {nsam_feat.shape}, top left corner: {nsam_feat[:, 0, 0][:5]}")
