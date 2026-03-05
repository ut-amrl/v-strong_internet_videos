import torch
from src.models.backbones import build_backbone

img_size = 1024
h, w = 576, 1024

# Create a random image (representing the actual video frame content)
import numpy as np
np.random.seed(0)
img_content = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

# 1. Padded image (as in the actual pipeline)
img_padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
img_padded[:h, :w] = img_content

# 2. Shifted image (to see if the top boundary behaves differently when NOT at the edge)
shift_y = 100
img_shifted = np.zeros((img_size, img_size, 3), dtype=np.uint8)
img_shifted[shift_y:shift_y+h, :w] = img_content

nsam = build_backbone(backbone_type='nanosam', backbone_variant='resnet18', img_size=img_size, trainable=False)
nsam.eval()

with torch.no_grad():
    feat_padded = nsam.encode([img_padded])[0] # (512, 64, 64)
    feat_shifted = nsam.encode([img_shifted])[0] # (512, 64, 64)

# The content at y=0 in padded should match the content at y=shift_y in shifted
# (roughly, downsampled by 16)
shift_feat_y = shift_y // 16

diff = torch.abs(feat_padded[:, 0, :] - feat_shifted[:, shift_feat_y, :]).mean()
print(f"Mean absolute difference at the top edge of the image content: {diff.item():.4f}")

# Look at the first few rows
print("\nTop 5 rows of feature map (channel 0) for PADDED (content starts at y=0):")
print(feat_padded[0, :5, 0])

print("\nRows corresponding to content start for SHIFTED (content starts at y=shift_feat_y):")
print(feat_shifted[0, shift_feat_y:shift_feat_y+5, 0])

# Are the first few rows of feat_padded significantly anomalous compared to the rest of the image?
print("\nMean feature norm across rows for PADDED:")
norms = torch.norm(feat_padded, dim=0).mean(dim=1)  # Mean across width -> (H,)
for i in range(10):
    print(f"Row {i:2d}: {norms[i].item():.4f}")
print(f"Mean of rows 10-20: {norms[10:20].mean().item():.4f}")
