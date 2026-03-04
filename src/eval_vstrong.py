import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data.vstrong_dataset import VStrongDataset
from models.vstrong_lit import VStrongLit


def _write_overlay(out_path: Path, overlay_bgr: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Eval V-STRONG checkpoint on a subset of frames and write overlays.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--points_per_class", type=int, default=64)
    parser.add_argument("--points_source", type=str, default="mixed", choices=["saved", "mask", "mixed"])
    parser.add_argument("--neg_top_frac", type=float, default=0.3)
    parser.add_argument("--sample_margin_px", type=int, default=10)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None, help="Force device like cpu/cuda. Default: auto.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    model = VStrongLit.load_from_checkpoint(str(ckpt_path), strict=False)
    model.eval()
    model_img_size = int(getattr(model, "img_size", getattr(model, "sam_img_size", 1024)))

    ds = VStrongDataset(
        dataset_dir=str(dataset_dir),
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        img_size=model_img_size,
        points_per_class=args.points_per_class,
        points_source=args.points_source,
        neg_top_frac=args.neg_top_frac,
        sample_margin_px=args.sample_margin_px,
    )

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n = min(int(args.num_frames), len(ds))
    print(f"Evaluating {n} frames from split={args.split} (dataset size={len(ds)})")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    wrote = 0
    for i in range(n):
        s = ds[i]
        emb = model._encode_images([s.resized_rgb]).to(device)  # 1xCxHfxWf
        z_map = model.proj(emb)[0].detach()  # DxHfxWf

        pos_pts = torch.from_numpy(s.pos_points_resized).to(device)
        viz = model._make_viz(
            resized_rgb=s.resized_rgb,
            z_map=z_map,
            pos_points=pos_pts,
        )
        if viz is None:
            continue

        frame_idx = int(s.frame_idx)
        out_path = out_dir / f"{frame_idx:06d}.jpg"
        _write_overlay(out_path, viz.overlay_bgr)
        wrote += 1

    print(f"Done. Wrote {wrote}/{n} overlays to {out_dir}")


if __name__ == "__main__":
    main()
