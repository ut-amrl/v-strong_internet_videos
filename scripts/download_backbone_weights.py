"""Download local checkpoints for supported frozen backbones."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import torch
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)


_OFFICIAL_URLS = {
    "sam": {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "dino": {
        "dino_vits16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        "dino_vits8": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth",
        "dino_vitb16": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        "dino_vitb8": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
    },
    "dinov2": {
        "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        "dinov2_vitg14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    },
}

_NANOSAM_MODELS = {
    "resnet18":  {"builder": resnet18,  "weights": ResNet18_Weights.IMAGENET1K_V1},
    "resnet34":  {"builder": resnet34,  "weights": ResNet34_Weights.IMAGENET1K_V1},
    "resnet50":  {"builder": resnet50,  "weights": ResNet50_Weights.IMAGENET1K_V2},
    "resnet101": {"builder": resnet101, "weights": ResNet101_Weights.IMAGENET1K_V2},
}


def _bootstrap_src_path():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


def _download(url: str, output_path: Path, retries: int = 3):
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0"}

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response, open(tmp_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(output_path)
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                wait_s = attempt * 2
                print(f"Download failed (attempt {attempt}/{retries}): {exc}")
                print(f"Retrying in {wait_s}s...")
                time.sleep(wait_s)

    raise RuntimeError(f"Failed to download after {retries} attempts: {last_error}") from last_error


def _download_nanosam_weights(variant: str, output_path: Path):
    spec = _NANOSAM_MODELS.get(variant)
    if spec is None:
        raise RuntimeError(f"No NanoSAM download mapping configured for variant: {variant}")

    model = spec["builder"](weights=spec["weights"])
    torch.save(model.state_dict(), str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Download local checkpoints for supported backbones.")
    parser.add_argument("--backbone", type=str, required=True, choices=["sam", "nanosam", "dino", "dinov2"])
    parser.add_argument("--size", type=str, default="small", help="small | medium | large")
    parser.add_argument("--variant", type=str, default=None, help="Optional concrete variant override.")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path. Default depends on backbone/variant.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists.")
    args = parser.parse_args()

    _bootstrap_src_path()
    from models.backbones import resolve_backbone_config

    resolved = resolve_backbone_config(
        backbone_type=args.backbone,
        backbone_size=args.size,
        backbone_variant=args.variant,
        img_size=None,
    )
    variant = resolved["backbone_variant"]
    if args.output is not None:
        output_path = Path(args.output)
    elif args.backbone == "nanosam":
        output_path = Path("checkpoints") / f"nanosam_{variant}.pth"
    else:
        output_path = Path("checkpoints") / f"{variant}.pth"
    if output_path.exists() and not args.force:
        print(f"Checkpoint already exists: {output_path}")
        print("Use --force to overwrite.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.backbone} weights")
    print(f"  size    : {resolved['backbone_size']}")
    print(f"  variant : {variant}")
    print(f"  output  : {output_path}")

    if args.backbone == "nanosam":
        print("  source  : torchvision ImageNet pretrained weights")
        _download_nanosam_weights(variant=variant, output_path=output_path)
    else:
        url = _OFFICIAL_URLS[args.backbone].get(variant)
        if url is None:
            raise RuntimeError(f"No official download URL configured for {args.backbone}:{variant}")
        # For SAM use the canonical filename embedded in the URL.
        if args.backbone == "sam" and args.output is None:
            output_path = Path("checkpoints") / Path(url).name
            if output_path.exists() and not args.force:
                print(f"Checkpoint already exists: {output_path}")
                print("Use --force to overwrite.")
                return
            output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  url     : {url}")
        _download(url=url, output_path=output_path)
    print(f"Saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()
