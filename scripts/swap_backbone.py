"""Swap the SAM backbone in a trained VStrongLit checkpoint with a distilled ResNet.

After training V-STRONG with a SAM encoder (e.g. sam_small.yaml), the learned
projection head and traversability vector live entirely in SAM's feature space.
If you then distill a ResNet to mimic that SAM encoder, you can transplant the
projection head directly — no retraining needed (or only a few fine-tune epochs).

Usage
-----
    python scripts/swap_backbone.py \\
        --vstrong_ckpt   logs/vstrong/checkpoints/vstrong/last.ckpt \\
        --distilled_backbone checkpoints/distilled_resnet101_samB.pth \\
        --backbone_variant resnet101 \\
        --output         checkpoints/vstrong_resnet101_samB.ckpt

The output .ckpt can be loaded with::

    model = VStrongLit.load_from_checkpoint("checkpoints/vstrong_resnet101_samB.ckpt")

or passed as `backbone_checkpoint` to a new VStrongLit for further fine-tuning.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

# ── Bootstrap src/ on path ─────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Swap a SAM backbone in a VStrongLit checkpoint with a distilled ResNet."
    )
    parser.add_argument(
        "--vstrong_ckpt",
        required=True,
        help="Path to the .ckpt produced by train.py (SAM-based V-STRONG training).",
    )
    parser.add_argument(
        "--distilled_backbone",
        required=True,
        help="Path to the distilled ResNet backbone .pth (produced by train_distill.py).",
    )
    parser.add_argument(
        "--backbone_variant",
        default="resnet101",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="Which ResNet variant the distilled backbone is (default: resnet101).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1024,
        help="Image size used during training (default: 1024).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the swapped checkpoint (.ckpt or .pth).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a quick forward pass to verify the swapped model works.",
    )
    args = parser.parse_args(argv)

    import torch
    from models.vstrong_lit import VStrongLit

    # ── 1. Load the SAM-trained checkpoint ──────────────────────────────────
    print(f"Loading SAM-trained V-STRONG checkpoint: {args.vstrong_ckpt}")
    sam_ckpt = torch.load(args.vstrong_ckpt, map_location="cpu")

    # Extract hyperparameters saved by Lightning so we can reconstruct the model
    hparams = sam_ckpt.get("hyper_parameters", {})
    state_dict = sam_ckpt.get("state_dict", sam_ckpt)

    print(f"  Hyperparameters found: {list(hparams.keys())}")

    # Pull projection head + traversability buffer weights from SAM checkpoint
    proj_keys   = {k: v for k, v in state_dict.items() if k.startswith("proj.")}
    trav_keys   = {k: v for k, v in state_dict.items() if k.startswith("traversability_")}

    if not proj_keys:
        raise RuntimeError(
            "No 'proj.*' keys found in the checkpoint state_dict. "
            "Make sure --vstrong_ckpt points to a Lightning .ckpt file."
        )

    print(f"  Found {len(proj_keys)} projection head tensors.")
    print(f"  Found {len(trav_keys)} traversability buffer tensors.")

    # ── 1b. Resolve the distilled backbone .pth ──────────────────────────────
    # If the user passes a Lightning distill .ckpt, extract and translate keys
    # the same way export_backbone_state_dict() does, so NanoSAMBackbone can load it.
    distilled_path = Path(args.distilled_backbone)
    resolved_backbone_path = distilled_path

    payload = torch.load(str(distilled_path), map_location="cpu")
    if "state_dict" in payload:
        print(f"\nDetected Lightning checkpoint at --distilled_backbone. Extracting backbone weights …")
        raw_sd = payload["state_dict"]
        # Keys are like "student.stem.0.weight", "student.layer1.0.conv1.weight", etc.
        out = {}
        for k, v in raw_sd.items():
            # strip leading "student." prefix
            stripped = k
            for prefix in ("student.", "model.", "encoder.", "backbone."):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            # translate stem.0.* → conv1.*, stem.1.* → bn1.*
            if stripped.startswith("stem.0."):
                out[stripped.replace("stem.0.", "conv1.")] = v
            elif stripped.startswith("stem.1."):
                out[stripped.replace("stem.1.", "bn1.")] = v
            elif any(stripped.startswith(p) for p in ("layer1.", "layer2.", "layer3.", "layer4.")):
                out[stripped] = v
        if not out:
            raise RuntimeError(
                "Could not extract any backbone keys from the Lightning distill checkpoint. "
                "Check that it was produced by train_distill.py."
            )
        # Save to a temp file so NanoSAMBackbone can load it from path
        import tempfile, os
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.save(out, tmp_path)
        resolved_backbone_path = Path(tmp_path)
        print(f"  Extracted {len(out)} backbone tensors → temp file {tmp_path}")

    # ── 2. Build a new VStrongLit with the distilled ResNet backbone ─────────
    print(f"\nBuilding new VStrongLit with NanoSAM ({args.backbone_variant}) backbone …")

    # Preserve all original hparams but override the backbone parts
    new_hparams = copy.deepcopy(hparams)
    new_hparams["backbone_type"]       = "nanosam"
    new_hparams["backbone_size"]       = None
    new_hparams["backbone_variant"]    = args.backbone_variant
    new_hparams["backbone_checkpoint"] = args.distilled_backbone
    new_hparams["img_size"]            = args.img_size
    # strip legacy SAM-only keys that VStrongLit no longer needs
    for legacy in ("sam_checkpoint", "sam_type", "sam_img_size"):
        new_hparams.pop(legacy, None)

    new_model = VStrongLit(**new_hparams)

    # ── 3. Transplant projection head and traversability vector ──────────────
    print("Transplanting projection head and traversability vector …")
    new_sd = new_model.state_dict()

    mismatched = []
    transplanted = []
    for k, v in {**proj_keys, **trav_keys}.items():
        if k in new_sd:
            if new_sd[k].shape == v.shape:
                new_sd[k] = v
                transplanted.append(k)
            else:
                mismatched.append((k, new_sd[k].shape, v.shape))
        else:
            print(f"  Warning: key {k!r} not found in new model — skipping.")

    if mismatched:
        print(f"\n  ⚠ Shape mismatches (backbone feature_dim changed) — skipping these keys:")
        for k, new_shape, old_shape in mismatched:
            print(f"    {k!r}: checkpoint={old_shape}  new_model={new_shape}")
        print(
            "\n  These layers will use random initialization and must be fine-tuned.\n"
            "  This is expected when the backbone feature_dim differs between SAM\n"
            "  (256-d) and a ResNet (e.g. resnet18/34 → 512-d, resnet50/101 → 2048-d)."
        )

    if transplanted:
        print(f"  ✓ Transplanted {len(transplanted)} tensors: {transplanted}")

    new_model.load_state_dict(new_sd)
    print("  ✓ Projection head transplanted successfully.")

    traversability_initialized = new_sd.get("traversability_initialized")
    if traversability_initialized is not None and traversability_initialized.item():
        print("  ✓ Traversability vector transplanted (was initialized in SAM run).")
    else:
        print("  ⚠ Traversability vector was NOT yet initialized in source checkpoint.")
        print("    A few forward passes will initialize it via EMA on the new model.")

    # ── 4. Optionally verify with a dummy forward pass ────────────────────────
    if args.verify:
        print("\nRunning dummy forward pass …")
        import numpy as np
        new_model.eval()
        dummy_img = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
        with torch.no_grad():
            features = new_model.backbone.encode([dummy_img])
            z = new_model.proj(features)
        print(f"  ✓ Forward pass OK — output shape: {z.shape}")

    # ── 5. Save ──────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap in a Lightning-compatible checkpoint dict so it can be loaded with
    # VStrongLit.load_from_checkpoint()
    new_ckpt = {
        "epoch": sam_ckpt.get("epoch", 0),
        "global_step": sam_ckpt.get("global_step", 0),
        "state_dict": new_model.state_dict(),
        "hyper_parameters": new_hparams,
        "pytorch-lightning_version": sam_ckpt.get("pytorch-lightning_version", ""),
    }
    torch.save(new_ckpt, str(out_path))
    print(f"\n✓ Swapped checkpoint saved → {out_path}")
    print(
        "\nTo fine-tune or evaluate, use:\n"
        f"  python train.py --config <your_distilled_resnet_vstrong_config.yaml>\n"
        "  and set:\n"
        f"    model:\n"
        f"      backbone: nanosam\n"
        f"      variant: {args.backbone_variant}\n"
        f"      backbone_checkpoint: {args.distilled_backbone}\n"
    )


if __name__ == "__main__":
    main()
