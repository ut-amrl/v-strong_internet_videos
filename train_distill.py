"""Distillation training entrypoint.

Trains a ResNet student to reproduce the feature maps of a frozen SAM teacher.
After training, the best student checkpoint is exported as a plain backbone
state dict that can be loaded by VStrongLit via NanoSAMBackbone.

Usage:
    conda run -n env_isaaclab python train_distill.py --config configs/distill/distill_resnet18.yaml
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path


def _bootstrap_src_path():
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))


def main(argv: list[str] | None = None):
    _bootstrap_src_path()

    import torch
    import yaml
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from torch.utils.data import DataLoader

    from data.distill_dataset import DistillDataset, distill_collate
    from models.distill_lit import DistillLit

    parser = argparse.ArgumentParser(description="Distill SAM into a ResNet student.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg      = cfg.get("data", {})
    teacher_cfg   = cfg.get("teacher", {})
    student_cfg   = cfg.get("student", {})
    optimizer_cfg = cfg.get("optimizer", {})
    trainer_cfg   = cfg.get("trainer", {})
    logging_cfg   = cfg.get("logging", {})
    paths_cfg     = cfg.get("paths", {})

    seed = int(trainer_cfg.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    # ── Datasets ────────────────────────────────────────────────────────────
    img_size = int(teacher_cfg.get("img_size", 1024))
    ds_kwargs = dict(
        dataset_dir=data_cfg["dataset_dir"],
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        img_size=img_size,
    )
    train_ds = DistillDataset(split="train", **ds_kwargs)
    val_ds   = DistillDataset(split="val",   **ds_kwargs)

    batch_size  = int(trainer_cfg.get("batch_size", 4))
    num_workers = int(trainer_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=distill_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=distill_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = DistillLit(
        sam_checkpoint=teacher_cfg["checkpoint"],
        sam_variant=teacher_cfg.get("sam_variant", "vit_h"),
        student_variant=student_cfg.get("variant", "resnet18"),
        student_pretrained=bool(student_cfg.get("pretrained", True)),
        img_size=img_size,
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
        lambda_mse=float(optimizer_cfg.get("lambda_mse", 1.0)),
        lambda_cos=float(optimizer_cfg.get("lambda_cos", 0.5)),
    )

    # ── Logging ─────────────────────────────────────────────────────────────
    logs_dir         = Path(paths_cfg.get("logs_dir", "logs"))
    checkpoint_dir   = Path(paths_cfg.get("checkpoint_dir", "logs/checkpoints/distill"))
    default_root_dir = Path(paths_cfg.get("default_root_dir", logs_dir))
    export_path      = Path(paths_cfg.get("export_path", "checkpoints/distilled_student.pth"))

    for p in (logs_dir, checkpoint_dir, default_root_dir, export_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    loggers = []
    if bool(logging_cfg.get("tensorboard_enabled", True)):
        loggers.append(TensorBoardLogger(
            save_dir=str(logs_dir),
            name=str(logging_cfg.get("tensorboard_name", "tensorboard")),
            version=str(logging_cfg.get("tensorboard_version", "distill")),
        ))
    wandb_mode = str(logging_cfg.get("wandb_mode", "disabled"))
    if wandb_mode != "disabled":
        loggers.append(WandbLogger(
            project=str(logging_cfg.get("wandb_project", "vstrong")),
            name=logging_cfg.get("wandb_name"),
            save_dir=str(logs_dir),
            offline=(wandb_mode == "offline"),
        ))

    # ── Callbacks ───────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="distill-{epoch:03d}-{step:06d}",
        monitor="val/loss_epoch",
        mode="min",
        save_last=True,
        save_top_k=1,
        enable_version_counter=False,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    devices = trainer_cfg.get("devices", "auto")
    if isinstance(devices, list):
        pass
    elif str(devices).lower() == "auto":
        devices = "auto"

    trainer = pl.Trainer(
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=devices,
        strategy=trainer_cfg.get("strategy", "auto"),
        num_nodes=int(trainer_cfg.get("num_nodes", 1)),
        precision=trainer_cfg.get("precision", "16-mixed"),
        max_epochs=int(trainer_cfg.get("max_epochs", 20)),
        logger=(loggers or False),
        log_every_n_steps=int(logging_cfg.get("log_every_n_steps", 10)),
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        fast_dev_run=bool(trainer_cfg.get("fast_dev_run", False)),
        callbacks=[checkpoint_cb],
        enable_checkpointing=True,
        default_root_dir=str(default_root_dir),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ── Export distilled backbone weights ───────────────────────────────────
    print(f"\nExporting distilled backbone weights → {export_path}")
    sd = model.student.export_backbone_state_dict()
    torch.save(sd, str(export_path))
    print("Done.")


if __name__ == "__main__":
    main()
