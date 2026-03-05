"""Config-driven V-STRONG training entrypoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader
import yaml

from data.vstrong_dataset import VStrongDataset, vstrong_collate
from models.backbones import resolve_backbone_config
from models.vstrong_lit import VStrongLit


_DEFAULT_CONFIG = {
    "data": {
        "val_ratio": 0.1,
        "points_per_class": 64,
        "points_source": "mixed",
        "neg_top_frac": 0.3,
        "sample_margin_px": 10,
    },
    "model": {
        "backbone": "sam",
        "size": "small",
        "variant": None,
        "checkpoint": None,
        "img_size": None,
        "freeze_backbone": True,
        "embed_dim": 64,
        "proj_hidden": 256,
        "temperature": 0.1,
        "traversability_ema_alpha": 0.999,
    },
    "optimizer": {
        "lr": 1e-3,
        "backbone_lr": None,
        "weight_decay": 1e-4,
    },
    "trainer": {
        "seed": 0,
        "batch_size": 2,
        "num_workers": 4,
        "accelerator": "auto",
        "devices": "auto",
        "strategy": "auto",
        "num_nodes": 1,
        "precision": "32-true",
        "max_epochs": 5,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "fast_dev_run": False,
    },
    "logging": {
        "tensorboard_enabled": True,
        "tensorboard_name": "tensorboard",
        "tensorboard_version": "vstrong",
        "wandb_mode": "disabled",
        "wandb_project": "vstrong",
        "wandb_name": None,
        "log_every_n_steps": 10,
        "log_images_every_n_steps": 200,
    },
    "paths": {
        "logs_dir": "logs",
        "checkpoint_dir": "logs/checkpoints/vstrong",
        "default_root_dir": "logs",
    },
}


def _deep_merge(base: dict, updates: dict):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    cfg = deepcopy(_DEFAULT_CONFIG)
    if "dataset" in raw_cfg and "data" not in raw_cfg:
        raw_cfg["data"] = raw_cfg["dataset"]
    if "training" in raw_cfg and "trainer" not in raw_cfg:
        raw_cfg["trainer"] = raw_cfg["training"]

    _deep_merge(cfg, raw_cfg)
    cfg["_legacy_dataset"] = raw_cfg.get("dataset", {})
    cfg["_legacy_training"] = raw_cfg.get("training", {})
    return cfg


def _parse_devices(devices_val):
    if isinstance(devices_val, int):
        if devices_val == 0:
            return [0]
        return devices_val
    if isinstance(devices_val, list):
        return devices_val

    value = str(devices_val).strip()
    if value.lower() == "auto":
        return "auto"
    if value == "0":
        return [0]
    if value.isdigit():
        return int(value)
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if all(part.isdigit() for part in parts):
            return [int(part) for part in parts]
    return value


def _require(value, label: str):
    if value in {None, ""}:
        raise RuntimeError(f"Missing required config value: {label}")
    return value


def _normalize_dataset_dirs(data_cfg: dict, legacy_dataset_cfg: dict) -> list[Path]:
    dataset_dirs_val = data_cfg.get("dataset_dirs")
    dataset_dir_val = data_cfg.get("dataset_dir", legacy_dataset_cfg.get("dataset_dir"))

    # Allow: dataset_dir: [a,b,c]
    if dataset_dirs_val is None and isinstance(dataset_dir_val, list):
        dataset_dirs_val = dataset_dir_val

    # Allow: dataset_dirs: [...] (preferred)
    if dataset_dirs_val is not None:
        if not isinstance(dataset_dirs_val, list):
            raise RuntimeError("data.dataset_dirs must be a list of paths.")
        dirs = [Path(str(p)) for p in dataset_dirs_val if str(p).strip()]
        if not dirs:
            raise RuntimeError("data.dataset_dirs is empty.")
        return dirs

    # Default: single dir
    dataset_dir = Path(_require(dataset_dir_val, "data.dataset_dir"))
    return [dataset_dir]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train V-STRONG from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    optimizer_cfg = cfg["optimizer"]
    trainer_cfg = cfg["trainer"]
    logging_cfg = cfg["logging"]
    paths_cfg = cfg["paths"]
    legacy_dataset_cfg = cfg["_legacy_dataset"]
    legacy_training_cfg = cfg["_legacy_training"]

    seed = int(trainer_cfg.get("seed", legacy_training_cfg.get("seed", 0)))
    pl.seed_everything(seed, workers=True)

    backbone_cfg = resolve_backbone_config(
        backbone_type=model_cfg.get("backbone", model_cfg.get("backbone_type", "sam")),
        backbone_size=model_cfg.get("size", model_cfg.get("backbone_size")),
        backbone_variant=model_cfg.get("variant", model_cfg.get("backbone_variant")),
        img_size=model_cfg.get("img_size"),
    )

    dataset_dirs = _normalize_dataset_dirs(data_cfg, legacy_dataset_cfg)
    missing = [p for p in dataset_dirs if not p.exists()]
    if missing:
        raise RuntimeError("Some dataset dirs do not exist: " + ", ".join(str(p) for p in missing))

    ds_kwargs = dict(
        dataset_dir=[str(p) for p in dataset_dirs],
        val_ratio=float(data_cfg.get("val_ratio", legacy_dataset_cfg.get("val_ratio", 0.1))),
        seed=seed,
        img_size=backbone_cfg["img_size"],
        points_per_class=int(data_cfg.get("points_per_class", legacy_dataset_cfg.get("points_per_class", 64))),
        points_source=str(data_cfg.get("points_source", legacy_dataset_cfg.get("points_source", "mixed"))),
        neg_top_frac=float(data_cfg.get("neg_top_frac", legacy_dataset_cfg.get("neg_top_frac", 0.3))),
        sample_margin_px=int(data_cfg.get("sample_margin_px", legacy_dataset_cfg.get("sample_margin_px", 10))),
        ignore_mask_path=data_cfg.get("ignore_mask_path", None),
    )
    train_ds = VStrongDataset(split="train", **ds_kwargs)
    val_ds = VStrongDataset(split="val", **ds_kwargs)

    batch_size = int(trainer_cfg.get("batch_size", legacy_training_cfg.get("batch_size", 2)))
    num_workers = int(trainer_cfg.get("num_workers", legacy_training_cfg.get("num_workers", 4)))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vstrong_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vstrong_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    logs_dir = Path(paths_cfg.get("logs_dir", "logs"))
    default_root_dir = Path(paths_cfg.get("default_root_dir", logs_dir))
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "logs/checkpoints/vstrong"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    default_root_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loggers = []
    if bool(logging_cfg.get("tensorboard_enabled", True)):
        loggers.append(
            TensorBoardLogger(
                save_dir=str(logs_dir),
                name=str(logging_cfg.get("tensorboard_name", "tensorboard")),
                version=str(logging_cfg.get("tensorboard_version", "vstrong")),
            )
        )

    wandb_mode = str(logging_cfg.get("wandb_mode", "disabled"))
    if wandb_mode != "disabled":
        loggers.append(
            WandbLogger(
                project=str(logging_cfg.get("wandb_project", "vstrong")),
                name=logging_cfg.get("wandb_name"),
                save_dir=str(logs_dir),
                offline=(wandb_mode == "offline"),
                log_model=False,
            )
        )

    model = VStrongLit(
        backbone_type=backbone_cfg["backbone_type"],
        backbone_size=backbone_cfg["backbone_size"],
        backbone_variant=backbone_cfg["backbone_variant"],
        backbone_checkpoint=model_cfg.get("checkpoint", model_cfg.get("backbone_checkpoint")),
        img_size=backbone_cfg["img_size"],
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        proj_hidden=int(model_cfg.get("proj_hidden", 256)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        traversability_ema_alpha=float(model_cfg.get("traversability_ema_alpha", 0.999)),
        lr=float(optimizer_cfg.get("lr", legacy_training_cfg.get("lr", 1e-3))),
        backbone_lr=optimizer_cfg.get("backbone_lr"),
        weight_decay=float(optimizer_cfg.get("weight_decay", legacy_training_cfg.get("weight_decay", 1e-4))),
        log_images_every_n_steps=int(logging_cfg.get("log_images_every_n_steps", 200)),
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="vstrong-{epoch:03d}-{step:06d}",
        monitor="val/loss_epoch",
        mode="min",
        save_last=True,
        save_top_k=1,
        enable_version_counter=False,
    )

    trainer = pl.Trainer(
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=_parse_devices(trainer_cfg.get("devices", "auto")),
        strategy=trainer_cfg.get("strategy", "auto"),
        num_nodes=int(trainer_cfg.get("num_nodes", 1)),
        precision=trainer_cfg.get("precision", "32-true"),
        max_epochs=int(trainer_cfg.get("max_epochs", 5)),
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


if __name__ == "__main__":
    main()
