import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader

from data.vstrong_dataset import VStrongDataset, vstrong_collate
from models.vstrong_lit import VStrongLit


def main():
    parser = argparse.ArgumentParser(description="Train V-STRONG (contrastive) from SAM + pos/neg points.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory (e.g. data/output_2)")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_img_size", type=int, default=1024)

    parser.add_argument("--points_per_class", type=int, default=64)
    parser.add_argument("--points_source", type=str, default="mixed", choices=["saved", "mask", "mixed"])
    parser.add_argument("--neg_top_frac", type=float, default=0.3)
    parser.add_argument("--sample_margin_px", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--proj_hidden", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--log_images_every_n_steps", type=int, default=200)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="logs/checkpoints/vstrong")

    parser.add_argument("--wandb_project", type=str, default="vstrong")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f"dataset_dir does not exist: {dataset_dir}")

    train_ds = VStrongDataset(
        dataset_dir=str(dataset_dir),
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        sam_img_size=args.sam_img_size,
        points_per_class=args.points_per_class,
        points_source=args.points_source,
        neg_top_frac=args.neg_top_frac,
        sample_margin_px=args.sample_margin_px,
    )
    val_ds = VStrongDataset(
        dataset_dir=str(dataset_dir),
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        sam_img_size=args.sam_img_size,
        points_per_class=args.points_per_class,
        points_source=args.points_source,
        neg_top_frac=args.neg_top_frac,
        sample_margin_px=args.sample_margin_px,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=vstrong_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vstrong_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    loggers = []
    tb_logger = TensorBoardLogger(save_dir="logs", name="tensorboard", version="vstrong")
    loggers.append(tb_logger)

    if args.wandb_mode != "disabled":
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            save_dir="logs",
            offline=(args.wandb_mode == "offline"),
            log_model=False,
        )
        loggers.append(wandb_logger)

    model = VStrongLit(
        sam_checkpoint=args.sam_checkpoint,
        sam_type=args.sam_type,
        sam_img_size=args.sam_img_size,
        embed_dim=args.embed_dim,
        proj_hidden=args.proj_hidden,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_images_every_n_steps=args.log_images_every_n_steps,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="vstrong-{epoch:03d}-{step:06d}",
        monitor="val/loss_epoch",
        mode="min",
        save_last=True,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint_cb],
        enable_checkpointing=True,
        default_root_dir="logs",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
