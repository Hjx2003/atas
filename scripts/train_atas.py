import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import swanlab
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.config import load_config
from src.datasets.dataset import build_dataset
from src.datasets.mosaic import make_2x2_mosaic
from src.losses.atas_losses import (
    global_to_global_loss,
    global_to_local_loss,
    local_to_local_loss,
    split_2x2_patch_tokens,
)
from src.models.atas import CLIPVisionWrapper
from src.utils.utils import load_checkpoint, save_checkpoint, set_seed, unwrap_model


def setup_distributed():
    """
    Works in both modes:
      Single GPU / normal python: no dist init, no DDP, no DistributedSampler.
      Multi GPU / torchrun: init NCCL, use DDP and DistributedSampler.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training with NCCL requires CUDA.")

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_rank(), dist.get_world_size()

    return False, 0, 0, 1


def setup_logger(log_dir, rank: int = 0):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ATAS")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def build_model(cfg, device):
    teacher = CLIPVisionWrapper(
        model_name=cfg.model.backbone.model_name,
        pretrained_path=cfg.model.backbone.pretrained_path,
    ).to(device)

    student = copy.deepcopy(teacher).to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student.train()
    return teacher, student


def train(args):
    distributed, local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    amp_enabled = cfg.train.amp and torch.cuda.is_available()
    set_seed(args.seed + rank)

    logger = setup_logger(cfg.logs.dir, rank=rank)

    if rank == 0:
        logger.info(f"Config path: {args.config}")
        logger.info(f"Using device: {device}")
        logger.info(f"distributed={distributed}, rank={rank}, world_size={world_size}")

    use_swanlab = bool(cfg.swanlab.enable) and rank == 0
    if use_swanlab:
        swanlab.init(
            project=cfg.swanlab.project,
            experiment_name=cfg.swanlab.name,
            config=json.loads(json.dumps(cfg, default=lambda o: o.__dict__)),
        )

    dataset = build_dataset(cfg.dataset)
    if rank == 0:
        logger.info(f"Dataset size: {len(dataset)}")

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None

    loader = DataLoader(
        dataset,
        # Paper says batch size 36/GPU.
        # Here that means 36 mosaic samples per GPU, requiring 36*4 raw object images.
        batch_size=cfg.train.batch_size * 4,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.train.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    teacher, student = build_model(cfg, device)

    if distributed:
        student = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    student_core = unwrap_model(student)

    optimizer = torch.optim.AdamW(
        student_core.visual.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            path=args.resume,
            student=student_core,
            teacher=teacher,
            optimizer=optimizer,
            scaler=scaler,
            strict=False,
            device=device,
        )
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {args.resume}")
            logger.info(f"Start epoch: {start_epoch}, global_step: {global_step}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        if distributed:
            sampler.set_epoch(epoch)

        student.train()

        epoch_loss = 0.0
        epoch_gld = 0.0
        epoch_lld = 0.0
        epoch_ggd = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{cfg.train.epochs}",
            dynamic_ncols=True,
            disable=(rank != 0),
        )

        for step, (images, _) in enumerate(pbar, start=1):
            global_step += 1
            images = images.to(device, non_blocking=True)

            # Raw batch: [B*4, 3, H, W] -> [B, 4, 3, H, W]
            b = images.size(0) // 4
            images = images[: b * 4].view(b, 4, *images.shape[1:])

            individual_images = images.reshape(b * 4, *images.shape[2:])
            mosaic_images = make_2x2_mosaic(
                images,
                output_size=cfg.dataset.image_size,
            )

            # Paper alignment:
            # 1) Global representations: teacher CLS from individual object-centric images.
            # 2) Local representations: student patch tokens from mosaic images.
            # 3) LLD: preserve teacher patch-token relational structure on the same mosaic image.
            with torch.no_grad():
                teacher_cls_individual, _ = teacher.forward_features(individual_images)
                _, teacher_patch_mosaic = teacher.forward_features(mosaic_images)
            # ====== 加在这里 ======
            # with torch.no_grad():

            #     t = F.normalize(
            #         teacher_cls_individual.float(),
            #         dim=-1,
            #     )

            #     sim = t @ t.t()

            #     offdiag = (
            #         sim.sum() - sim.diag().sum()
            #     ) / (
            #         sim.numel() - sim.size(0)
            #     )

            #     if rank == 0 and global_step % cfg.logs.log_interval == 0:

            #         logger.info(
            #             f"teacher self sim "
            #             f"diag={sim.diag().mean().item():.6f}, "
            #             f"offdiag={offdiag.item():.6f}, "
            #             f"feature_std={t.std(dim=0).mean().item():.6f}"
            #         )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                student_cls_individual, _ = student_core.forward_features(individual_images)
                _, student_patch_mosaic = student_core.forward_features(mosaic_images)
                # # ===== 加在这里 =====
                # with torch.no_grad():
                #     s = F.normalize(student_cls_individual.float(), dim=-1)
                #     t = F.normalize(teacher_cls_individual.float(), dim=-1)

                #     logits = s @ t.t()

                #     diag = logits.diag().mean()

                #     offdiag = (
                #         (logits.sum() - logits.diag().sum())
                #         / (logits.numel() - logits.size(0))
                #     )

                #     logger.info(
                #         f"debug ggd diag={diag.item():.6f}, "
                #         f"offdiag={offdiag.item():.6f}, "
                #         f"s_norm={student_cls_individual.float().norm(dim=-1).mean().item():.6f}, "
                #         f"t_norm={teacher_cls_individual.float().norm(dim=-1).mean().item():.6f}"
                #     )
                # GLD:
                # Split the student mosaic patches into 4 quadrants.
                # Each quadrant is supervised by the corresponding teacher CLS from the original image.
                student_patch_quadrants = split_2x2_patch_tokens(student_patch_mosaic)
                student_patch_for_gld = student_patch_quadrants.reshape(
                    b * 4,
                    student_patch_quadrants.size(2),
                    student_patch_quadrants.size(3),
                )

                loss_gld = global_to_local_loss(
                    student_patch=student_patch_for_gld,
                    teacher_cls=teacher_cls_individual,
                    tau=cfg.loss.tau,
                )

                # LLD:
                # Preserve local pairwise structure between teacher and student patch tokens.
                loss_lld = local_to_local_loss(
                    student_patch=student_patch_mosaic,
                    teacher_patch=teacher_patch_mosaic,
                )

                # GGD:
                # Preserve global CLS semantics for individual object-centric images.
                loss_ggd = global_to_global_loss(
                    student_cls=student_cls_individual,
                    teacher_cls=teacher_cls_individual,
                    tau=cfg.loss.tau,
                )

                loss = (
                    cfg.loss.gld_weight * loss_gld
                    + cfg.loss.lld_weight * loss_lld
                    + cfg.loss.ggd_weight * loss_ggd
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach())
            gld_value = float(loss_gld.detach())
            lld_value = float(loss_lld.detach())
            ggd_value = float(loss_ggd.detach())

            epoch_loss += loss_value
            epoch_gld += gld_value
            epoch_lld += lld_value
            epoch_ggd += ggd_value

            if rank == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "gld": f"{gld_value:.4f}",
                        "lld": f"{lld_value:.4f}",
                        "ggd": f"{ggd_value:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

                if global_step % cfg.logs.log_interval == 0:
                    logger.info(
                        f"epoch={epoch} step={step}/{len(loader)} "
                        f"global_step={global_step} "
                        f"loss={loss_value:.6f} "
                        f"gld={gld_value:.6f} "
                        f"lld={lld_value:.6f} "
                        f"ggd={ggd_value:.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6e}"
                    )

                    if use_swanlab:
                        swanlab.log(
                            {
                                "train/loss": loss_value,
                                "train/loss_gld": gld_value,
                                "train/loss_lld": lld_value,
                                "train/loss_ggd": ggd_value,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/epoch": epoch,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )

        num_steps = max(1, len(loader))
        avg_loss = epoch_loss / num_steps
        avg_gld = epoch_gld / num_steps
        avg_lld = epoch_lld / num_steps
        avg_ggd = epoch_ggd / num_steps

        if rank == 0:
            logger.info(
                f"[Epoch {epoch}] "
                f"avg_loss={avg_loss:.6f} "
                f"avg_gld={avg_gld:.6f} "
                f"avg_lld={avg_lld:.6f} "
                f"avg_ggd={avg_ggd:.6f}"
            )

            if use_swanlab:
                swanlab.log(
                    {
                        "epoch/loss": avg_loss,
                        "epoch/loss_gld": avg_gld,
                        "epoch/loss_lld": avg_lld,
                        "epoch/loss_ggd": avg_ggd,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            if epoch % cfg.train.save_interval == 0:
                path = save_checkpoint(
                    save_dir=cfg.train.save_dir,
                    epoch=epoch,
                    student=student_core,
                    teacher=teacher,
                    optimizer=optimizer,
                    scaler=scaler,
                    config_path=args.config,
                    global_step=global_step,
                )
                logger.info(f"Saved checkpoint: {path}")

    if rank == 0:
        final_path = save_checkpoint(
            save_dir=cfg.train.save_dir,
            epoch=cfg.train.epochs,
            student=student_core,
            teacher=teacher,
            optimizer=optimizer,
            scaler=scaler,
            config_path=args.config,
            global_step=global_step,
        )
        logger.info(f"Finished. Final checkpoint: {final_path}")

        if use_swanlab:
            swanlab.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/atas_imagenet.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    args = parser.parse_args()

    train(args)
