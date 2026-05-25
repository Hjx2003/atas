from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    save_dir,
    epoch,
    student,
    teacher,
    optimizer,
    scaler,
    config_path,
    global_step=0,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f"atas_epoch_{epoch}.pt"
    student_core = unwrap_model(student)

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "student": student_core.state_dict(),
        "teacher": teacher.state_dict() if teacher is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config_path": config_path,
    }

    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(
    path,
    student,
    teacher=None,
    optimizer=None,
    scaler=None,
    strict=False,
    device="cpu",
):
    ckpt = torch.load(path, map_location=device)
    student_core = unwrap_model(student)

    student_core.load_state_dict(ckpt["student"], strict=strict)

    if teacher is not None and ckpt.get("teacher") is not None:
        teacher.load_state_dict(ckpt["teacher"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    return start_epoch, global_step
