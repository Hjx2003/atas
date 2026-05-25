import math
import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def contrastive_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    ATAS Eq. (2) / Eq. (4), one-direction contrastive distillation.

    student_feat: [M, D]
    teacher_feat: [M, D]
    """
    student_feat = normalize(student_feat.float())
    teacher_feat = normalize(teacher_feat.float())

    logits = (student_feat @ teacher_feat.t()) / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def global_to_local_loss(
    student_patch: torch.Tensor,
    teacher_cls: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    ATAS Eq. (1) + Eq. (2).

    student_patch: [M, Nq, D]
        Local patch tokens from student mosaic quadrants.
    teacher_cls: [M, D]
        CLS tokens from individual object-centric teacher images.

    Eq. (1):
        w_i^s = sum_j softmax(phi(e_ij^s, c_i^t)/tau) * e_ij^s
    Eq. (2):
        contrastive alignment between w_i^s and c_i^t.
    """
    patch_norm = normalize(student_patch.float())
    cls_norm = normalize(teacher_cls.float()).unsqueeze(1)

    sim = (patch_norm * cls_norm).sum(dim=-1)
    weights = torch.softmax(sim / tau, dim=1).unsqueeze(-1)
    pooled_local = (weights * student_patch).sum(dim=1)

    return contrastive_loss(pooled_local, teacher_cls, tau=tau)


def local_to_local_loss(
    student_patch: torch.Tensor,
    teacher_patch: torch.Tensor,
) -> torch.Tensor:
    """
    ATAS Eq. (3): preserve pairwise local-token similarity structure.

    student_patch: [B, N, D]
    teacher_patch: [B, N, D]
    """
    student_patch = normalize(student_patch)
    teacher_patch = normalize(teacher_patch)

    student_sim = student_patch @ student_patch.transpose(1, 2)
    teacher_sim = teacher_patch @ teacher_patch.transpose(1, 2)

    return F.mse_loss(student_sim, teacher_sim)


def global_to_global_loss(
    student_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    ATAS Eq. (4): preserve teacher CLS/global representation.
    """
    return contrastive_loss(student_cls, teacher_cls, tau=tau)


def split_2x2_patch_tokens(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Split ViT patch tokens from a 2x2 mosaic image into four quadrant groups.

    Args:
        patch_tokens: [B, N, D], where N must be a square number.

    Returns:
        [B, 4, N/4, D] with order:
        top-left, top-right, bottom-left, bottom-right.
    """
    bsz, num_patches, dim = patch_tokens.shape
    grid = int(math.sqrt(num_patches))

    if grid * grid != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square grid")

    if grid % 2 != 0:
        raise ValueError(f"2x2 mosaic requires an even patch grid, got {grid}x{grid}")

    x = patch_tokens.reshape(bsz, grid, grid, dim)
    h = grid // 2

    quadrants = [
        x[:, :h, :h, :],
        x[:, :h, h:, :],
        x[:, h:, :h, :],
        x[:, h:, h:, :],
    ]

    return torch.stack(
        [q.reshape(bsz, h * h, dim) for q in quadrants],
        dim=1,
    )
