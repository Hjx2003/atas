import torch
import torch.nn.functional as F


def make_2x2_mosaic(images: torch.Tensor, output_size: int) -> torch.Tensor:
    """
    Build a 2x2 mosaic from four object-centric images.

    Args:
        images: [B, 4, 3, H, W], normalized CLIP images.
        output_size: final spatial size, usually 224 for ViT-B/16.

    Returns:
        mosaic: [B, 3, output_size, output_size]

    Order:
        images[:, 0] -> top-left
        images[:, 1] -> top-right
        images[:, 2] -> bottom-left
        images[:, 3] -> bottom-right
    """
    if images.ndim != 5:
        raise ValueError(f"images must be [B, 4, 3, H, W], got {tuple(images.shape)}")
    if images.size(1) != 4:
        raise ValueError(f"2x2 mosaic requires exactly 4 images, got {images.size(1)}")

    x1, x2, x3, x4 = images[:, 0], images[:, 1], images[:, 2], images[:, 3]
    top = torch.cat([x1, x2], dim=-1)
    bottom = torch.cat([x3, x4], dim=-1)
    mosaic = torch.cat([top, bottom], dim=-2)

    # If four 224x224 images are concatenated, this compresses 448x448 -> 224x224.
    # Each object occupies one quadrant, matching split_2x2_patch_tokens in the loss.
    mosaic = F.interpolate(
        mosaic,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    return mosaic
