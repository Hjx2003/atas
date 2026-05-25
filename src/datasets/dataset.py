from torchvision import transforms
from torchvision.datasets import ImageFolder


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_imagenet_dataset(root: str, image_size: int):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    return ImageFolder(root=root, transform=transform)


def build_dataset(cfg):
    if cfg.type != "imagenet":
        raise ValueError(f"Unsupported dataset type: {cfg.type}")

    return build_imagenet_dataset(
        root=cfg.root,
        image_size=cfg.image_size,
    )