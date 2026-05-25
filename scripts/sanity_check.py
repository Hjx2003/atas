"""Sanity checks: import modules and run a tiny forward pass to validate shapes."""
from pathlib import Path
import sys
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parent.parent
src_path = str(repo_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
from src.models.build import build_model
from src.datasets.builders import build_dataset
from src.losses.gld_loss import GlobalLocalDistillationLoss


def make_cfg():
    cfg = SimpleNamespace()
    cfg.dataset = SimpleNamespace()
    cfg.dataset.type = 'imagenet'
    cfg.dataset.root = './data/imagenet'
    cfg.dataset.image_size = 224
    cfg.dataset.collator = 'atas'

    cfg.train = SimpleNamespace()
    cfg.train.batch_size = 4
    cfg.train.num_workers = 0
    cfg.train.epochs = 1
    cfg.train.lr = 1e-4
    cfg.train.save_dir = './checkpoints'
    cfg.train.save_interval = 1

    cfg.model = SimpleNamespace()
    cfg.model.type = 'atas'
    cfg.model.backbone = SimpleNamespace()
    cfg.model.backbone.type = 'openai_clip'
    cfg.model.backbone.patch_grid_size = None

    cfg.loss = SimpleNamespace()
    cfg.loss.ggd_weight = 1.0
    cfg.loss.lld_weight = 0.01
    cfg.loss.gld_weight = 1.0
    cfg.loss.temperature = 1.0

    return cfg


def run():
    cfg = make_cfg()

    # Dummy backbone
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 512
        def forward(self, x):
            b = x.shape[0]
            cls = torch.randn(b, self.dim)
            patch = torch.randn(b, 196, self.dim)
            return {'cls_token': cls, 'patch_tokens': patch}

    model = build_model(cfg, Dummy())
    model.eval()

    # fake batch
    batch = {
        'images': torch.randn(4, 3, cfg.dataset.image_size, cfg.dataset.image_size)
    }

    out = model(batch)
    print('Model forward keys:', list(out.keys()))


if __name__ == '__main__':
    run()
