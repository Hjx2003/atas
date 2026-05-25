import torch
import open_clip


class CLIPVisionWrapper(torch.nn.Module):

    def __init__(
        self,
        model_name="ViT-B-16",
        pretrained_path=None,
    ):
        super().__init__()

        # 只创建模型结构
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=None,
        )

        # 从本地加载 checkpoint
        if pretrained_path is not None:

            print(f"Loading pretrained from: {pretrained_path}")

            ckpt = torch.load(
                pretrained_path,
                map_location="cpu",
            )

            # 兼容不同 checkpoint 格式
            if isinstance(ckpt, dict):

                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]

                elif "model" in ckpt:
                    ckpt = ckpt["model"]

                elif "student" in ckpt:
                    ckpt = ckpt["student"]

            msg = model.load_state_dict(
                ckpt,
                strict=False,
            )

            print(msg)

        self.model = model
        self.visual = model.visual

    def forward_features(self, x):
        visual = self.visual

        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0],
                    1,
                    visual.class_embedding.shape[0],
                    dtype=x.dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )

        x = x + visual.positional_embedding.to(x.dtype)

        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)

        x = visual.ln_pre(x)

        if getattr(visual.transformer, "batch_first", False):
            x = visual.transformer(x)
        else:
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)

        cls = x[:, 0]
        patch = x[:, 1:]

        cls = visual.ln_post(cls)
        patch = visual.ln_post(patch)

        if visual.proj is not None:
            cls = cls @ visual.proj
            patch = patch @ visual.proj

        return cls, patch