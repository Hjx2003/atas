# atas
本项目用于复现 ATAS 训练流程：基于预训练 CLIP，使用 ImageNet 无标注图像进行自蒸馏训练。训练阶段冻结 teacher CLIP，仅更新 student 的 image encoder，并使用 GLD、LLD、GGD 三个损失函数进行优化。
