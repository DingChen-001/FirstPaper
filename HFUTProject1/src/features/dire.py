# src/features/dire.py
class DIREFeature(nn.Module):
    def __init__(self, model_path="models/dire/diffusion_model"):
        super().__init__()
        # 加载预训练扩散模型
        self.pipe = DiffusionPipeline.from_pretrained(model_path)
        self.vae = self.pipe.vae
        self.vae.requires_grad_(False)

        # 多尺度残差计算
        self.res_blocks = nn.ModuleList([
            nn.Conv2d(3, 64, 5, stride=2),
            nn.Conv2d(64, 128, 3, stride=2)
        ])

    def compute_residual(self, x):
        """计算扩散重构残差"""
        # 编码到潜空间
        latents = self.vae.encode(x).latent_dist.sample()
        # 解码重建图像
        recon = self.vae.decode(latents).sample
        return torch.abs(x - recon) * 10  # 放大残差

    def forward(self, x):
        residual = self.compute_residual(x)
        features = []
        for block in self.res_blocks:
            residual = block(residual)
            features.append(F.adaptive_avg_pool2d(residual, 1))
        return torch.cat(features, dim=1)