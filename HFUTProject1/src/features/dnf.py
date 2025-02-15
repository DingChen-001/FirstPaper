# src/features/dnf.py
from diffusers import DDIMScheduler, UNet2DModel

class DiffusionNoiseFeature(nn.Module):
    def __init__(self, ddim_config_path="config/ddim_config.yaml"):
        super().__init__()
        # 加载DDIM模型配置
        with open(ddim_config_path) as f:
            config = yaml.safe_load(f)
        
        # 初始化DDIM组件
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config['num_train_timesteps'],
            beta_schedule=config['beta_schedule']
        )
        self.unet = UNet2DModel.from_pretrained(config['model_path'])
        self.unet.requires_grad_(False)  # 冻结参数

        # 逆向过程噪声预测头
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def reverse_process(self, x, steps=100):
        """执行DDIM逆向过程"""
        x_prev = x.clone()
        for t in reversed(range(0, steps)):
            alpha_bar = self.scheduler.alphas_cumprod[t]
            pred_noise = self.unet(x_prev, t).sample
            # DDIM更新规则
            x_prev = (x_prev - (1 - alpha_bar)**0.5 * pred_noise) / alpha_bar**0.5
        return x_prev

    def forward(self, x):
        # 步骤1：执行完整逆向过程
        reversed_x = self.reverse_process(x)
        # 步骤2：计算预测噪声残差
        noise_residual = self.noise_predictor(x - reversed_x)
        return noise_residual.mean(dim=[2,3])  # 全局池化