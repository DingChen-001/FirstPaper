import matplotlib.pyplot as plt

def plot_feature_heatmaps(features_dict):
    """绘制各特征通道的激活热力图"""
    plt.figure(figsize=(15, 10))
    for idx, (name, feat) in enumerate(features_dict.items()):
        # 取第一个批次样本的均值
        avg_feat = feat[0].mean(dim=0).cpu().numpy()
        
        plt.subplot(2, 2, idx+1)
        plt.imshow(avg_feat, cmap='viridis')
        plt.title(f'{name} Feature Map')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('feature_heatmaps.png')
    
class SSIM(nn.Module):
    """
    Structure Similarity Index Metric
    用于DIRE特征的重建质量评估
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window = self._gaussian_window(window_size, sigma)
        self.window_size = window_size
        self.channel = 3  # 假设输入为RGB图像
        
    def _gaussian_window(self, size, sigma):
        """生成高斯卷积核"""
        coords = torch.arange(size).float() - size//2
        g = torch.exp(-(coords**2) / (2*sigma**2))
        g /= g.sum()
        return g.view(1, 1, -1) * g.view(1, -1, 1)  # 2D高斯核
    
    def forward(self, img1, img2):
        # 参数校验
        if img1.size() != img2.size():
            raise ValueError("Input images must have the same dimensions")
        
        # 扩展维度（批量、通道）
        B, C, H, W = img1.size()
        window = self.window.repeat(C, 1, 1, 1).to(img1.device)
        
        # 计算均值
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=C)
        
        # 计算方差与协方差
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=C) - mu1_mu2
        
        # SSIM计算公式
        C1 = (0.01 * 1)**2  # 假设动态范围[0,1]
        C2 = (0.03 * 1)**2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()