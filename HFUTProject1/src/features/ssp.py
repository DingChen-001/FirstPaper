# src/features/ssp.py
class PatchSelector(nn.Module):
    """动态高频补丁选择模块"""
    def __init__(self, patch_size=64, top_k=5):
        super().__init__()
        self.patch_size = patch_size
        self.top_k = top_k
        self.variance_conv = nn.Conv2d(3, 1, patch_size, stride=patch_size//2)

    def forward(self, x):
        # 计算局部方差图
        var_map = self.variance_conv(x**2) - (self.variance_conv(x))**2
        # 选择方差最大的k个区域
        B, _, H, W = var_map.shape
        _, indices = torch.topk(var_map.view(B, -1), self.top_k)
        return indices

class SSPFeature(nn.Module):
    def __init__(self, encoder_dim=128):
        super().__init__()
        self.selector = PatchSelector()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, encoder_dim, 3, padding=1)
        )

    def extract_patches(self, x, indices):
        patches = []
        for b in range(x.size(0)):
            h = (indices[b] // x.size(3)) * (self.selector.patch_size // 2)
            w = (indices[b] % x.size(3)) * (self.selector.patch_size // 2)
            patches.append(x[b:b+1, :, h:h+self.selector.patch_size, w:w+self.selector.patch_size])
        return torch.cat(patches, dim=0)

    def forward(self, x):
        indices = self.selector(x)
        patches = self.extract_patches(x, indices)
        encoded = self.encoder(patches)
        return encoded.mean(dim=[2,3])  # 全局平均