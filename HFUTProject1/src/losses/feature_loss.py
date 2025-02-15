import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFeatureLoss(nn.Module):
    """
    多尺度特征对比损失（用于预训练特征提取器）
    论文参考：DNF论文公式(5)的改进版本
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, feat_real, feat_fake):
        """
        输入: 
            feat_real: 真实图像特征 [B, D]
            feat_fake: 生成图像特征 [B, D]
        输出: 
            对比损失值
        """
        # 拼接所有特征
        features = torch.cat([feat_real, feat_fake], dim=0)  # [2B, D]
        
        # 计算相似度矩阵
        sim_matrix = self.cos_sim(features.unsqueeze(1), features.unsqueeze(0))  # [2B, 2B]
        sim_matrix /= self.temperature
        
        # 构建标签（对角线为匹配样本）
        labels = torch.arange(features.size(0)).to(features.device)
        
        # 交叉熵损失（对角线为正样本）
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class ReconstructionLoss(nn.Module):
    """DIRE特征专用的重建损失（L1 + SSIM）"""
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIM(window_size=11)  # 需实现SSIM计算

    def forward(self, original, reconstructed):
        l1_loss = F.l1_loss(original, reconstructed)
        ssim_loss = 1 - self.ssim(original, reconstructed)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss