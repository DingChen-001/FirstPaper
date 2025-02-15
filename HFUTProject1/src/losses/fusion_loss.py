import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalFusionLoss(nn.Module):
    """
    融合模型的多任务损失函数：
    - 分类交叉熵损失（主损失）
    - 特征一致性损失（辅助损失）
    - 注意力稀疏正则（防止过拟合）
    """
    def __init__(self, alpha=0.5, beta=0.1):
        super().__init__()
        self.alpha = alpha  # 一致性损失权重
        self.beta = beta    # 正则项权重
        self.ce_loss = nn.CrossEntropyLoss()
        
    def feature_consistency(self, feat_dict):
        """特征间余弦相似度一致性约束"""
        loss = 0
        keys = list(feat_dict.keys())
        # 计算所有特征对的两两相似度
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                sim = F.cosine_similarity(feat_dict[keys[i]], feat_dict[keys[j]], dim=1)
                loss += torch.var(sim)  # 惩罚相似度波动
        return loss / len(keys)
    
    def attention_sparsity(self, attn_weights):
        """注意力权重稀疏性正则（参考论文公式12）"""
        return torch.mean(torch.sum(attn_weights**2, dim=-1))

    def forward(self, outputs, feat_dict, attn_weights, labels):
        main_loss = self.ce_loss(outputs, labels)
        consist_loss = self.feature_consistency(feat_dict)
        sparsity_loss = self.attention_sparsity(attn_weights)
        
        total_loss = main_loss + self.alpha*consist_loss + self.beta*sparsity_loss
        return {
            "total": total_loss,
            "ce": main_loss,
            "consistency": consist_loss,
            "sparsity": sparsity_loss
        }