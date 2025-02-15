import torch
import torch.nn as nn
from .crossvit import CrossViT
from src.features import DNFExtractor, DIREExtractor, LGradExtractor, SSPExtractor

class FusionModel(nn.Module):
    def __init__(self, dnf_path, dire_path, lgrad_path, ssp_patch_size):
        super(FusionModel, self).__init__()
        self.dnf_extractor = DNFExtractor(dnf_path)
        self.dire_extractor = DIREExtractor(dire_path)
        self.lgrad_extractor = LGradExtractor(lgrad_path)
        self.ssp_extractor = SSPExtractor(ssp_patch_size)
        
        # 定义 CrossViT 融合层
        self.crossvit = CrossViT(embed_dim=256, num_heads=8, depth=3)
        
        # 定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, image):
        # 提取各个特征
        dnf_feat = self.dnf_extractor.compute_dnf(image)
        dire_feat = self.dire_extractor.compute_dire(image)
        lgrad_feat = self.lgrad_extractor.compute_lgrad(image)
        ssp_feat = self.ssp_extractor.compute_ssp(image)
        
        # 融合特征
        fused_feat = self.crossvit(dnf_feat, dire_feat)
        fused_feat = self.crossvit(fused_feat, lgrad_feat)
        fused_feat = self.crossvit(fused_feat, ssp_feat)
        
        # 分类
        output = self.classifier(fused_feat)
        return output