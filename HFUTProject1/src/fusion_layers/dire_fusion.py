import torch
import torch.nn as nn
from models.crossvit import CrossViT

class DIRE_Fusion(nn.Module):
    def __init__(self, crossvit_model):
        super(DIRE_Fusion, self).__init__()
        self.crossvit = crossvit_model
        self.fusion_layer = nn.Linear(2048, 1024)  # Example fusion layer

    def forward(self, dire_features):
        crossvit_features = self.crossvit(dire_features)
        fused_features = self.fusion_layer(torch.cat((dire_features, crossvit_features), dim=1))
        return fused_features