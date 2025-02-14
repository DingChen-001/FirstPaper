import torch
import torch.nn as nn
from models.crossvit import CrossViT

class DNF_Fusion(nn.Module):
    def __init__(self, crossvit_model):
        super(DNF_Fusion, self).__init__()
        self.crossvit = crossvit_model
        self.fusion_layer = nn.Linear(2048, 1024)  # Example fusion layer

    def forward(self, dnf_features):
        crossvit_features = self.crossvit(dnf_features)
        fused_features = self.fusion_layer(torch.cat((dnf_features, crossvit_features), dim=1))
        return fused_features