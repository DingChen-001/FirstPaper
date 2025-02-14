import torch.nn as nn

class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        return self.loss_fn(outputs, labels.float())