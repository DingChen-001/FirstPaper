import torch
from torch.utils.data import DataLoader
from src.datasets.dataset import CustomDataset
from src.datasets.preprocess import get_transforms
from src.features.dire import compute_dire
from src.features.dnf import compute_dnf
from src.models.crossvit import CrossViT
from src.fusion_layers.dire_fusion import DIRE_Fusion
from src.fusion_layers.dnf_fusion import DNF_Fusion
from src.losses.feature_loss import FeatureLoss
from src.optimizers.optimizer import get_optimizer

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms()
    dataset = CustomDataset(root="data/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    crossvit_model = CrossViT(num_classes=1).to(device)
    dire_model = script_util.Model().to(device)  # 加载DIRE模型
    dnf_model = Model().to(device)  # 加载DNF模型
    dire_fusion_model = DIRE_Fusion(crossvit_model).to(device)
    dnf_fusion_model = DNF_Fusion(crossvit_model).to(device)

    criterion = FeatureLoss()
    optimizer = get_optimizer(list(dire_fusion_model.parameters()) + list(dnf_fusion_model.parameters()), lr=1e-4)

    for epoch in range(10):
        for images, labels in dataloader:
            dire_features = compute_dire(images, dire_model, device)
            dnf_features = compute_dnf(images, dnf_model, device)

            dire_fused = dire_fusion_model(dire_features)
            dnf_fused = dnf_fusion_model(dnf_features)

            outputs = dire_fused + dnf_fused
            loss = criterion(outputs, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # 保存模型
    torch.save(dire_fusion_model.state_dict(), "models/trained/dire_model.pth")
    torch.save(dnf_fusion_model.state_dict(), "models/trained/dnf_model.pth")

if __name__ == "__main__":
    train()