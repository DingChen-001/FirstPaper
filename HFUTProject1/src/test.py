import torch
from torch.utils.data import DataLoader
from src.datasets.dataset import CustomDataset
from src.datasets.preprocess import get_transforms
from src.features.dire import compute_dire
from src.features.dnf import compute_dnf
from src.models.crossvit import CrossViT
from src.fusion_layers.dire_fusion import DIRE_Fusion
from src.fusion_layers.dnf_fusion import DNF_Fusion

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms()
    dataset = CustomDataset(root="data/test", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    crossvit_model = CrossViT(num_classes=1).to(device)
    dire_model = script_util.Model().to(device)  # 加载DIRE模型
    dnf_model = Model().to(device)  # 加载DNF模型
    dire_fusion_model = DIRE_Fusion(crossvit_model).to(device)
    dnf_fusion_model = DNF_Fusion(crossvit_model).to(device)

    dire_fusion_model.load_state_dict(torch.load("models/trained/dire_model.pth"))
    dnf_fusion_model.load_state_dict(torch.load("models/trained/dnf_model.pth"))

    dire_fusion_model.eval()
    dnf_fusion_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            dire_features = compute_dire(images, dire_model, device)
            dnf_features = compute_dnf(images, dnf_model, device)

            dire_fused = dire_fusion_model(dire_features)
            dnf_fused = dnf_fusion_model(dnf_features)

            outputs = dire_fused + dnf_fused
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test()