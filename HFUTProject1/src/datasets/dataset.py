from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.real_images = os.listdir(os.path.join(root, "real_images"))
        self.generated_images = os.listdir(os.path.join(root, "generated_images"))

    def __len__(self):
        return len(self.real_images) + len(self.generated_images)

    def __getitem__(self, idx):
        if idx < len(self.real_images):
            img_path = os.path.join(self.root, "real_images", self.real_images[idx])
            label = 0  # Real image
        else:
            img_path = os.path.join(self.root, "generated_images", self.generated_images[idx - len(self.real_images)])
            label = 1  # Generated image

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label