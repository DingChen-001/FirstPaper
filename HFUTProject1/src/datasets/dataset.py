from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class GANDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', img_size=256):
        self.paths = self._load_paths(data_root, split)
        self.transform = Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
        ])
    
    def _load_paths(self, root, split):
        real_dir = f"{root}/processed/{split}/real_images"
        gen_dir = f"{root}/processed/{split}/generated_images"
        real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
        gen_paths = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir)]
        return real_paths + gen_paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = 0 if 'real_images' in self.paths[idx] else 1
        return self.transform(img), torch.tensor(label)