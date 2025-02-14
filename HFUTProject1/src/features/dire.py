import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 引入DIRE的扩散模型和工具
from guided_diffusion.guided_diffusion import script_util
from guided_diffusion.guided_diffusion import gaussian_diffusion as gd

def compute_dire(image, diffusion_model, device):
    """
    Compute DIRE feature for a given image.
    """
    diffusion_model.eval()
    with torch.no_grad():
        image = image.to(device)
        seq = list(map(int, np.linspace(0, diffusion_model.num_diffusion_timesteps, diffusion_model.steps + 1)))
        # 使用DDIM的逆过程计算DIRE
        dire = script_util.inversion_first(image, seq, diffusion_model)
        dire = torch.abs(image - dire)  # 计算DIRE
    return dire