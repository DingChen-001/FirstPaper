import torch
from torchvision.utils import save_image
from dnf.diffusion import Model
from dnf.utils import inversion_first, norm

def compute_dnf(image, diffusion_model, device):
    """
    Compute DNF feature for a given image.
    """
    diffusion_model.eval()
    with torch.no_grad():
        image = image.to(device)
        seq = list(map(int, np.linspace(0, diffusion_model.num_diffusion_timesteps, diffusion_model.steps + 1)))
        dnf = inversion_first(image, seq, diffusion_model)
        dnf = norm(dnf)
    return dnf