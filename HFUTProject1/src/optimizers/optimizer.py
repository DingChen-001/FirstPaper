import torch.optim as optim

def get_optimizer(params, lr=1e-4, optimizer_type="adam"):
    if optimizer_type == "adam":
        return optim.Adam(params, lr=lr)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")