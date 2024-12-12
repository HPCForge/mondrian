import torch


def available_devices():
    if torch.cuda.is_available():
        return ["cpu", "cuda"]
    return ["cpu"]
