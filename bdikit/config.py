import os
import torch

DEFAULT_DEVICE: str = os.getenv("DEFAULT_DEVICE", default="cpu")


def get_device() -> str:
    if DEFAULT_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return DEFAULT_DEVICE
