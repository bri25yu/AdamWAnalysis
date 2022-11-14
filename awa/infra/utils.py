from numpy import ndarray

from torch import Tensor


TORCH_DEVICE = "cuda"


def to_numpy(t: Tensor) -> ndarray:
    return t.detach().cpu().numpy()
