from enum import Enum
import math
from typing import Any, Protocol
import torch
from .had_utils import get_hadamard


class Transform(Protocol):
    def __call__(
        self, w: torch.Tensor, a: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

def smooth_transform(
    w: torch.Tensor, a: torch.Tensor, alpha: float = 0.5, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    a_max = a.abs().max(dim=0).values
    w_max = w.abs().max(dim=0).values.clamp(min=1e-5)
    s_factor = (a_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
    return w * s_factor, a / s_factor


def rotation_tranform(
    w: torch.Tensor, a: torch.Tensor, dev: str = "cpu", random: bool = False, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    n = w.shape[-1]
    w, a = w.to(dev).to(torch.float32), a.to(dev).to(torch.float32)
    r, K = get_hadamard(n)
    if random:
        r *= torch.randint(0, 2, (n,)).float() * 2 - 1
    r = r.to(dev)
    return (w @ r / math.sqrt(K)).half().cpu(), (a @ r / math.sqrt(K)).half().cpu()


def smooth_rotation_transform(
    w: torch.Tensor,
    a: torch.Tensor,
    alpha: float = 0.5,
    dev: str = "cpu",
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    w, a = smooth_transform(w, a, alpha)
    return rotation_tranform(w, a, dev)

class TransformationType(Enum):
    NONE = (lambda w, a, **kwargs: (w, a), "original")
    SMOOTH = (smooth_transform, "smooth")
    ROTATION = (rotation_tranform, "rotation")
    SMOOTH_ROTATION = (smooth_rotation_transform, "smooth_rotation")
