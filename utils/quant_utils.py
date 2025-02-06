import torch


def get_quant_scale(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    return (x.max(dim=1).values - x.min(dim=1).values) / (2**bits - 1)


def quant_dequant(
    x: torch.Tensor,
    bits: int = 4,
    sym: bool = True,
    per_tensor: bool = False,
):
    if bits == 16:
        return x

    assert 0 < bits < 16, "Number of bits should be in (0, 16)"

    init_shape = x.shape
    if per_tensor:
        reshaped_x = x.reshape(-1, x.shape[-2] * x.shape[-1])
    else:
        reshaped_x = x.reshape(-1, x.shape[-1])

    tmp = torch.zeros(reshaped_x.shape[0])
    xmin = torch.minimum(reshaped_x.min(1)[0], tmp)
    xmax = torch.maximum(reshaped_x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmax == 0
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
        scale = (xmax / maxq).unsqueeze(1)
        scale[tmp] = 1
        zero = torch.zeros_like(scale)
    else:
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp], xmax[tmp] = -1, 1
        maxq = torch.tensor(2**bits - 1)
        minq = torch.zeros(1)
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        scale = scale.unsqueeze(1)
        zero = zero.unsqueeze(1)

    q = torch.clamp(torch.round(reshaped_x / scale) + zero, minq, maxq)
    return (scale * (q - zero)).reshape(init_shape).half()
