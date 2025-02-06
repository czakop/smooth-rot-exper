import os
from typing import Iterable, Iterator
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from .model_utils import ModelType, get_module, load_model
from .quant_utils import quant_dequant, get_quant_scale
from .transform_utils import Transform

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times"],
    }
)

DEFAULT_COLORS = ["blue", "green", "red", "orange"]
DEFAULT_LINESTYLES = ["-"] * 4


class ActivationLoader:
    def __init__(self, model_type: ModelType, path: str = "./activations"):
        self.model_type = model_type
        self.path = path

    def load(
        self,
        layer_idx: int,
        module: str,
        input: bool = True,
    ) -> torch.Tensor:
        return torch.load(
            f"{self.path}/{self.model_type.name}/{layer_idx}_{module}_{'input' if input else 'output'}.pt",
            weights_only=True,
        )[0]


class WeightLoader:
    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()

    def load(self, layer_idx: int, module_name: str) -> torch.Tensor:
        decoder_layer = self.model.model.layers[layer_idx]
        module = get_module(decoder_layer, module_name)
        return module.weight.data


class ModuleIterator(Iterable[tuple[int, str, torch.Tensor, torch.Tensor]]):
    def __init__(self, model_type: ModelType, activation_path: str = "./activations"):
        model = load_model(model_type)
        self.n_layers = len(model.model.layers)
        self.w_loader = WeightLoader(model)
        self.a_loader = ActivationLoader(model_type, activation_path)

    def iter(
        self,
        module_names: list[str] | None = None,
        layers: list[int] | None = None,
    ) -> Iterable[tuple[int, str, torch.Tensor, torch.Tensor]]:
        if module_names:
            self.module_names = module_names
        else:
            self.module_names = ["k_proj", "o_proj", "gate_proj", "down_proj"]
        self.layers = layers or range(self.n_layers)
        return self

    def __iter__(self) -> Iterator[tuple[int, str, torch.Tensor, torch.Tensor]]:
        assert self.layers, "No layers specified"
        for layer_idx in self.layers:
            for module_name in self.module_names:
                w = self.w_loader.load(layer_idx, module_name)
                a = self.a_loader.load(layer_idx, module_name, input=True)
                yield layer_idx, module_name, w, a


def get_channel_magnitudes(
    x: torch.Tensor, dim: int = 0, sort: int = 0
) -> torch.Tensor:
    cm = torch.linalg.vector_norm(x.to(torch.float32), ord=2, dim=dim)
    if sort:
        cm = torch.sort(cm, descending=sort > 0).values
    return cm


def get_channel_magnitude_std(x: torch.Tensor) -> float:
    cm = get_channel_magnitudes(x, dim=0)
    return (cm.std()).item()


def get_quant_error(
    x: torch.Tensor,
    bits: int = 8,
    sym: bool = True,
    per_tensor: bool = False,
) -> float:
    q_x = quant_dequant(x, bits=bits, sym=sym, per_tensor=per_tensor)
    return torch.linalg.matrix_norm(x - q_x).pow(2).item()


def get_quant_layer_error(
    w: torch.Tensor,
    x: torch.Tensor,
    bits: int = 8,
    sym: bool = True,
    per_tensor: bool = False,
    device: str = "cpu",
) -> float:
    x_dev, w_dev = x.to(device), w.to(device)
    y = x_dev @ w_dev.T
    q_w = quant_dequant(w, bits=bits, sym=sym, per_tensor=per_tensor).to(device)
    q_a = quant_dequant(x, bits=bits, sym=sym, per_tensor=per_tensor).to(device)
    q_y = q_a @ q_w.T
    return torch.linalg.matrix_norm(y - q_y).pow(2).detach().cpu().item()


def quant_error_estimate(w: torch.Tensor, a: torch.Tensor, bits: int) -> float:
    delta_w = get_quant_scale(w, bits)
    delta_a = get_quant_scale(a, bits)
    return (
        (
            torch.linalg.matrix_norm(w).pow(2) * delta_a.pow(2).sum()
            + torch.linalg.matrix_norm(a).pow(2) * delta_w.pow(2).sum()
        )
        / 12
    ).item()


def get_stats(
    iterator: ModuleIterator,
    transform: Transform | None = None,
    save_path: str | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    if save_path and os.path.exists(save_path):
        return pd.read_csv(save_path)
    df_stats = pd.DataFrame()
    for layer_idx, module_name, w, a in iterator.iter():
        if transform:
            w, a = transform(w, a, dev=device)
        w, a = w.to(torch.float32), a.to(torch.float32)
        new_row = pd.DataFrame(
            {
                "layer": layer_idx,
                "module": module_name,
                "layer_quant_error": get_quant_layer_error(w, a, 4, device=device),
                "a_channel_magnitude_std": get_channel_magnitude_std(a),
                "w_channel_magnitude_std": get_channel_magnitude_std(w),
                "a_norm": torch.linalg.matrix_norm(a).item(),
                "w_norm": torch.linalg.matrix_norm(w).item(),
                "a_abs_mean": a.abs().mean().item(),
                "a_abs_median": a.abs().median().item(),
            },
            index=[0],
        )
        df_stats = pd.concat([df_stats, new_row], ignore_index=True)
    if save_path:
        df_stats.to_csv(save_path, index=False)
    return df_stats


def get_layerwise_stat(df: pd.DataFrame, stat: str, module: str) -> pd.Series:
    return df[df["module"] == module][stat].reset_index(drop=True)


def plot_layerwise_stat(
    df: pd.DataFrame,
    stat: str,
    stat_label: str,
    title: str,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    if not colors:
        colors = DEFAULT_COLORS
    if not linestyles:
        linestyles = DEFAULT_LINESTYLES
    plt.rcParams.update({"font.size": 16})
    _ = plt.figure(figsize=(8, 6))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Layer", fontsize=18)
    plt.ylabel(stat_label, fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    for i, module in enumerate(df["module"].unique()):
        stats = get_layerwise_stat(df, stat, module)
        plt.plot(
            stats,
            label=module,
            color=colors[i],
            linestyle=linestyles[i % len(linestyles)],
        )
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_quant_error(
    dfs: list[pd.DataFrame],
    df_names: list[str],
    module: str,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    if not colors:
        colors = DEFAULT_COLORS
    if not linestyles:
        linestyles = DEFAULT_LINESTYLES
    plt.rcParams.update({"font.size": 16})
    _ = plt.figure(figsize=(8, 6))
    plt.xlabel("Layer", fontsize=18)
    plt.ylabel("Quantization Error", fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for i, (df, name) in enumerate(zip(dfs, df_names)):
        layer_quant_error = get_layerwise_stat(df, "layer_quant_error", module)
        plt.plot(
            layer_quant_error,
            label=name,
            color=colors[i],
            linestyle=linestyles[i % len(linestyles)],
        )

    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_quant_bins(
    x: torch.Tensor,
    quant_bits: list[int],
    colors: list[str] | None = None,
    channel: int | torch.Tensor | None = None,
    save_path: str | None = None,
) -> None:
    if not colors:
        colors = DEFAULT_COLORS
    plt.rcParams.update({"font.size": 20})
    _ = plt.figure(figsize=(8, 6))
    for i, bits in enumerate(quant_bits):
        q_x = quant_dequant(x, bits)
        if channel is None:
            c = q_x.abs().max(dim=1).values.argmax()
        else:
            c = channel
        sorted_values = torch.sort(q_x[c].abs()).values
        plt.plot(sorted_values, label=f"{bits} bits", color=colors[i])
    plt.xlabel("Channel", fontsize=24)
    plt.ylabel("Quantized Value", fontsize=24)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_channel_magnitude_std(
    dfs: list[pd.DataFrame],
    df_names: list[str],
    module: str,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    save_dir: str | None = None,
) -> None:
    if not colors:
        colors = DEFAULT_COLORS
    if not linestyles:
        linestyles = DEFAULT_LINESTYLES
    for x, x_name in [("a", "Activation"), ("w", "Weight")]:
        plt.rcParams.update({"font.size": 16})
        _ = plt.figure(figsize=(8, 6))
        plt.xlabel("Layer", fontsize=18)
        plt.ylabel("Quantization Difficulty", fontsize=18)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for i, (df, name) in enumerate(zip(dfs, df_names)):
            stat = get_layerwise_stat(df, f"{x}_channel_magnitude_std", module)
            plt.plot(
                stat,
                label=name,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
            )
        plt.legend()
        if save_dir:
            plt.savefig(f"{save_dir}/{module}_{x}_channel_magnitude_std.pdf")
        else:
            plt.show()


def plot_activation(
    t: torch.Tensor,
    x_label: str = "Channel",
    y_label: str = "Token",
    save_path: str | None = None,
) -> None:
    plt.rcParams.update({"font.size": 20})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_tokens, hidden_dim = t.shape
    # X corresponds to channels (1..hidden_dim), Y to tokens (1..n_tokens)
    x = np.arange(1, hidden_dim + 1)
    y = np.arange(1, n_tokens + 1)
    X, Y = np.meshgrid(x, y)
    Z = t.abs().cpu().numpy()
    _ = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="coolwarm",
        rstride=1,
        cstride=1,
        linewidth=0.5,
        antialiased=True,
        zorder=1,
        rasterized=True,
    )

    ax.set_xlabel(x_label, fontsize=24, labelpad=10)
    ax.set_ylabel(y_label, fontsize=24, labelpad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
