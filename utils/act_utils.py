import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
from typing import Iterator, Any
from contextlib import contextmanager
from .model_utils import iter_modules


class ActivationRegistry:
    _input_modules: list[str] = list()
    _output_modules: list[str] = list()
    _input_registry: dict[str, list[torch.Tensor]] = dict()
    _output_registry: dict[str, list[torch.Tensor]] = dict()

    @classmethod
    def set_input_modules(cls, input_modules: list[str]):
        cls._input_modules = input_modules

    @classmethod
    def set_output_modules(cls, output_modules: list[str]):
        cls._output_modules = output_modules

    @classmethod
    def empty(cls) -> None:
        cls._input_registry = {module_name: [] for module_name in cls._input_modules}
        cls._output_registry = {module_name: [] for module_name in cls._output_modules}

    @classmethod
    def register_input(cls, module_name: str, tensor: torch.Tensor):
        cls._input_registry[module_name].append(tensor)

    @classmethod
    def register_output(cls, module_name: str, tensor: torch.Tensor):
        cls._output_registry[module_name].append(tensor)

    @classmethod
    def save(cls, folder: Path | str, prefix: str = "", empty: bool = True):
        if isinstance(folder, str):
            folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        for module_name in cls._input_registry:
            if cls._input_registry[module_name]:
                torch.save(
                    torch.cat(cls._input_registry[module_name], dim=0),
                    folder / f"{prefix}{module_name}_input.pt",
                )
        for module_name in cls._output_registry:
            if cls._output_registry[module_name]:
                torch.save(
                    torch.cat(cls._output_registry[module_name], dim=0),
                    folder / f"{prefix}{module_name}_output.pt",
                )
        if empty:
            cls.empty()


class Catcher(torch.nn.Module):
    def __init__(self, module: nn.Module, input_shape: tuple[int, ...]):
        super().__init__()
        self.module = module
        self.batch_idx = 0
        self.attention_mask = None
        self.position_embeddings = None
        dtype = next(iter(module.parameters())).dtype
        self.register_buffer("inputs", torch.zeros(*input_shape, dtype=dtype))

    def forward(self, inp: torch.Tensor, **kwargs: Any):
        self.inputs[self.batch_idx] = inp
        self.batch_idx += 1
        if not self.attention_mask:
            self.attention_mask = kwargs["attention_mask"]
        if not self.position_embeddings:
            self.position_embeddings = kwargs["position_embeddings"]
        raise ValueError("Stop forward pass")


def hook_factory(module_name: str, capture_input: bool = True):
    def hook(m: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor):
        if capture_input:
            ActivationRegistry.register_input(module_name, input[0].detach().cpu())
        else:
            ActivationRegistry.register_output(module_name, output.detach().cpu())

    return hook


@contextmanager
def capture_activation(
    layer: nn.Module,
    modules_to_capture_input: list[str] | None,
    modules_to_capture_output: list[str] | None = None,
) -> Iterator[None]:
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    if modules_to_capture_input:
        for module_name, module in iter_modules(layer, modules_to_capture_input):
            hook_handles.append(
                module.register_forward_hook(
                    hook_factory(module_name, capture_input=True)
                )
            )
    if modules_to_capture_output:
        for module_name, module in iter_modules(layer, modules_to_capture_output):
            hook_handles.append(
                module.register_forward_hook(
                    hook_factory(module_name, capture_input=False)
                )
            )
    try:
        yield
    finally:
        for h in hook_handles:
            h.remove()


def capture_parameters(
    model: nn.Module, input_ids: torch.Tensor, dev: str, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # Convert the whole text of evaluation dataset into batches of sequences.
    nbatches = input_ids.numel() // model.seqlen // batch_size  # The tail is truncated.
    input_ids = (
        input_ids[:, : nbatches * model.seqlen * batch_size]
        .view(-1, batch_size, model.seqlen)
        .to(dev)
    )  # (nbatches, batch_size, seqlen)

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    first_layer_catcher = Catcher(
        model.model.layers[0],
        (nbatches, batch_size, model.seqlen, model.config.hidden_size),
    )
    model.model.layers[0] = first_layer_catcher.to(dev)

    for batch in input_ids:
        try:
            model(batch)
        except ValueError:
            pass

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.layers[0] = first_layer_catcher.module.cpu()

    torch.cuda.empty_cache()

    return (
        first_layer_catcher.inputs,
        first_layer_catcher.attention_mask,
        first_layer_catcher.position_embeddings,
    )


def capture_layer_act(
    decoder_layer: nn.Module,
    layer_input: torch.Tensor,
    modules_to_capture_input: list[str],
    dev: str,
    **kwargs: Any,
) -> None:
    with capture_activation(
        decoder_layer, modules_to_capture_input, modules_to_capture_output=None
    ):
        # Process each sequence in the batch one by one to avoid OOM.
        nb, bs, sq, em = layer_input.shape
        layer_input = layer_input.view(nb * bs, sq, em)
        for seq_idx in range(layer_input.shape[0]):
            # Extract the current sequence across all dimensions.
            seq = layer_input[seq_idx : seq_idx + 1].to(dev)
            decoder_layer(seq, **kwargs)


@torch.no_grad()
def capture_act(
    model: nn.Module,
    input: torch.Tensor,
    save_folder: Path | str,
    layers_to_capture: list[int] | None = None,
    modules_to_capture_input: list[str] | None = None,
    dev: str = "cpu",
    batch_size: int = 1,
) -> None:
    model.eval()
    model.config.use_cache = False

    layer_input, attention_mask, position_embeddings = capture_parameters(
        model, input, dev, batch_size
    )
    layers = model.model.layers

    if layers_to_capture is None:
        layers_to_capture = list(range(len(model.model.layers)))
    if modules_to_capture_input is None:
        modules_to_capture_input = ["k_proj", "o_proj", "gate_proj", "down_proj"]

    # No support for capturing output activations for now
    ActivationRegistry.set_input_modules(modules_to_capture_input)
    ActivationRegistry.empty()

    layer_output = torch.zeros_like(layer_input)
    for i in tqdm(range(len(layers)), desc="Capturing activations"):
        if i > max(layers_to_capture):
            break
        layer = layers[i].to(dev)
        if i in layers_to_capture:
            capture_layer_act(
                layer,
                layer_input,
                modules_to_capture_input,
                dev,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            ActivationRegistry.save(save_folder, prefix=f"{i}_", empty=True)
        for j, batch in enumerate(layer_input):
            layer_output[j] = layer(
                batch,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        layer_input, layer_output = layer_output, layer_input
