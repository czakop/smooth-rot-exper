import torch
from torch import nn
import transformers
from enum import Enum
from typing import Iterator


class ModelType(Enum):
    LLAMA_2_7B = "meta-llama/Llama-2-7b-hf"
    LLAMA_2_13B = "meta-llama/Llama-2-13b-hf"


def load_model(
    model_type: ModelType,
    dtype: torch.dtype = torch.float16,
    hf_token: str | None = None,
) -> torch.nn.Module:
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_type.value, torch_dtype=dtype, token=hf_token
    )


def iter_modules(
    decoder_layer: nn.Module, module_names: list[str]
) -> Iterator[tuple[str, nn.Module]]:
    for module_name in module_names:
        module = get_module(decoder_layer, module_name)
        yield module_name, module


def get_module(decoder_layer: nn.Module, module_name: str) -> nn.Module:
    module: nn.Module | None = getattr(
        decoder_layer.self_attn, module_name, None
    ) or getattr(decoder_layer.mlp, module_name, None)
    assert module, f"Module {module_name} not found in decoder layer"
    return module
