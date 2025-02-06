import torch
import random
import datasets
from transformers import AutoTokenizer


def get_wikitext2_sample(model: str, length: int, random_seed: int = 0) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model)
    data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    input_ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    random.seed(random_seed)
    i = random.randint(0, input_ids.shape[1] - length - 1)
    return input_ids[:, i : i + length]
