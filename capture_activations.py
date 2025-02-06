import argparse
from utils import ModelType, load_model, get_wikitext2_sample, capture_act

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture activations of a model"
    )

    parser.add_argument(
        "--model",
        type=ModelType,
        default=ModelType.LLAMA_2_7B,
        help="Model name (default: meta-llama/Llama-2-7b-hf). Other options: meta-llama/Llama-2-13b-hf",
    )
    parser.add_argument(
        "--save_act_path",
        type=str,
        help="Path to save captured activations",
    )
    parser.add_argument(
        "--seqlen",
        default=128,
        type=int,
        help="Sequence length of recorded activations (default: 128)",
    )
    parser.add_argument(
        "--nsamples",
        default=1,
        type=int,
        help="Number of samples to capture (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="List of layer indices to capture activations (e.g. --layers 0 1, default: all layers)",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        type=str,
        help="Hugging Face token",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed (default: 0)",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    model = load_model(args.model, hf_token=args.hf_token)
    model.seqlen = args.seqlen
    model.to(args.device)

    # Get a sample from the Wikitext-2 dataset
    input = get_wikitext2_sample(
        args.model.value, args.seqlen * args.nsamples, args.seed
    )

    # Capture activations
    save_folder = f"{args.save_act_path}/{args.model.name}"
    capture_act(
        model=model,
        input=input,
        save_folder=save_folder,
        layers_to_capture=args.layers,
        modules_to_capture_input=None, # Use default modules to capture input
        dev=args.device,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()