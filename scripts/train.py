"""Train the pilot-starved receiver; run from the repository root."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/receiver.pt"))
    args = parser.parse_args()

    import torch
    from lightweight_receiver.config import SimConfig
    from lightweight_receiver.training import train
    from lightweight_receiver.metrics import parameter_count

    cfg = SimConfig(num_examples=args.num_examples, epochs=args.epochs,
                    batch_size=args.batch_size, seed=args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = train(cfg, device)
    config = asdict(cfg)
    # Store plain dtype names so the checkpoint supports weights-only loading.
    config["tf_rdtype"] = cfg.tf_rdtype.name
    config["tf_cdtype"] = cfg.tf_cdtype.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "model_state_dict": model.state_dict()}, args.output)
    print(f"Saved {args.output} ({parameter_count(model):,} parameters)")


if __name__ == "__main__":
    main()
