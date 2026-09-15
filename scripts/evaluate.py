"""Evaluate a trained receiver against the classical OFDM baseline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/receiver.pt"))
    parser.add_argument("--num-examples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ebn0-db", type=float, nargs="+", default=list(range(13)))
    parser.add_argument("--seed", type=int, default=None, help="Override checkpoint data seed.")
    parser.add_argument("--seed-noise", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    import tensorflow as tf
    import torch
    from lightweight_receiver.config import SimConfig
    from lightweight_receiver.models import build_model
    from lightweight_receiver.evaluation import evaluate
    from lightweight_receiver.metrics import parameter_count

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    config["tf_rdtype"] = tf.as_dtype(config["tf_rdtype"])
    config["tf_cdtype"] = tf.as_dtype(config["tf_cdtype"])
    config["num_examples"] = args.num_examples
    if args.seed is not None:
        config["seed"] = args.seed
    if args.seed_noise is not None:
        config["seed_noise"] = args.seed_noise
    cfg = SimConfig(**config)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate(model, cfg, args.ebn0_db, args.batch_size, device)
    print(json.dumps({"parameters": parameter_count(model), "results": results}, indent=2))


if __name__ == "__main__":
    main()
