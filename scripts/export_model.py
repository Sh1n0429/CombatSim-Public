"""Export an RLlib checkpoint as a safe, portable model."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from combatsim.observation_schema import OBSERVATION_VERSION
from combatsim.tournament import MODEL_FORMAT
from combatsim.checkpoints import read_policy_checkpoint, validate_checkpoint_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", required=True,
                        help="Anonymous ID used in the ranking, for example model-07")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    validate_checkpoint_schema(checkpoint)
    # This pickle is read only on a trusted local machine. The organizer receives
    # model.pth, which is later loaded with weights_only=True and executes no model code.
    weights, _, _ = read_policy_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.pth"
    torch.save(weights, model_path)
    training_state_path = checkpoint / "training-state.json"
    training_state = (json.loads(training_state_path.read_text(encoding="utf-8"))
                      if training_state_path.exists() else {})
    metadata = {
        "format": MODEL_FORMAT,
        "model_id": args.model_id,
        "observation_version": OBSERVATION_VERSION,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_iteration": training_state.get("training_iteration"),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {args.model_id}: {output}")


if __name__ == "__main__":
    main()
