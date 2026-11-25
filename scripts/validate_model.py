"""Validate one or more model folders without running model code."""
import argparse
from pathlib import Path

import numpy as np
import torch

from combatsim.env import ACTION_SHAPE
from combatsim.set_transformer import OBSERVATION_SHAPE
from combatsim.tournament import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    seen = set()
    for directory in args.models:
        model_id, network, metadata = load_model(directory)
        if model_id in seen:
            raise ValueError(f"Duplicate model_id: {model_id}")
        seen.add(model_id)
        sample = {key: torch.as_tensor(np.zeros((1, *shape), dtype=np.float32))
                  for key, shape in OBSERVATION_SHAPE.items()}
        sample["self_mask"][:, 0] = 1
        with torch.inference_mode():
            logits, value = network(sample)
        expected = [(1, size) for size in ACTION_SHAPE]
        if [tuple(head.shape) for head in logits] != expected:
            raise ValueError(f"Unexpected output shape for {model_id}")
        print(f"OK: {model_id} | {Path(directory).resolve()} | value={value.item():+.3f}")


if __name__ == "__main__":
    main()
