#!/usr/bin/env python3
"""Score historical or prefixed normalized scenarios with a trained flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT as ROOT  # noqa: F401
from crypto_generative.models import ConditionalFlowGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/normalizing_flow/conditional_realnvp_best.pt",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="NPZ containing arrays target_returns [n,120,2] and condition_features [n,14].",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.scenarios, allow_pickle=False) as data:
        trajectories = data["target_returns"]
        conditions = data["condition_features"]
    flow = ConditionalFlowGenerator.load(args.checkpoint, device=args.device)
    log_prob_dim = flow.log_prob(trajectories, conditions, per_dimension=True)
    order = np.argsort(log_prob_dim)
    print("Lowest conditional log-density scenarios (least plausible under the model):")
    for index in order[: min(10, len(order))]:
        print(f"scenario={index} log_prob_per_dim={log_prob_dim[index]:.6f}")


if __name__ == "__main__":
    main()
