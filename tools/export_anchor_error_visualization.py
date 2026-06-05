import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scene.error_field import ErrorField, save_anchor_error_visualization_ply


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export anchor point clouds colored by error-field prediction and residual."
    )
    parser.add_argument("--model_dir", required=True, help="Scaffold-GS output directory.")
    parser.add_argument("--iteration", type=int, default=None, help="Checkpoint iteration. Defaults to error_field checkpoint iteration.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit Scaffold-GS checkpoint path.")
    parser.add_argument("--error_field_checkpoint", default=None, help="Optional explicit error_field_latest.pth path.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory for PLY/JSON files.")
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    error_ckpt_path = args.error_field_checkpoint or os.path.join(
        model_dir, "error_field", "error_field_latest.pth"
    )
    error_state = torch.load(error_ckpt_path, map_location="cpu")

    iteration = args.iteration
    if iteration is None:
        iteration = int(error_state.get("iteration", -1))
    if iteration < 0:
        raise ValueError("Could not infer iteration; pass --iteration explicitly.")

    checkpoint_path = args.checkpoint or os.path.join(model_dir, "chkpnt{}.pth".format(iteration))
    output_dir = args.output_dir or os.path.join(model_dir, "error_field", "anchor_visualization")

    model_state, checkpoint_iter = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(model_state, dict):
        raise ValueError("Unsupported checkpoint format: expected dict model state.")

    required = [
        "anchor",
        "anchor_residue_num_ema",
        "anchor_residue_den_ema",
        "anchor_residue_seen",
    ]
    missing = [name for name in required if name not in model_state]
    if missing:
        raise KeyError("Checkpoint is missing required residue tensors: {}".format(", ".join(missing)))

    error_field = ErrorField(
        num_freqs=int(error_state.get("num_freqs", 6)),
        hidden_dim=int(error_state.get("hidden_dim", 64)),
    )
    error_field.load_state_dict(error_state["model_state"])

    summary = save_anchor_error_visualization_ply(
        error_field=error_field,
        anchor_pos=model_state["anchor"].float(),
        residue_num_ema=model_state["anchor_residue_num_ema"].float(),
        residue_den_ema=model_state["anchor_residue_den_ema"].float(),
        residue_seen=model_state["anchor_residue_seen"].float(),
        bbox_min=error_state["bbox_min"].float(),
        bbox_max=error_state["bbox_max"].float(),
        output_dir=output_dir,
        iteration=iteration,
        eps=args.eps,
    )
    summary["model_dir"] = model_dir
    summary["checkpoint_path"] = checkpoint_path
    summary["checkpoint_iteration"] = int(checkpoint_iter)
    summary["error_field_checkpoint_path"] = error_ckpt_path
    summary["error_field_iteration"] = int(error_state.get("iteration", -1))

    summary_path = summary.get("summary_path")
    if summary_path:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
