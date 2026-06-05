import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scene.error_field import ErrorField, query_error_field_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export quantile-thresholded 3D grid error maps from an error field checkpoint."
    )
    parser.add_argument("--model_dir", required=True, help="Scaffold-GS output directory.")
    parser.add_argument("--iteration", type=int, default=None, help="Iteration suffix for output files.")
    parser.add_argument("--error_field_checkpoint", default=None, help="Optional explicit error_field_latest.pth path.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory for PLY/JSON files.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def scalar_to_rgb(values):
    values = torch.clamp(values.float(), 0.0, 1.0)
    red = torch.clamp(1.5 - torch.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = torch.clamp(1.5 - torch.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = torch.clamp(1.5 - torch.abs(4.0 * values - 1.0), 0.0, 1.0)
    return torch.round(torch.stack([red, green, blue], dim=1) * 255.0).to(torch.uint8)


def quantile_summary(values):
    return {
        str(q): float(torch.quantile(values.view(-1), q).detach().item())
        for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
    }


def write_score_ply(path, xyz, score):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz_cpu = xyz.detach().cpu().float()
    score_cpu = score.detach().cpu().float()
    colors = scalar_to_rgb(score_cpu)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(xyz_cpu.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float error_score\n")
        f.write("end_header\n")
        for p, c, s in zip(xyz_cpu.tolist(), colors.tolist(), score_cpu.tolist()):
            f.write(
                "{:.8f} {:.8f} {:.8f} {} {} {} {:.8f}\n".format(
                    p[0], p[1], p[2], c[0], c[1], c[2], s
                )
            )


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

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    error_field = ErrorField(
        num_freqs=int(error_state.get("num_freqs", 6)),
        hidden_dim=int(error_state.get("hidden_dim", 64)),
    ).to(device)
    error_field.load_state_dict(error_state["model_state"])

    bbox_min = error_state["bbox_min"].float().to(device)
    bbox_max = error_state["bbox_max"].float().to(device)
    xyz, score = query_error_field_grid(
        error_field=error_field,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=args.resolution,
        chunk=args.chunk,
        device=device,
    )

    score_cpu = score.detach().cpu()
    xyz_cpu = xyz.detach().cpu()
    output_dir = args.output_dir or os.path.join(model_dir, "error_field", "grid_error_map")
    os.makedirs(output_dir, exist_ok=True)

    exports = {}
    thresholds = {}
    for label, q in (("top5", 0.95), ("top10", 0.90)):
        threshold = torch.quantile(score_cpu.view(-1), q)
        mask = score_cpu >= threshold
        path = os.path.join(output_dir, "grid_error_{}_{}.ply".format(label, iteration))
        write_score_ply(path, xyz_cpu[mask], score_cpu[mask])
        exports[label] = path
        thresholds[label] = {
            "quantile": float(q),
            "threshold": float(threshold.item()),
            "point_count": int(mask.sum().item()),
        }

    summary = {
        "iteration": int(iteration),
        "model_dir": model_dir,
        "error_field_checkpoint_path": error_ckpt_path,
        "error_field_iteration": int(error_state.get("iteration", -1)),
        "resolution": int(args.resolution),
        "grid_point_count": int(score_cpu.numel()),
        "score_min": float(score_cpu.min().item()),
        "score_mean": float(score_cpu.mean().item()),
        "score_max": float(score_cpu.max().item()),
        "score_std": float(score_cpu.std(unbiased=False).item()),
        "score_quantiles": quantile_summary(score_cpu),
        "thresholds": thresholds,
        "bbox_min": [float(x) for x in error_state["bbox_min"].float().tolist()],
        "bbox_max": [float(x) for x in error_state["bbox_max"].float().tolist()],
        "outputs": exports,
    }
    summary_path = os.path.join(output_dir, "grid_error_map_summary_{}.json".format(iteration))
    summary["summary_path"] = summary_path
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
