#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData


def resolve_ply(path):
    path = Path(path)
    if path.is_file():
        return path
    point_cloud = path / "point_cloud"
    if not point_cloud.exists():
        raise FileNotFoundError(f"No point_cloud directory under {path}")
    iterations = []
    for child in point_cloud.iterdir():
        match = re.fullmatch(r"iteration_(\d+)", child.name)
        if match and (child / "point_cloud.ply").exists():
            iterations.append((int(match.group(1)), child / "point_cloud.ply"))
    if not iterations:
        raise FileNotFoundError(f"No point_cloud.ply found under {point_cloud}")
    return max(iterations)[1]


def read_cfg_args(model_path):
    cfg_path = Path(model_path) / "cfg_args"
    if not cfg_path.exists():
        return {}
    text = cfg_path.read_text(errors="ignore")
    payload = {}
    for key in ("voxel_size", "n_offsets", "tree_candidate_expand_mode", "tree_candidate_expand_ratio"):
        match = re.search(rf"{key}=([^,\)]+)", text)
        if match:
            raw = match.group(1).strip()
            try:
                payload[key] = ast.literal_eval(raw)
            except Exception:
                payload[key] = raw.strip("'\"")
    return payload


def quantile_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    qs = np.quantile(values, [0.01, 0.05, 0.5, 0.9, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p50": float(qs[2]),
        "p90": float(qs[3]),
        "p95": float(qs[4]),
        "p99": float(qs[5]),
        "max": float(values.max()),
    }


def load_tree_stats(ply_path):
    stats_path = ply_path.with_name(ply_path.stem + "_tree_stats.json")
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Diagnose Scaffold-GS anchor offset and tree growth scale.")
    parser.add_argument("path", help="Model output directory or point_cloud.ply path")
    parser.add_argument("--voxel_size", type=float, default=None, help="Override voxel size for normalized stats")
    parser.add_argument("--json", action="store_true", help="Print compact JSON only")
    args = parser.parse_args()

    ply_path = resolve_ply(args.path)
    model_path = ply_path.parents[2] if ply_path.parent.name.startswith("iteration_") else ply_path.parent
    cfg = read_cfg_args(model_path)
    voxel_size = args.voxel_size if args.voxel_size is not None else cfg.get("voxel_size")

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names
    offset_names = sorted([n for n in names if n.startswith("f_offset_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    scale_names = sorted([n for n in names if n.startswith("scale_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    if len(offset_names) % 3 != 0:
        raise ValueError(f"Offset property count is not divisible by 3: {len(offset_names)}")

    anchors = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    offsets_flat = np.stack([vertex[n] for n in offset_names], axis=1).astype(np.float64)
    n_offsets = len(offset_names) // 3
    offsets = offsets_flat.reshape(anchors.shape[0], 3, n_offsets).transpose(0, 2, 1)
    scales_log = np.stack([vertex[n] for n in scale_names[:3]], axis=1).astype(np.float64)
    scales = np.exp(scales_log)
    offset_world = offsets * scales[:, None, :]
    offset_norm = np.linalg.norm(offsets, axis=2)
    offset_world_norm = np.linalg.norm(offset_world, axis=2)

    result = {
        "ply": str(ply_path),
        "anchor_count": int(anchors.shape[0]),
        "n_offsets": int(n_offsets),
        "cfg": cfg,
        "offset_norm": quantile_stats(offset_norm),
        "offset_world_norm": quantile_stats(offset_world_norm),
        "scaling_xyz": quantile_stats(scales),
        "tree_stats": load_tree_stats(ply_path),
    }
    if voxel_size is not None:
        result["voxel_size"] = float(voxel_size)
        result["offset_world_norm_over_voxel"] = quantile_stats(offset_world_norm / max(float(voxel_size), 1e-12))

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
