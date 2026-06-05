#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = ("", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_labeled_path(value):
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label, Path(path)


def read_ascii_ply(path):
    path = Path(path)
    properties = []
    vertex_count = None
    header_lines = 0
    in_vertex = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
            elif stripped.startswith("element ") and not stripped.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and stripped.startswith("property"):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"No vertex element found in {path}")
    if vertex_count == 0:
        return {name: np.empty((0,), dtype=np.float32) for name in properties}

    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(properties):
        raise ValueError(f"PLY column count mismatch in {path}: {data.shape[1]} vs {len(properties)}")
    return {name: data[:, idx] for idx, name in enumerate(properties)}


def find_image(image_dir, image_name):
    base = Path(image_name)
    candidates = []
    if base.suffix:
        candidates.append(image_dir / base.name)
    else:
        candidates.extend(image_dir / f"{base.name}{suffix}" for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def project_points(points, camera, flip_y=False, negative_z=False):
    center = np.asarray(camera["position"], dtype=np.float64)
    rotation_c2w = np.asarray(camera["rotation"], dtype=np.float64)
    world_to_camera_rotation = rotation_c2w.T
    camera_points = (points.astype(np.float64) - center) @ world_to_camera_rotation.T

    z = camera_points[:, 2]
    if negative_z:
        z = -z
    eps = 1e-6
    valid = z > eps

    fx = float(camera["fx"])
    fy = float(camera["fy"])
    width = int(camera["width"])
    height = int(camera["height"])

    u = fx * camera_points[:, 0] / np.where(valid, z, 1.0) + width * 0.5
    if flip_y:
        v = height * 0.5 - fy * camera_points[:, 1] / np.where(valid, z, 1.0)
    else:
        v = fy * camera_points[:, 1] / np.where(valid, z, 1.0) + height * 0.5

    valid &= (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    return u, v, z, valid


def score_colors(scores):
    if scores.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    lo = float(np.nanmin(scores))
    hi = float(np.nanmax(scores))
    if hi <= lo + 1e-12:
        t = np.ones_like(scores, dtype=np.float32)
    else:
        t = ((scores - lo) / (hi - lo)).clip(0.0, 1.0)
    colors = np.zeros((scores.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = 255
    colors[:, 1] = (255.0 * (1.0 - t)).astype(np.uint8)
    colors[:, 2] = 0
    return colors


def draw_overlay(image_path, output_path, u, v, colors, alpha, point_radius, max_points):
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    count = u.shape[0]
    if max_points > 0 and count > max_points:
        order = np.linspace(0, count - 1, max_points, dtype=np.int64)
        u = u[order]
        v = v[order]
        colors = colors[order]

    a = int(255 * alpha)
    r = int(point_radius)
    for x, y, color in zip(u, v, colors):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(int(color[0]), int(color[1]), int(color[2]), a))

    result = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def load_point_cloud(path):
    ply = read_ascii_ply(path)
    for name in ("x", "y", "z"):
        if name not in ply:
            raise ValueError(f"{path} is missing property {name}")
    points = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    if "error_score" in ply:
        scores = ply["error_score"].astype(np.float32)
    elif "error_field_score" in ply:
        scores = ply["error_field_score"].astype(np.float32)
    elif "pred_error" in ply:
        scores = ply["pred_error"].astype(np.float32)
    else:
        scores = np.ones((points.shape[0],), dtype=np.float32)
    return points, scores


def main():
    parser = argparse.ArgumentParser(description="Project 3D error PLY points onto source images.")
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--cameras_json", type=Path)
    parser.add_argument("--iteration", default=5000, type=int)
    parser.add_argument("--point_cloud", action="append", default=[], help="label=/path/to/file.ply")
    parser.add_argument("--max_cameras", default=38, type=int)
    parser.add_argument("--point_radius", default=3, type=int)
    parser.add_argument("--alpha", default=0.65, type=float)
    parser.add_argument("--max_points_per_image", default=30000, type=int)
    parser.add_argument("--flip_y", action="store_true")
    parser.add_argument("--negative_z", action="store_true")
    args = parser.parse_args()

    cameras_json = args.cameras_json or args.model_dir / "cameras.json"
    with cameras_json.open("r", encoding="utf-8") as f:
        cameras = json.load(f)
    if args.max_cameras > 0:
        cameras = cameras[: args.max_cameras]

    point_clouds = [parse_labeled_path(value) for value in args.point_cloud]
    if not point_clouds:
        grid_dir = args.model_dir / "error_field" / "grid_error_map"
        point_clouds = [
            ("grid_error_top5", grid_dir / f"grid_error_top5_{args.iteration}.ply"),
            ("grid_error_top10", grid_dir / f"grid_error_top10_{args.iteration}.ply"),
        ]

    overlay_root = args.model_dir / "error_field" / "image_overlays"
    summary = {
        "model_dir": str(args.model_dir),
        "image_dir": str(args.image_dir),
        "cameras_json": str(cameras_json),
        "iteration": args.iteration,
        "flip_y": args.flip_y,
        "negative_z": args.negative_z,
        "point_clouds": [],
    }

    for label, point_path in point_clouds:
        points, scores = load_point_cloud(point_path)
        colors = score_colors(scores)
        label_summary = {
            "label": label,
            "point_cloud": str(point_path),
            "input_points": int(points.shape[0]),
            "views": [],
        }
        out_dir = overlay_root / label

        for idx, camera in enumerate(cameras):
            image_name = camera.get("img_name") or camera.get("image_name") or camera.get("name")
            image_path = find_image(args.image_dir, image_name)
            view_summary = {"camera_index": idx, "image_name": image_name}
            if image_path is None:
                view_summary["status"] = "missing_image"
                label_summary["views"].append(view_summary)
                continue

            u, v, _, valid = project_points(points, camera, args.flip_y, args.negative_z)
            projected = int(valid.sum())
            view_summary["projected_points"] = projected
            output_path = out_dir / f"{idx:03d}_{Path(image_name).stem}_{label}.png"
            view_summary["output_path"] = str(output_path)

            if projected > 0:
                draw_overlay(
                    image_path,
                    output_path,
                    u[valid],
                    v[valid],
                    colors[valid],
                    args.alpha,
                    args.point_radius,
                    args.max_points_per_image,
                )
                view_summary["status"] = "ok"
            else:
                Image.open(image_path).convert("RGB").save(output_path)
                view_summary["status"] = "no_projected_points"
            label_summary["views"].append(view_summary)

        summary["point_clouds"].append(label_summary)

    overlay_root.mkdir(parents=True, exist_ok=True)
    summary_path = overlay_root / "overlay_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved overlay summary to {summary_path}")


if __name__ == "__main__":
    main()
