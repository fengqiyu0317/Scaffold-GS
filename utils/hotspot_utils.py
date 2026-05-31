import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def _luma(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _edge_response(gray):
    dx = F.pad(torch.abs(gray[:, 1:] - gray[:, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[1:, :] - gray[:-1, :]), (0, 0, 0, 1))
    return torch.maximum(dx, dy)


def _percentile_threshold(values, percentile):
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(0.0, dtype=values.dtype, device=values.device)
    k = int(math.ceil((float(percentile) / 100.0) * flat.numel()))
    k = max(1, min(k, flat.numel()))
    return torch.kthvalue(flat, k).values


def _project_points(points, camera, height, width):
    if points.numel() == 0:
        return None, None, None
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)
    clip = points_h @ camera.full_proj_transform
    denom = clip[:, 3]
    ndc = clip[:, :3] / (denom[:, None] + 1e-7)
    px = torch.round((ndc[:, 0] + 1.0) * 0.5 * (width - 1)).long()
    py = torch.round((1.0 - ndc[:, 1]) * 0.5 * (height - 1)).long()
    valid = (
        (denom > 1e-6)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
        & torch.isfinite(ndc).all(dim=1)
    )
    return px, py, valid


def _view_diverse(view_dirs, min_angle_deg):
    dirs = [np.asarray(v, dtype=np.float32) for v in view_dirs]
    if len(dirs) < 2:
        return False, 0.0
    max_angle = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            dot = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
            max_angle = max(max_angle, angle)
            if angle >= float(min_angle_deg):
                return True, max_angle
    return False, max_angle


class ErrorHotspotField:
    def __init__(self, opt, voxel_size):
        self.voxel_size = float(voxel_size) * float(opt.hotspot_voxel_multiplier)
        self.start = int(opt.hotspot_start)
        self.until = int(opt.hotspot_until)
        self.update_interval = int(opt.hotspot_update_interval)
        self.grow_interval = int(opt.hotspot_grow_interval)
        self.max_pixels_per_view = int(opt.hotspot_max_pixels_per_view)
        self.reproj_radius_px = float(opt.hotspot_reproj_radius_px)
        self.min_support_views = int(opt.hotspot_min_support_views)
        self.min_view_angle_deg = float(opt.hotspot_min_view_angle_deg)
        self.error_multiplier = float(opt.hotspot_error_mean_multiplier)
        self.score_clip = float(opt.hotspot_score_clip)
        self.high_error_percentile = float(opt.hotspot_high_error_percentile)
        self.highlight_luma_threshold = float(opt.hotspot_highlight_luma_threshold)
        self.highlight_deficit_threshold = float(opt.hotspot_highlight_deficit_threshold)
        self.thin_luma_threshold = float(opt.hotspot_thin_luma_threshold)
        self.thin_chroma_max = float(opt.hotspot_thin_chroma_max)
        self.edge_threshold = float(opt.hotspot_edge_threshold)
        self.min_anchor_count = int(opt.hotspot_min_anchor_count)
        self.add_min_votes = int(opt.hotspot_add_min_votes)
        self.candidate_multiplier = int(opt.hotspot_add_candidate_multiplier)
        self.stats = defaultdict(self._new_stat)
        self.global_error_sum = 0.0
        self.global_error_pixels = 0
        self.sampled_pixels = 0
        self.matched_pixels = 0
        self.last_active = []
        self.last_summary = {}

    @staticmethod
    def _new_stat():
        return {
            "error_sum": 0.0,
            "rgb_error_sum": 0.0,
            "count": 0,
            "support_views": set(),
            "view_dirs": {},
            "last_seen_iter": 0,
            "mask_counts": Counter(),
        }

    def _key_tensor(self, xyz):
        return torch.floor(xyz.detach().float() / self.voxel_size).long()

    def _key_tuple(self, xyz_np):
        return tuple(np.floor(xyz_np / self.voxel_size).astype(np.int64).tolist())

    def _center(self, key):
        return [(float(k) + 0.5) * self.voxel_size for k in key]

    def should_update(self, iteration):
        return self.start <= iteration <= self.until and iteration % max(self.update_interval, 1) == 0

    def should_grow(self, iteration):
        return self.start <= iteration <= self.until and iteration % max(self.grow_interval, 1) == 0

    def update(self, iteration, viewpoint_camera, render_image, gt_image, neural_xyz, visibility_filter):
        if neural_xyz is None or neural_xyz.numel() == 0:
            return
        rgb_error = torch.abs(render_image.detach() - gt_image.detach()).mean(dim=0)
        gt_luma = _luma(gt_image.detach())
        render_luma = _luma(render_image.detach())
        luma_deficit = torch.relu(gt_luma - render_luma)
        high_error = rgb_error > _percentile_threshold(rgb_error, self.high_error_percentile)
        highlight = (gt_luma > self.highlight_luma_threshold) & (luma_deficit > self.highlight_deficit_threshold)
        chroma = gt_image.detach().max(dim=0).values - gt_image.detach().min(dim=0).values
        edges = _edge_response(gt_luma)
        thin_bright = (gt_luma > self.thin_luma_threshold) & (chroma < self.thin_chroma_max) & (edges > self.edge_threshold)
        score_map = rgb_error + highlight.float() * luma_deficit * 2.0 + thin_bright.float() * edges * 1.5
        self.global_error_sum += float(rgb_error.sum().item())
        self.global_error_pixels += int(rgb_error.numel())

        height, width = rgb_error.shape
        active = visibility_filter.detach()
        if active.sum() == 0:
            return
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        xyz = neural_xyz.detach()[active]
        px, py, valid = _project_points(xyz, viewpoint_camera, height, width)
        if valid is None or valid.sum() == 0:
            return
        px = px[valid]
        py = py[valid]
        active_idx = active_idx[valid]
        xyz = xyz[valid]
        mask_hit = high_error[py, px] | highlight[py, px] | thin_bright[py, px]
        if mask_hit.sum() == 0:
            return
        hit_ids = torch.nonzero(mask_hit, as_tuple=False).squeeze(1)
        hit_scores = score_map[py[hit_ids], px[hit_ids]]
        k = min(self.max_pixels_per_view, hit_ids.numel())
        if hit_ids.numel() > k:
            hit_ids = hit_ids[torch.topk(hit_scores, k=k).indices]
        self.sampled_pixels += int(k)
        cam_center = viewpoint_camera.camera_center.detach().cpu().numpy()
        view_name = str(viewpoint_camera.image_name)
        selected_xyz = xyz[hit_ids].detach().cpu().numpy()
        selected_px = px[hit_ids]
        selected_py = py[hit_ids]
        for local_i, point in enumerate(selected_xyz):
            x = int(selected_px[local_i].item())
            y = int(selected_py[local_i].item())
            mask_code = 0
            if bool(high_error[y, x].item()):
                mask_code |= 1
            if bool(highlight[y, x].item()):
                mask_code |= 2
            if bool(thin_bright[y, x].item()):
                mask_code |= 4
            key = self._key_tuple(point)
            stat = self.stats[key]
            stat["error_sum"] += float(score_map[y, x].item())
            stat["rgb_error_sum"] += float(rgb_error[y, x].item())
            stat["count"] += 1
            stat["support_views"].add(view_name)
            stat["last_seen_iter"] = int(iteration)
            stat["mask_counts"].update([str(mask_code)])
            if view_name not in stat["view_dirs"]:
                direction = cam_center - point
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    stat["view_dirs"][view_name] = (direction / norm).astype(float).tolist()
            self.matched_pixels += 1

    def _anchor_key_counts(self, anchor_xyz):
        keys = self._key_tensor(anchor_xyz).detach().cpu().numpy()
        return Counter(tuple(row.tolist()) for row in keys)

    def rebuild_active(self, anchor_xyz=None):
        anchor_counts = self._anchor_key_counts(anchor_xyz) if anchor_xyz is not None else Counter()
        global_error = self.global_error_sum / max(self.global_error_pixels, 1)
        active = []
        for key, stat in self.stats.items():
            support_views = len(stat["support_views"])
            mean_rgb = stat["rgb_error_sum"] / max(stat["count"], 1)
            diverse, max_angle = _view_diverse(stat["view_dirs"].values(), self.min_view_angle_deg)
            is_active = support_views >= self.min_support_views and diverse and mean_rgb >= global_error * self.error_multiplier
            if not is_active:
                continue
            raw_score = mean_rgb / max(global_error, 1e-6)
            active.append({
                "key": key,
                "center": self._center(key),
                "score": min(raw_score, self.score_clip),
                "support_views": support_views,
                "mean_rgb_error": mean_rgb,
                "max_view_angle_deg": max_angle,
                "near_anchor_count": int(anchor_counts.get(key, 0)),
                "count": int(stat["count"]),
                "last_seen_iter": int(stat["last_seen_iter"]),
            })
        active.sort(key=lambda item: (-item["score"], -item["support_views"], item["near_anchor_count"]))
        self.last_active = active
        self.last_summary = {
            "hotspot_voxels": len(self.stats),
            "active_hotspot_voxels": len(active),
            "sampled_pixels": self.sampled_pixels,
            "matched_pixels": self.matched_pixels,
            "global_mean_rgb_error": global_error,
        }
        return active

    def query_anchor_scores(self, anchor_xyz):
        active = self.rebuild_active(anchor_xyz)
        scores = torch.zeros((anchor_xyz.shape[0],), dtype=torch.float32, device=anchor_xyz.device)
        if not active:
            return scores
        active_score = {item["key"]: float(item["score"]) for item in active}
        anchor_keys = self._key_tensor(anchor_xyz).detach().cpu().numpy()
        values = [active_score.get(tuple(row.tolist()), 0.0) for row in anchor_keys]
        return torch.tensor(values, dtype=torch.float32, device=anchor_xyz.device)

    def propose_candidates(self, anchor_xyz, budget):
        active = self.rebuild_active(anchor_xyz)
        low_density = [item for item in active if item["near_anchor_count"] < self.min_anchor_count and item["count"] >= self.add_min_votes]
        if not low_density:
            return None, None, {"active": len(active), "density_pass": 0, "proposed": 0}
        limit = min(len(low_density), int(budget) * max(self.candidate_multiplier, 1))
        selected = low_density[:limit]
        anchor_keys = self._key_tensor(anchor_xyz).detach().cpu().numpy()
        key_to_first_anchor = {}
        for idx, row in enumerate(anchor_keys):
            key = tuple(row.tolist())
            if key not in key_to_first_anchor:
                key_to_first_anchor[key] = idx
        centers = []
        parents = []
        for item in selected:
            key = item["key"]
            parent = key_to_first_anchor.get(key)
            if parent is None:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            parent = key_to_first_anchor.get((key[0] + dx, key[1] + dy, key[2] + dz))
                            if parent is not None:
                                break
                        if parent is not None:
                            break
                    if parent is not None:
                        break
            if parent is None:
                continue
            centers.append(item["center"])
            parents.append(parent)
            if len(centers) >= int(budget):
                break
        if not centers:
            return None, None, {"active": len(active), "density_pass": len(low_density), "proposed": 0}
        return (
            torch.tensor(centers, dtype=torch.float32, device=anchor_xyz.device),
            torch.tensor(parents, dtype=torch.long, device=anchor_xyz.device),
            {"active": len(active), "density_pass": len(low_density), "proposed": len(centers)},
        )

    def write_summary(self, output_dir, iteration):
        os.makedirs(output_dir, exist_ok=True)
        payload = dict(self.last_summary)
        payload["iteration"] = int(iteration)
        payload["top_active"] = self.last_active[:50]
        path = os.path.join(output_dir, f"hotspot_stats_{int(iteration):06d}.json")
        with open(path, "w") as fp:
            json.dump(payload, fp, indent=2)
        return path
