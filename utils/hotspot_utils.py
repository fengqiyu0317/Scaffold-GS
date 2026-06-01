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


def _sample_mask_pixels(mask, score_map, max_pixels):
    hit = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
    if hit.numel() == 0:
        return None, None
    k = min(int(max_pixels), int(hit.numel()))
    scores = score_map.reshape(-1)[hit]
    if hit.numel() > k:
        hit = hit[torch.topk(scores, k=k).indices]
    height, width = score_map.shape
    y = torch.div(hit, width, rounding_mode="floor").long()
    x = (hit - y * width).long()
    return x, y


def _view_depth(points, camera):
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)
    view = points_h @ camera.world_view_transform
    return view[:, 2]


def _unproject_pixels_with_view_depth(px, py, depth, camera, height, width):
    ndc_x = (px.float() / max(width - 1, 1)) * 2.0 - 1.0
    ndc_y = 1.0 - (py.float() / max(height - 1, 1)) * 2.0
    view_x = ndc_x * depth * math.tan(float(camera.FoVx) * 0.5)
    view_y = ndc_y * depth * math.tan(float(camera.FoVy) * 0.5)
    view = torch.stack([view_x, view_y, depth, torch.ones_like(depth)], dim=1)
    world = view @ torch.inverse(camera.world_view_transform)
    return world[:, :3] / (world[:, 3:4] + 1e-7)


class ErrorHotspotField:
    def __init__(self, opt, voxel_size):
        self.voxel_size = float(voxel_size) * float(opt.hotspot_voxel_multiplier)
        self.start = int(opt.hotspot_start)
        self.until = int(opt.hotspot_until)
        self.update_interval = int(opt.hotspot_update_interval)
        self.grow_interval = int(opt.hotspot_grow_interval)
        self.max_pixels_per_view = int(opt.hotspot_max_pixels_per_view)
        self.reproj_radius_px = float(opt.hotspot_reproj_radius_px)
        self.attribution_mode = str(getattr(opt, "hotspot_attribution_mode", "center")).lower()
        self.depth_radius_cap_px = int(getattr(opt, "hotspot_depth_radius_cap_px", math.ceil(self.reproj_radius_px)))
        self.depth_min_weight = float(getattr(opt, "hotspot_depth_min_weight", 1e-4))
        self.depth_min_radii = float(getattr(opt, "hotspot_depth_min_radii", 1.0))
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
        self.density_ratio_thresh = float(getattr(opt, "hotspot_density_ratio_thresh", 0.7))
        self.add_min_votes = int(opt.hotspot_add_min_votes)
        self.candidate_multiplier = int(opt.hotspot_add_candidate_multiplier)
        self.stats = defaultdict(self._new_stat)
        self.global_error_sum = 0.0
        self.global_error_pixels = 0
        self.sampled_pixels = 0
        self.matched_pixels = 0
        self.depth_attributed_pixels = 0
        self.center_attributed_pixels = 0
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

    @staticmethod
    def _mask_type(mask_counts):
        return "unified_hotspot"

    @staticmethod
    def _mask_type_weight(mask_type):
        return 1.0

    def _center(self, key):
        return [(float(k) + 0.5) * self.voxel_size for k in key]

    def should_update(self, iteration):
        return self.start <= iteration <= self.until and iteration % max(self.update_interval, 1) == 0

    def should_grow(self, iteration):
        return self.start <= iteration <= self.until and iteration % max(self.grow_interval, 1) == 0

    def update(self, iteration, viewpoint_camera, render_image, gt_image, neural_xyz, visibility_filter, radii=None, neural_opacity=None):
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
        mask = high_error | highlight | thin_bright
        self.global_error_sum += float(rgb_error.sum().item())
        self.global_error_pixels += int(rgb_error.numel())

        if self.attribution_mode == "footprint_depth":
            self._update_footprint_depth(
                iteration, viewpoint_camera, rgb_error, score_map, high_error, highlight, thin_bright,
                mask, neural_xyz, visibility_filter, radii, neural_opacity
            )
        else:
            self._update_center(
                iteration, viewpoint_camera, rgb_error, score_map, high_error, highlight, thin_bright,
                neural_xyz, visibility_filter
            )

    def _record_point(self, iteration, viewpoint_camera, point, x, y, rgb_error, score_map, high_error, highlight, thin_bright):
        mask_code = 0
        if bool(high_error[y, x].item()):
            mask_code |= 1
        if bool(highlight[y, x].item()):
            mask_code |= 2
        if bool(thin_bright[y, x].item()):
            mask_code |= 4
        point_np = point.detach().cpu().numpy() if torch.is_tensor(point) else point
        key = self._key_tuple(point_np)
        stat = self.stats[key]
        stat["error_sum"] += float(score_map[y, x].item())
        stat["rgb_error_sum"] += float(rgb_error[y, x].item())
        stat["count"] += 1
        view_name = str(viewpoint_camera.image_name)
        stat["support_views"].add(view_name)
        stat["last_seen_iter"] = int(iteration)
        stat["mask_counts"].update([str(mask_code)])
        if view_name not in stat["view_dirs"]:
            cam_center = viewpoint_camera.camera_center.detach().cpu().numpy()
            direction = cam_center - point_np
            norm = float(np.linalg.norm(direction))
            if norm > 1e-8:
                stat["view_dirs"][view_name] = (direction / norm).astype(float).tolist()

    def _update_center(self, iteration, viewpoint_camera, rgb_error, score_map, high_error, highlight, thin_bright, neural_xyz, visibility_filter):
        height, width = rgb_error.shape
        active = visibility_filter.detach()
        if active.sum() == 0:
            return
        xyz = neural_xyz.detach()[active]
        px, py, valid = _project_points(xyz, viewpoint_camera, height, width)
        if valid is None or valid.sum() == 0:
            return
        px = px[valid]
        py = py[valid]
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
        selected_xyz = xyz[hit_ids]
        selected_px = px[hit_ids]
        selected_py = py[hit_ids]
        for local_i, point in enumerate(selected_xyz):
            x = int(selected_px[local_i].item())
            y = int(selected_py[local_i].item())
            self._record_point(iteration, viewpoint_camera, point, x, y, rgb_error, score_map, high_error, highlight, thin_bright)
            self.matched_pixels += 1
            self.center_attributed_pixels += 1

    def _update_footprint_depth(self, iteration, viewpoint_camera, rgb_error, score_map, high_error, highlight, thin_bright, mask, neural_xyz, visibility_filter, radii, neural_opacity):
        if radii is None:
            self._update_center(iteration, viewpoint_camera, rgb_error, score_map, high_error, highlight, thin_bright, neural_xyz, visibility_filter)
            return
        height, width = rgb_error.shape
        sample_x, sample_y = _sample_mask_pixels(mask, score_map, self.max_pixels_per_view)
        if sample_x is None:
            return
        sample_count = int(sample_x.numel())
        self.sampled_pixels += sample_count
        sample_radius_cap = torch.full(
            (sample_count,),
            float(self.reproj_radius_px),
            dtype=torch.float32,
            device=rgb_error.device,
        )

        active = visibility_filter.detach()
        if active.sum() == 0:
            return
        xyz = neural_xyz.detach()[active]
        active_radii = radii.detach()[active].float()
        if neural_opacity is not None and neural_opacity.numel() == neural_xyz.shape[0]:
            active_opacity = neural_opacity.detach().reshape(-1)[active].float().clamp_min(0.0)
        else:
            active_opacity = torch.ones_like(active_radii, dtype=torch.float32)
        px, py, valid = _project_points(xyz, viewpoint_camera, height, width)
        if valid is None or valid.sum() == 0:
            return
        xyz = xyz[valid]
        px = px[valid]
        py = py[valid]
        active_radii = active_radii[valid]
        active_opacity = active_opacity[valid]
        depth = _view_depth(xyz, viewpoint_camera).float()
        valid_depth = torch.isfinite(depth) & (depth > float(getattr(viewpoint_camera, "znear", 0.01)))
        if valid_depth.sum() == 0:
            return
        px = px[valid_depth]
        py = py[valid_depth]
        depth = depth[valid_depth]
        active_radii = active_radii[valid_depth].clamp_min(float(self.depth_min_radii))
        active_opacity = active_opacity[valid_depth]
        radius_cap = max(float(self.depth_radius_cap_px), float(self.reproj_radius_px), 1.0)
        radius = active_radii.clamp(max=radius_cap)

        sample_index = torch.full((height, width), -1, dtype=torch.long, device=rgb_error.device)
        sample_ids = torch.arange(sample_count, dtype=torch.long, device=rgb_error.device)
        sample_index[sample_y, sample_x] = sample_ids
        depth_num = torch.zeros(sample_count, dtype=torch.float32, device=rgb_error.device)
        weight_sum = torch.zeros(sample_count, dtype=torch.float32, device=rgb_error.device)
        cap = max(int(math.ceil(max(float(self.depth_radius_cap_px), float(self.reproj_radius_px)))), 1)
        for dy in range(-cap, cap + 1):
            sy = py + dy
            y_ok = (sy >= 0) & (sy < height)
            if not bool(y_ok.any().item()):
                continue
            for dx in range(-cap, cap + 1):
                sx = px + dx
                ok = y_ok & (sx >= 0) & (sx < width)
                if not bool(ok.any().item()):
                    continue
                sid = sample_index[sy[ok], sx[ok]]
                hit = sid >= 0
                if not bool(hit.any().item()):
                    continue
                local_radius = radius[ok][hit]
                hit_sid = sid[hit]
                sample_cap = sample_radius_cap[hit_sid]
                effective_radius = torch.minimum(local_radius, sample_cap)
                dist2 = float(dx * dx + dy * dy)
                in_radius = dist2 <= (effective_radius * effective_radius)
                if not bool(in_radius.any().item()):
                    continue
                sid = hit_sid[in_radius]
                local_depth = depth[ok][hit][in_radius]
                local_opacity = active_opacity[ok][hit][in_radius]
                local_radius = effective_radius[in_radius]
                sigma = torch.clamp(local_radius * 0.5, min=1.0)
                weight = local_opacity * torch.exp(torch.full_like(sigma, -dist2) / (2.0 * sigma * sigma))
                depth_num.index_add_(0, sid, local_depth * weight)
                weight_sum.index_add_(0, sid, weight)
        valid_sample = weight_sum > self.depth_min_weight
        if valid_sample.sum() == 0:
            return
        matched_x = sample_x[valid_sample]
        matched_y = sample_y[valid_sample]
        expected_depth = depth_num[valid_sample] / weight_sum[valid_sample].clamp_min(1e-8)
        points = _unproject_pixels_with_view_depth(matched_x, matched_y, expected_depth, viewpoint_camera, height, width)
        for local_i, point in enumerate(points):
            x = int(matched_x[local_i].item())
            y = int(matched_y[local_i].item())
            self._record_point(iteration, viewpoint_camera, point, x, y, rgb_error, score_map, high_error, highlight, thin_bright)
        matched = int(valid_sample.sum().item())
        self.matched_pixels += matched
        self.depth_attributed_pixels += matched

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
            mask_type = self._mask_type(stat["mask_counts"])
            min_support_views = self.min_support_views
            reproj_radius_px = self.reproj_radius_px
            is_active = support_views >= min_support_views and diverse and mean_rgb >= global_error * self.error_multiplier
            if not is_active:
                continue
            raw_score = mean_rgb / max(global_error, 1e-6)
            active.append({
                "key": key,
                "center": self._center(key),
                "score": min(raw_score, self.score_clip),
                "support_views": support_views,
                "min_support_views": int(min_support_views),
                "mean_rgb_error": mean_rgb,
                "max_view_angle_deg": max_angle,
                "near_anchor_count": int(anchor_counts.get(key, 0)),
                "count": int(stat["count"]),
                "last_seen_iter": int(stat["last_seen_iter"]),
                "mask_type": mask_type,
                "mask_type_weight": self._mask_type_weight(mask_type),
                "reproj_radius_px": float(reproj_radius_px),
            })
        active.sort(key=lambda item: (-item["score"], -item["support_views"], item["near_anchor_count"]))
        self.last_active = active
        self.last_summary = {
            "hotspot_voxels": len(self.stats),
            "active_hotspot_voxels": len(active),
            "sampled_pixels": self.sampled_pixels,
            "matched_pixels": self.matched_pixels,
            "depth_attributed_pixels": self.depth_attributed_pixels,
            "center_attributed_pixels": self.center_attributed_pixels,
            "attribution_mode": self.attribution_mode,
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
        if not active:
            return None, None, {"active": 0, "density_pass": 0, "proposed": 0}
        occupied_counts = [item["near_anchor_count"] for item in active if item["near_anchor_count"] > 0]
        mean_active_density = float(sum(occupied_counts)) / max(len(occupied_counts), 1)

        low_density = []
        for item in active:
            if item["count"] < self.add_min_votes:
                continue
            local_count = float(item["near_anchor_count"])
            low_relative_density = local_count < self.density_ratio_thresh * max(mean_active_density, 1.0)
            low_local_count = local_count < float(self.min_anchor_count)
            if not (low_relative_density or low_local_count):
                continue
            target_density = max(float(self.min_anchor_count), mean_active_density * self.density_ratio_thresh, 1.0)
            density_deficit_score = max(0.5, min(target_density / max(local_count, 1.0), 3.0))
            scored = dict(item)
            scored["density_deficit_score"] = float(density_deficit_score)
            scored["candidate_score"] = float(item["score"]) * float(density_deficit_score)
            low_density.append(scored)
        if not low_density:
            return None, None, {"active": len(active), "density_pass": 0, "proposed": 0, "mean_active_density": mean_active_density}
        low_density.sort(key=lambda item: (-item["candidate_score"], -item["support_views"], item["near_anchor_count"]))
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
            return None, None, {"active": len(active), "density_pass": len(low_density), "proposed": 0, "mean_active_density": mean_active_density}
        selected_types = Counter(item["mask_type"] for item in selected[:len(centers)])
        return (
            torch.tensor(centers, dtype=torch.float32, device=anchor_xyz.device),
            torch.tensor(parents, dtype=torch.long, device=anchor_xyz.device),
            {
                "active": len(active),
                "density_pass": len(low_density),
                "proposed": len(centers),
                "mean_active_density": mean_active_density,
                "selected_mask_types": dict(selected_types),
            },
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
