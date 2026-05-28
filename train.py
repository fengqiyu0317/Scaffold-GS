#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torch.nn.functional as F
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def build_refinement_error_map(image, gt_image, opt):
    image = image.detach().clamp(0.0, 1.0)
    gt_image = gt_image.detach().clamp(0.0, 1.0)

    luma_weight = getattr(opt, "error_luma_weight", 0.50)
    chroma_weight = getattr(opt, "error_chroma_weight", 0.40)
    edge_weight = getattr(opt, "error_edge_weight", 0.35)
    highlight_weight = getattr(opt, "error_highlight_weight", 0.75)
    highlight_threshold = getattr(opt, "error_highlight_threshold", 0.65)
    structure_weight = getattr(opt, "error_structure_weight", 0.50)
    structure_kernel = max(3, int(getattr(opt, "error_structure_kernel", 15)))
    local_max_weight = getattr(opt, "error_local_max_weight", 0.60)
    local_mean_weight = getattr(opt, "error_local_mean_weight", 0.15)
    local_kernel = max(1, int(getattr(opt, "error_local_kernel", 7)))
    if local_kernel % 2 == 0:
        local_kernel += 1
    if structure_kernel % 2 == 0:
        structure_kernel += 1

    rgb_error = torch.abs(image - gt_image).mean(dim=0, keepdim=True)

    luma_coeff = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    pred_luma = (image * luma_coeff).sum(dim=0, keepdim=True)
    gt_luma = (gt_image * luma_coeff).sum(dim=0, keepdim=True)
    luma_error = torch.abs(pred_luma - gt_luma)

    pred_chroma = image - pred_luma
    gt_chroma = gt_image - gt_luma
    chroma_error = torch.sqrt((pred_chroma - gt_chroma).pow(2).sum(dim=0, keepdim=True) + 1e-12) / (3.0 ** 0.5)

    highlight_mask = ((gt_luma - highlight_threshold) / max(1e-6, 1.0 - highlight_threshold)).clamp(0.0, 1.0)
    highlight_error = torch.relu(gt_luma - pred_luma) * highlight_mask

    base_error = rgb_error + luma_weight * luma_error + chroma_weight * chroma_error + highlight_weight * highlight_error
    base_error = base_error / max(1e-6, 1.0 + luma_weight + chroma_weight + highlight_weight)

    if edge_weight > 0:
        sobel_x = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 4.0
        sobel_y = image.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 4.0
        pred_luma_4d = pred_luma.unsqueeze(0)
        gt_luma_4d = gt_luma.unsqueeze(0)
        pred_grad_x = F.conv2d(pred_luma_4d, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_luma_4d, sobel_y, padding=1)
        gt_grad_x = F.conv2d(gt_luma_4d, sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt_luma_4d, sobel_y, padding=1)
        edge_error = torch.sqrt((pred_grad_x - gt_grad_x).pow(2) + (pred_grad_y - gt_grad_y).pow(2) + 1e-12).squeeze(0)
        base_error = (base_error + edge_weight * edge_error) / (1.0 + edge_weight)

    error_4d = base_error.unsqueeze(0)
    if structure_weight > 0 and structure_kernel > 1:
        structure_padding = structure_kernel // 2
        vertical_error = F.max_pool2d(error_4d, kernel_size=(structure_kernel, 3), stride=1, padding=(structure_padding, 1))
        horizontal_error = F.max_pool2d(error_4d, kernel_size=(3, structure_kernel), stride=1, padding=(1, structure_padding))
        line_error = torch.maximum(vertical_error, horizontal_error)
        error_4d = (error_4d + structure_weight * line_error) / (1.0 + structure_weight)

    if local_kernel > 1:
        padding = local_kernel // 2
        local_max = F.max_pool2d(error_4d, kernel_size=local_kernel, stride=1, padding=padding)
        local_mean = F.avg_pool2d(error_4d, kernel_size=local_kernel, stride=1, padding=padding)
        error_4d = (error_4d + local_max_weight * local_max + local_mean_weight * local_mean) / max(1e-6, 1.0 + local_max_weight + local_mean_weight)

    return error_4d.squeeze(0).squeeze(0)


def compute_luma(image):
    coeff = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (image * coeff).sum(dim=0, keepdim=True)


def sobel_luma_edges(luma):
    sobel_x = luma.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 4.0
    sobel_y = luma.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 4.0
    luma_4d = luma.unsqueeze(0)
    return F.conv2d(luma_4d, sobel_x, padding=1).squeeze(0), F.conv2d(luma_4d, sobel_y, padding=1).squeeze(0)


def build_component_refinement_maps(image, gt_image, opt):
    image_detached = image.detach().clamp(0.0, 1.0)
    gt_detached = gt_image.detach().clamp(0.0, 1.0)
    pred_luma = compute_luma(image_detached)
    gt_luma = compute_luma(gt_detached)
    deficit = torch.relu(gt_luma - pred_luma)

    dot_kernel = max(3, int(getattr(opt, "component_dot_kernel", 9)))
    line_kernel = max(5, int(getattr(opt, "component_line_kernel", 21)))
    if dot_kernel % 2 == 0:
        dot_kernel += 1
    if line_kernel % 2 == 0:
        line_kernel += 1

    local_mean = F.avg_pool2d(gt_luma.unsqueeze(0), kernel_size=dot_kernel, stride=1, padding=dot_kernel // 2).squeeze(0)
    local_contrast = torch.relu(gt_luma - local_mean)
    highlight_seed = (gt_luma > getattr(opt, "component_highlight_threshold", 0.62)).float()
    deficit_seed = (deficit > getattr(opt, "component_deficit_threshold", 0.03)).float()
    contrast_seed = (local_contrast > getattr(opt, "component_local_contrast_threshold", 0.08)).float()
    bright_density = F.avg_pool2d(highlight_seed.unsqueeze(0), kernel_size=dot_kernel, stride=1, padding=dot_kernel // 2).squeeze(0)
    small_highlight = highlight_seed * deficit_seed * contrast_seed * (bright_density < 0.45).float()
    small_highlight = F.max_pool2d(small_highlight.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)

    rgb_max = gt_detached.max(dim=0, keepdim=True)[0]
    rgb_min = gt_detached.min(dim=0, keepdim=True)[0]
    saturation = rgb_max - rgb_min
    white_seed = ((gt_luma > getattr(opt, "component_white_luma_threshold", 0.58)) &
                  (saturation < getattr(opt, "component_white_saturation_threshold", 0.28))).float()
    vertical_context = F.max_pool2d(white_seed.unsqueeze(0), kernel_size=(line_kernel, 3), stride=1, padding=(line_kernel // 2, 1)).squeeze(0)
    horizontal_context = F.max_pool2d(white_seed.unsqueeze(0), kernel_size=(3, line_kernel), stride=1, padding=(1, line_kernel // 2)).squeeze(0)
    gt_grad_x, _ = sobel_luma_edges(gt_luma)
    vertical_edge_seed = (gt_grad_x.abs() > gt_grad_x.abs().mean().clamp_min(1e-6) * 1.5).float()
    thin_vertical = white_seed * vertical_context * (1.0 - 0.5 * horizontal_context).clamp(0.0, 1.0)
    thin_vertical = torch.maximum(thin_vertical, white_seed * vertical_edge_seed)
    thin_vertical = F.max_pool2d(thin_vertical.unsqueeze(0), kernel_size=(line_kernel, 3), stride=1, padding=(line_kernel // 2, 1)).squeeze(0)

    component_mask = torch.maximum(small_highlight, thin_vertical).detach().clamp(0.0, 1.0)
    highlight_mask = small_highlight.detach().clamp(0.0, 1.0)
    vertical_mask = thin_vertical.detach().clamp(0.0, 1.0)
    component_map = torch.maximum(component_mask * torch.maximum(deficit, local_contrast), vertical_mask * (deficit + gt_grad_x.abs())).detach()
    return component_mask, highlight_mask, vertical_mask, component_map.squeeze(0)


def masked_mean(value, mask):
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def component_refinement_loss(image, gt_image, opt):
    component_mask, highlight_mask, vertical_mask, _ = build_component_refinement_maps(image, gt_image, opt)
    if component_mask.sum() <= 0:
        return image.new_tensor(0.0), component_mask, highlight_mask, vertical_mask

    pred_luma = compute_luma(image.clamp(0.0, 1.0))
    gt_luma = compute_luma(gt_image.clamp(0.0, 1.0))
    l1_map = torch.abs(image - gt_image).mean(dim=0, keepdim=True)
    component_loss = masked_mean(l1_map, component_mask)

    if highlight_mask.sum() > 0:
        highlight_deficit = masked_mean(torch.relu(gt_luma - pred_luma), highlight_mask)
    else:
        highlight_deficit = image.new_tensor(0.0)

    if vertical_mask.sum() > 0:
        pred_grad_x, _ = sobel_luma_edges(pred_luma)
        gt_grad_x, _ = sobel_luma_edges(gt_luma)
        vertical_edge_loss = masked_mean(torch.abs(pred_grad_x - gt_grad_x), vertical_mask)
    else:
        vertical_edge_loss = image.new_tensor(0.0)

    loss = (getattr(opt, "component_loss_weight", 0.10) * component_loss +
            getattr(opt, "highlight_deficit_weight", 0.08) * highlight_deficit +
            getattr(opt, "vertical_edge_weight", 0.04) * vertical_edge_loss)
    return loss, component_mask, highlight_mask, vertical_mask


def sample_neural_gaussian_errors(viewpoint_camera, neural_xyz, visibility_filter, error_map, opt=None):
    if neural_xyz.numel() == 0:
        empty_errors = torch.zeros((0, 1), dtype=error_map.dtype, device=error_map.device)
        empty_mask = torch.zeros((0,), dtype=torch.bool, device=error_map.device)
        return empty_errors, empty_mask

    height, width = error_map.shape
    errors = torch.zeros((neural_xyz.shape[0], 1), dtype=error_map.dtype, device=error_map.device)
    valid_mask = torch.zeros((neural_xyz.shape[0],), dtype=torch.bool, device=error_map.device)

    active_mask = visibility_filter.detach()
    if active_mask.sum() == 0:
        return errors, valid_mask

    xyz = neural_xyz.detach()[active_mask]
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_hom = torch.cat([xyz, ones], dim=1)
    projected = torch.matmul(xyz_hom, viewpoint_camera.full_proj_transform)
    denom = projected[:, 3].abs().clamp_min(1e-7)
    ndc = projected[:, :3] / denom.unsqueeze(1)

    grid = torch.stack([ndc[:, 0], -ndc[:, 1]], dim=-1)
    in_image = (grid[:, 0] >= -1.0) & (grid[:, 0] <= 1.0) & (grid[:, 1] >= -1.0) & (grid[:, 1] <= 1.0)

    active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
    if in_image.sum() > 0:
        target_indices = active_indices[in_image]
        valid_grid = grid[in_image]
        error_image = error_map.view(1, 1, height, width)
        sample_radius = 0.0 if opt is None else float(getattr(opt, "error_sample_radius", 4.0))
        sample_max_weight = 0.0 if opt is None else float(getattr(opt, "error_sample_max_weight", 0.85))
        if sample_radius > 0.0:
            dx = 2.0 * sample_radius / max(width - 1, 1)
            dy = 2.0 * sample_radius / max(height - 1, 1)
            offsets = valid_grid.new_tensor([
                [0.0, 0.0], [dx, 0.0], [-dx, 0.0], [0.0, dy], [0.0, -dy],
                [dx, dy], [dx, -dy], [-dx, dy], [-dx, -dy],
            ])
            sample_grid = (valid_grid.unsqueeze(1) + offsets.unsqueeze(0)).view(1, valid_grid.shape[0], offsets.shape[0], 2)
            sampled = F.grid_sample(error_image, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(valid_grid.shape[0], offsets.shape[0])
            center_errors = sampled[:, 0]
            max_errors = sampled.max(dim=1)[0]
            sampled_errors = center_errors * (1.0 - sample_max_weight) + max_errors * sample_max_weight
        else:
            sample_grid = valid_grid.view(1, -1, 1, 2)
            sampled_errors = F.grid_sample(error_image, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(-1)
        errors[target_indices, 0] = sampled_errors
        valid_mask[target_indices] = True

    return errors, valid_mask


def project_points_to_image(viewpoint_camera, xyz):
    if xyz.numel() == 0:
        empty_grid = torch.zeros((0, 2), dtype=xyz.dtype, device=xyz.device)
        empty_xy = torch.zeros((0, 2), dtype=xyz.dtype, device=xyz.device)
        empty_depth = torch.zeros((0,), dtype=xyz.dtype, device=xyz.device)
        empty_mask = torch.zeros((0,), dtype=torch.bool, device=xyz.device)
        return empty_grid, empty_xy, empty_depth, empty_mask

    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_hom = torch.cat([xyz, ones], dim=1)
    projected = torch.matmul(xyz_hom, viewpoint_camera.full_proj_transform)
    denom = projected[:, 3].abs().clamp_min(1e-7)
    ndc = projected[:, :3] / denom.unsqueeze(1)
    grid = torch.stack([ndc[:, 0], -ndc[:, 1]], dim=-1)
    in_image = (grid[:, 0] >= -1.0) & (grid[:, 0] <= 1.0) & (grid[:, 1] >= -1.0) & (grid[:, 1] <= 1.0)

    width = float(viewpoint_camera.image_width)
    height = float(viewpoint_camera.image_height)
    pixel_xy = torch.stack([
        (grid[:, 0] + 1.0) * 0.5 * max(width - 1.0, 1.0),
        (grid[:, 1] + 1.0) * 0.5 * max(height - 1.0, 1.0),
    ], dim=-1)

    view_xyz = torch.matmul(xyz_hom, viewpoint_camera.world_view_transform)
    depth = view_xyz[:, 2]
    return grid, pixel_xy, depth, in_image


def sample_map_neighborhood(value_map, grid, radius_px=0.0):
    if grid.numel() == 0:
        return torch.zeros((0,), dtype=value_map.dtype, device=value_map.device)

    height, width = value_map.shape
    image = value_map.view(1, 1, height, width)
    radius_px = float(radius_px)
    if radius_px <= 0.0:
        return F.grid_sample(image, grid.view(1, -1, 1, 2), mode='bilinear', padding_mode='zeros', align_corners=True).view(-1)

    dx = 2.0 * radius_px / max(width - 1, 1)
    dy = 2.0 * radius_px / max(height - 1, 1)
    offsets = grid.new_tensor([
        [0.0, 0.0], [dx, 0.0], [-dx, 0.0], [0.0, dy], [0.0, -dy],
        [dx, dy], [dx, -dy], [-dx, dy], [-dx, -dy],
    ])
    sample_grid = (grid.unsqueeze(1) + offsets.unsqueeze(0)).view(1, grid.shape[0], offsets.shape[0], 2)
    sampled = F.grid_sample(image, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return sampled.view(grid.shape[0], offsets.shape[0]).max(dim=1)[0]


def build_component_candidate_points(viewpoint_camera, component_map, component_mask, pixel_xy, depth, reliable_filter, loose_filter, opt):
    if int(getattr(opt, 'component_proposal_level', 0)) < 2:
        return torch.zeros((0, 3), dtype=component_map.dtype, device=component_map.device)

    mask_2d = component_mask.squeeze(0) if component_mask.dim() == 3 else component_mask
    score_map = (component_map * (mask_2d > 0).float()).clamp_min(0.0)
    if score_map.max() <= 0:
        return torch.zeros((0, 3), dtype=component_map.dtype, device=component_map.device)

    nms_kernel = max(3, int(getattr(opt, 'component_proposal_nms_kernel', 9)))
    if nms_kernel % 2 == 0:
        nms_kernel += 1
    pooled = F.max_pool2d(score_map.view(1, 1, *score_map.shape), kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2).view_as(score_map)
    positive = score_map > score_map[score_map > 0].mean().clamp_min(1e-8)
    maxima = (score_map >= pooled) & positive
    coords_yx = torch.nonzero(maxima, as_tuple=False)
    if coords_yx.numel() == 0:
        return torch.zeros((0, 3), dtype=component_map.dtype, device=component_map.device)

    values = score_map[coords_yx[:, 0], coords_yx[:, 1]]
    max_points = max(1, int(getattr(opt, 'component_proposal_max_points', 256)))
    if coords_yx.shape[0] > max_points:
        top_idx = torch.topk(values, k=max_points, largest=True)[1]
        coords_yx = coords_yx[top_idx]

    candidate_pixels = torch.stack([coords_yx[:, 1].float(), coords_yx[:, 0].float()], dim=1).to(component_map.device)
    attribution_radius = float(getattr(opt, 'component_attribution_radius_px', 4.0))
    if reliable_filter is not None and reliable_filter.sum() > 0:
        reliable_pixels = pixel_xy[reliable_filter]
        if reliable_pixels.shape[0] > 16384:
            keep = torch.randperm(reliable_pixels.shape[0], device=reliable_pixels.device)[:16384]
            reliable_pixels = reliable_pixels[keep]
        min_dist = torch.full((candidate_pixels.shape[0],), float('inf'), dtype=component_map.dtype, device=component_map.device)
        for start in range(0, reliable_pixels.shape[0], 4096):
            dist = torch.cdist(candidate_pixels, reliable_pixels[start:start + 4096])
            min_dist = torch.minimum(min_dist, dist.min(dim=1)[0])
        candidate_pixels = candidate_pixels[min_dist > attribution_radius]
        if candidate_pixels.numel() == 0:
            return torch.zeros((0, 3), dtype=component_map.dtype, device=component_map.device)

    depth_filter = loose_filter if loose_filter is not None and loose_filter.sum() > 0 else torch.isfinite(depth)
    depth_pixels = pixel_xy[depth_filter]
    depth_values = depth[depth_filter]
    finite = torch.isfinite(depth_values)
    depth_pixels = depth_pixels[finite]
    depth_values = depth_values[finite]
    if depth_values.numel() == 0:
        return torch.zeros((0, 3), dtype=component_map.dtype, device=component_map.device)
    if depth_values.shape[0] > 32768:
        keep = torch.randperm(depth_values.shape[0], device=depth_values.device)[:32768]
        depth_pixels = depth_pixels[keep]
        depth_values = depth_values[keep]

    loose_radius = float(getattr(opt, 'component_loose_radius_px', 12.0))
    ray_samples = max(1, int(getattr(opt, 'component_ray_depth_samples', 3)))
    width = float(viewpoint_camera.image_width)
    height = float(viewpoint_camera.image_height)
    tanfovx = np.tan(float(viewpoint_camera.FoVx) * 0.5)
    tanfovy = np.tan(float(viewpoint_camera.FoVy) * 0.5)
    inv_view = torch.inverse(viewpoint_camera.world_view_transform)

    world_points = []
    for pix in candidate_pixels:
        distances = torch.norm(depth_pixels - pix.unsqueeze(0), dim=1)
        nearby_depth = depth_values[distances <= loose_radius]
        if nearby_depth.numel() == 0:
            continue
        nearby_depth = torch.sort(nearby_depth)[0]
        if ray_samples == 1 or nearby_depth.numel() == 1:
            sample_depths = nearby_depth[nearby_depth.shape[0] // 2].view(1)
        else:
            q = torch.linspace(0.25, 0.75, steps=ray_samples, device=nearby_depth.device)
            indices = (q * (nearby_depth.shape[0] - 1)).long().clamp(0, nearby_depth.shape[0] - 1)
            sample_depths = nearby_depth[indices]

        ndc_x = 2.0 * pix[0] / max(width - 1.0, 1.0) - 1.0
        ndc_y = -(2.0 * pix[1] / max(height - 1.0, 1.0) - 1.0)
        for d in sample_depths:
            cam_point = torch.stack([ndc_x * tanfovx * d, ndc_y * tanfovy * d, d, torch.ones_like(d)]).view(1, 4)
            world_hom = torch.matmul(cam_point, inv_view)
            world_points.append(world_hom[:, :3] / world_hom[:, 3:].clamp_min(1e-7))

    if not world_points:
        return torch.zeros((0, 3), dtype=component_map.dtype, device=component_map.device)

    candidates = torch.cat(world_points, dim=0)
    reproj_grid, _, _, in_image = project_points_to_image(viewpoint_camera, candidates)
    reproj_score = sample_map_neighborhood(mask_2d, reproj_grid, radius_px=1.0)
    keep = in_image & (reproj_score > 0.1)
    return candidates[keep].detach()


def analyze_component_refinement(viewpoint_camera, neural_xyz, visibility_filter, selection_mask, neural_opacity, component_map, component_mask, opt):
    component_scores = torch.zeros((neural_xyz.shape[0], 1), dtype=component_map.dtype, device=component_map.device)
    component_score_filter = torch.zeros((neural_xyz.shape[0],), dtype=torch.bool, device=component_map.device)
    proposal_scores = torch.zeros_like(component_scores)
    proposal_filter = torch.zeros_like(component_score_filter)
    candidate_xyz = torch.zeros((0, 3), dtype=component_map.dtype, device=component_map.device)
    if neural_xyz.numel() == 0:
        return component_scores, component_score_filter, proposal_scores, proposal_filter, candidate_xyz

    grid, pixel_xy, depth, in_image = project_points_to_image(viewpoint_camera, neural_xyz.detach())
    active = visibility_filter.detach() & in_image
    attribution_radius = float(getattr(opt, 'component_attribution_radius_px', 4.0))
    loose_radius = float(getattr(opt, 'component_loose_radius_px', 12.0))
    center_scores = sample_map_neighborhood(component_map, grid, radius_px=attribution_radius)
    loose_scores = sample_map_neighborhood(component_map, grid, radius_px=loose_radius)

    if neural_opacity is not None and selection_mask is not None and neural_opacity.numel() == selection_mask.numel():
        selected_opacity = neural_opacity.detach().view(-1)[selection_mask.detach()].view(-1)
    else:
        selected_opacity = torch.ones((neural_xyz.shape[0],), dtype=component_map.dtype, device=component_map.device)
    if selected_opacity.shape[0] != neural_xyz.shape[0]:
        selected_opacity = torch.ones((neural_xyz.shape[0],), dtype=component_map.dtype, device=component_map.device)

    positive = active & (center_scores > 0)
    depth_ok = positive.clone()
    if positive.sum() > 0:
        positive_depth = depth[positive]
        center_depth = positive_depth.median()
        band = max(float(getattr(opt, 'component_attribution_depth_abs_band', 0.05)), abs(float(center_depth.detach().item())) * float(getattr(opt, 'component_attribution_depth_rel_band', 0.15)))
        depth_ok = torch.abs(depth - center_depth) <= band

    opacity_ok = selected_opacity > float(getattr(opt, 'component_attribution_min_opacity', 0.01))
    reliable_filter = positive & depth_ok & opacity_ok
    component_scores[:, 0] = center_scores
    component_score_filter = reliable_filter

    proposal_level = int(getattr(opt, 'component_proposal_level', 0))
    if proposal_level >= 1:
        loose_positive = active & (loose_scores > 0)
        loose_depth_ok = loose_positive.clone()
        if loose_positive.sum() > 0:
            loose_depth = depth[loose_positive]
            center_depth = loose_depth.median()
            band = max(float(getattr(opt, 'component_attribution_depth_abs_band', 0.05)), abs(float(center_depth.detach().item())) * float(getattr(opt, 'component_loose_depth_rel_band', 0.45)))
            loose_depth_ok = torch.abs(depth - center_depth) <= band
        proposal_filter = loose_positive & loose_depth_ok & (~reliable_filter)
        proposal_scores[:, 0] = loose_scores

    candidate_xyz = build_component_candidate_points(
        viewpoint_camera, component_map, component_mask, pixel_xy, depth,
        reliable_filter, active & (loose_scores > 0), opt,
    )
    return component_scores, component_score_filter, proposal_scores, proposal_filter, candidate_xyz


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              dataset.use_viewdist_pe, dataset.view_pe_freqs, dataset.dist_pe_freqs, dataset.pe_include_input)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        neural_errors = None
        neural_error_filter = None
        component_scores = None
        component_score_filter = None
        component_proposal_scores = None
        component_proposal_filter = None
        component_candidate_xyz = None
        component_active = (getattr(opt, "use_component_refinement", False)
                            and iteration >= getattr(opt, "component_refine_start", opt.update_from)
                            and iteration < getattr(opt, "component_refine_until", opt.update_until))
        if opt.use_error_aware_refinement and iteration < opt.update_until and iteration > opt.start_stat:
            error_map = build_refinement_error_map(image, gt_image, opt)
            neural_errors, neural_error_filter = sample_neural_gaussian_errors(
                viewpoint_cam, render_pkg["neural_xyz"], visibility_filter, error_map, opt
            )
        component_loss = image.new_tensor(0.0)
        if component_active and iteration < opt.update_until and iteration > opt.start_stat:
            component_loss, component_mask, _, _ = component_refinement_loss(image, gt_image, opt)
            _, _, _, component_map = build_component_refinement_maps(image, gt_image, opt)
            component_scores, component_score_filter, component_proposal_scores, component_proposal_filter, component_candidate_xyz = analyze_component_refinement(
                viewpoint_cam,
                render_pkg["neural_xyz"],
                visibility_filter,
                offset_selection_mask,
                render_pkg["neural_opacity"],
                component_map,
                component_mask,
                opt,
            )
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg + component_loss

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(
                    viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask,
                    neural_errors=neural_errors,
                    neural_error_filter=neural_error_filter,
                    neural_offset_indices=render_pkg.get("neural_offset_indices", None),
                    neural_anchor_indices=render_pkg.get("neural_anchor_indices", None),
                    component_scores=component_scores,
                    component_score_filter=component_score_filter,
                    component_proposal_scores=component_proposal_scores,
                    component_proposal_filter=component_proposal_filter,
                    component_candidate_xyz=component_candidate_xyz,
                )
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                if getattr(gaussians, "use_error_aware_refinement", False):
                    del gaussians.offset_error_accum
                    del gaussians.offset_error_denom
                    del gaussians.anchor_error_accum
                    del gaussians.anchor_error_denom
                if getattr(gaussians, "use_component_refinement", False):
                    del gaussians.offset_component_accum
                    del gaussians.offset_component_denom
                    del gaussians.anchor_component_accum
                    del gaussians.anchor_component_denom
                    del gaussians.anchor_component_views
                    del gaussians.offset_component_proposal_accum
                    del gaussians.offset_component_proposal_denom
                    gaussians.component_candidate_xyz = torch.empty((0, 3), device="cuda")
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              dataset.use_viewdist_pe, dataset.view_pe_freqs, dataset.dist_pe_freqs, dataset.pe_include_input)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
