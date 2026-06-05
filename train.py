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
from utils.loss_utils import l1_loss, ssim, ssim_error_map
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.error_field import (
    ErrorField,
    save_anchor_error_visualization_ply,
    save_error_field_checkpoint,
    save_error_points_ply,
    train_error_field_steps,
)
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

def build_residue_error_map(image, gt_image, opt):
    image = image.detach().clamp(0.0, 1.0)
    gt_image = gt_image.detach().clamp(0.0, 1.0)
    l1_map = torch.abs(image - gt_image).mean(dim=0)
    ssim_map = ssim_error_map(image, gt_image).detach()
    lambda_dssim = float(getattr(opt, "lambda_dssim", 0.2))
    return ((1.0 - lambda_dssim) * l1_map + lambda_dssim * ssim_map).contiguous()


def parse_error_field_validation_deltas(value):
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip()]
    else:
        parts = [value]
    deltas = sorted({int(p) for p in parts if int(p) > 0})
    return deltas


def _normalize_anchor_residual(gaussians, eps):
    residue_seen = gaussians.anchor_residue_seen.detach().view(-1)
    residue_den = gaussians.anchor_residue_den_ema.detach().view(-1)
    residue_num = gaussians.anchor_residue_num_ema.detach().view(-1)
    valid = torch.logical_and(residue_seen > 0, residue_den > 0)
    residual_raw = torch.zeros_like(residue_den)
    residual_raw[valid] = residue_num[valid] / residue_den[valid].clamp_min(eps)
    residual_norm = torch.zeros_like(residual_raw)
    if int(valid.sum().item()) >= 2:
        valid_residual = residual_raw[valid]
        r_low = torch.quantile(valid_residual, 0.30)
        r_high = torch.quantile(valid_residual, 0.90)
        residual_norm[valid] = torch.clamp((residual_raw[valid] - r_low) / (r_high - r_low + eps), 0.0, 1.0)
    return valid, residual_norm


def _anchor_gradient_and_grow_candidate(gaussians, opt):
    anchor_count = int(gaussians.get_anchor.shape[0])
    required = anchor_count * int(gaussians.n_offsets)
    if gaussians.offset_gradient_accum.numel() < required or gaussians.offset_denom.numel() < required:
        device = gaussians.get_anchor.device
        return torch.full((anchor_count,), float('nan'), device=device), torch.zeros(anchor_count, dtype=torch.bool, device=device)

    grad_accum = gaussians.offset_gradient_accum.detach()[:required].float().view(required, -1)
    denom = gaussians.offset_denom.detach()[:required].float().view(required, -1)
    grads = grad_accum / denom.clamp_min(1.0)
    grads[torch.isnan(grads)] = 0.0
    grads_norm = torch.norm(grads, dim=-1).view(anchor_count, int(gaussians.n_offsets))
    denom_by_anchor = denom.view(anchor_count, int(gaussians.n_offsets), -1).squeeze(-1)
    offset_mask = denom_by_anchor > float(opt.update_interval) * float(opt.success_threshold) * 0.5
    grow_offset = torch.logical_and(grads_norm >= float(opt.densify_grad_threshold), offset_mask)
    return grads_norm.max(dim=1).values, grow_offset.any(dim=1)


def _uid_membership(values, members):
    values = values.detach().view(-1).long()
    members = members.detach().view(-1).long()
    if members.numel() == 0 or values.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    max_uid = int(torch.maximum(values.max(), members.max()).item())
    lookup = torch.zeros(max_uid + 1, dtype=torch.bool, device=values.device)
    lookup[members.to(values.device)] = True
    return lookup[values]


def _current_anchor_validation_state(gaussians, opt):
    valid, residual_norm = _normalize_anchor_residual(gaussians, float(opt.residue_div_eps))
    gradient_score, grow_candidate = _anchor_gradient_and_grow_candidate(gaussians, opt)
    return {
        'uid': gaussians.get_anchor_uid.detach().view(-1).long(),
        'valid': valid,
        'residual_norm': residual_norm,
        'gradient_score': gradient_score,
        'grow_candidate': grow_candidate,
    }


def freeze_error_field_validation_snapshot(error_field, gaussians, bbox_min, bbox_max, iteration, opt, logger=None):
    state = _current_anchor_validation_state(gaussians, opt)
    valid = state['valid']
    valid_count = int(valid.sum().item())
    if valid_count < 2:
        return None

    error_field.eval()
    anchor_pos = gaussians.get_anchor.detach()
    pred_chunks = []
    chunk = 65536
    eps = float(opt.residue_div_eps)
    for start in range(0, anchor_pos.shape[0], chunk):
        x_norm = torch.clamp((anchor_pos[start:start + chunk] - bbox_min) / (bbox_max - bbox_min + eps), 0.0, 1.0)
        pred_chunks.append(error_field(x_norm).detach())
    pred_error = torch.cat(pred_chunks, dim=0)

    high_quantile = float(getattr(opt, 'error_field_validation_high_quantile', 0.9))
    threshold = torch.quantile(pred_error[valid].view(-1), high_quantile)
    high = torch.logical_and(valid, pred_error > threshold)
    if int(high.sum().item()) == 0:
        high = torch.logical_and(valid, pred_error >= threshold)
    other = torch.logical_and(valid, ~high)
    deltas = parse_error_field_validation_deltas(getattr(opt, 'error_field_validation_deltas', '100,500,1000'))
    if not deltas:
        return None

    snapshot = {
        'origin_iter': int(iteration),
        'due_iters': [int(iteration) + d for d in deltas],
        'reported_due_iters': set(),
        'high_uid': state['uid'][high].detach().clone(),
        'other_uid': state['uid'][other].detach().clone(),
        'pred_error_threshold': float(threshold.detach().item()),
        'high_quantile': high_quantile,
        'valid_anchor_count_t': valid_count,
        'high_anchor_count_t': int(high.sum().item()),
        'other_anchor_count_t': int(other.sum().item()),
        'accum': {
            'high_residual_sum': 0.0,
            'other_residual_sum': 0.0,
            'high_gradient_sum': 0.0,
            'other_gradient_sum': 0.0,
            'high_grow_sum': 0.0,
            'other_grow_sum': 0.0,
            'high_count': 0,
            'other_count': 0,
            'high_gradient_count': 0,
            'other_gradient_count': 0,
            'high_grow_count': 0,
            'other_grow_count': 0,
        },
    }
    if logger:
        logger.info(
            '[ITER {}] Freeze error-field validation: high {}/{} anchors at q{:.2f}; due {}'.format(
                iteration, snapshot['high_anchor_count_t'], valid_count, high_quantile, deltas
            )
        )
    return snapshot


def _accumulate_group(accum, prefix, state, mask):
    mask = torch.logical_and(mask, state['valid'])
    count = int(mask.sum().item())
    if count <= 0:
        return
    accum[f'{prefix}_residual_sum'] += float(state['residual_norm'][mask].sum().detach().item())
    accum[f'{prefix}_count'] += count

    gradient = state['gradient_score']
    grad_mask = torch.logical_and(mask, torch.isfinite(gradient))
    grad_count = int(grad_mask.sum().item())
    if grad_count > 0:
        accum[f'{prefix}_gradient_sum'] += float(gradient[grad_mask].sum().detach().item())
        accum[f'{prefix}_gradient_count'] += grad_count

    accum[f'{prefix}_grow_sum'] += float(state['grow_candidate'][mask].float().sum().detach().item())
    accum[f'{prefix}_grow_count'] += count


def update_error_field_validation_snapshots(pending, gaussians, opt, iteration, tb_writer=None, logger=None, dataset_name=None):
    if not pending:
        return []
    state = _current_anchor_validation_state(gaussians, opt)
    active = []
    for snapshot in pending:
        if iteration <= snapshot['origin_iter']:
            active.append(snapshot)
            continue
        high_mask = _uid_membership(state['uid'], snapshot['high_uid'])
        other_mask = _uid_membership(state['uid'], snapshot['other_uid'])
        _accumulate_group(snapshot['accum'], 'high', state, high_mask)
        _accumulate_group(snapshot['accum'], 'other', state, other_mask)

        for due_iter in snapshot['due_iters']:
            if iteration < due_iter or due_iter in snapshot['reported_due_iters']:
                continue
            accum = snapshot['accum']
            high_res = accum['high_residual_sum'] / max(accum['high_count'], 1)
            other_res = accum['other_residual_sum'] / max(accum['other_count'], 1)
            high_grad = accum['high_gradient_sum'] / max(accum['high_gradient_count'], 1)
            other_grad = accum['other_gradient_sum'] / max(accum['other_gradient_count'], 1)
            high_grow = accum['high_grow_sum'] / max(accum['high_grow_count'], 1)
            other_grow = accum['other_grow_sum'] / max(accum['other_grow_count'], 1)
            delta = int(due_iter - snapshot['origin_iter'])
            if tb_writer:
                prefix = f'{dataset_name}/error_field_lag{delta}'
                tb_writer.add_scalar(f'{prefix}/high_residual_future_mean', high_res, iteration)
                tb_writer.add_scalar(f'{prefix}/other_residual_future_mean', other_res, iteration)
                tb_writer.add_scalar(f'{prefix}/residual_future_delta', high_res - other_res, iteration)
                tb_writer.add_scalar(f'{prefix}/high_gradient_future_mean', high_grad, iteration)
                tb_writer.add_scalar(f'{prefix}/other_gradient_future_mean', other_grad, iteration)
                tb_writer.add_scalar(f'{prefix}/gradient_future_delta', high_grad - other_grad, iteration)
                tb_writer.add_scalar(f'{prefix}/high_grow_candidate_rate_window', high_grow, iteration)
                tb_writer.add_scalar(f'{prefix}/other_grow_candidate_rate_window', other_grow, iteration)
                tb_writer.add_scalar(f'{prefix}/grow_candidate_rate_window_delta', high_grow - other_grow, iteration)
            if logger:
                logger.info(
                    '[ITER {}] Error-field delayed validation t={} Δ={}: residual {:.4f} vs {:.4f}, '
                    'gradient {:.6f} vs {:.6f}, grow_rate {:.4f} vs {:.4f}'.format(
                        iteration, snapshot['origin_iter'], delta,
                        high_res, other_res, high_grad, other_grad, high_grow, other_grow,
                    )
                )
            snapshot['reported_due_iters'].add(due_iter)
        if len(snapshot['reported_due_iters']) < len(snapshot['due_iters']):
            active.append(snapshot)
    return active

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


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    error_field = None
    error_field_optim = None
    error_field_trained = False
    error_field_bbox_min = None
    error_field_bbox_max = None
    if opt.use_error_field:
        error_field = ErrorField(
            num_freqs=opt.error_field_num_freqs,
            hidden_dim=opt.error_field_hidden_dim,
        ).cuda()
        error_field_optim = torch.optim.Adam(error_field.parameters(), lr=opt.error_field_lr)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    pending_error_field_validations = []
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
        residue_pkg = None
        if (opt.use_residue_tracking or opt.use_adaptive_k or opt.use_error_field) and iteration < opt.update_until and iteration > opt.start_stat:
            residue_error_map = build_residue_error_map(image, gt_image, opt)
            with torch.no_grad():
                residue_pkg = render(
                    viewpoint_cam, gaussians, pipe, background,
                    visible_mask=voxel_visible_mask, residue_error_map=residue_error_map
                )

        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

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
            if ((opt.use_residue_tracking or opt.use_adaptive_k or opt.use_error_field) and hasattr(gaussians, "anchor_residue_seen")
                    and iteration % opt.residue_log_interval == 0 and gaussians.anchor_residue_seen.numel() > 0):
                residue_seen = gaussians.anchor_residue_seen.squeeze(1) > 0
                if residue_seen.sum() > 0:
                    residue_values = gaussians.get_anchor_residue[residue_seen]
                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/residue/mean', residue_values.mean().item(), iteration)
                        tb_writer.add_scalar(f'{dataset_name}/residue/max', residue_values.max().item(), iteration)
                        tb_writer.add_scalar(f'{dataset_name}/residue/observed_anchors', residue_seen.sum().item(), iteration)
                        tb_writer.add_scalar(f'{dataset_name}/residue/mean_den', gaussians.anchor_residue_den_ema[residue_seen].mean().item(), iteration)

            if (opt.use_adaptive_k and hasattr(gaussians, "anchor_active_offsets")
                    and iteration % opt.residue_log_interval == 0 and gaussians.anchor_active_offsets.numel() > 0):
                active_k = gaussians.anchor_active_offsets.float()
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/adaptive_k/mean', active_k.mean().item(), iteration)
                    tb_writer.add_scalar(f'{dataset_name}/adaptive_k/min', active_k.min().item(), iteration)
                    tb_writer.add_scalar(f'{dataset_name}/adaptive_k/max', active_k.max().item(), iteration)
                    if hasattr(gaussians, "adaptive_residue_den_window") and gaussians.adaptive_residue_den_window.numel() > 0:
                        observed_window = gaussians.adaptive_seen_window.squeeze(1) >= 1
                        if observed_window.sum() > 0:
                            tb_writer.add_scalar(f'{dataset_name}/adaptive_k/window_den_mean', gaussians.adaptive_residue_den_window[observed_window].mean().item(), iteration)
                            tb_writer.add_scalar(f'{dataset_name}/adaptive_k/window_seen_mean', gaussians.adaptive_seen_window[observed_window].mean().item(), iteration)

            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if opt.use_error_field and error_field_trained:
                    error_field_dir = os.path.join(scene.model_path, "error_field")
                    save_error_field_checkpoint(
                        error_field, error_field_optim, error_field_bbox_min, error_field_bbox_max,
                        os.path.join(error_field_dir, "error_field_latest.pth"), iteration, opt,
                    )
                    exported_points = save_error_points_ply(
                        error_field, error_field_bbox_min, error_field_bbox_max,
                        os.path.join(error_field_dir, "high_error_points_{}.ply".format(iteration)),
                        resolution=opt.error_field_query_resolution,
                        threshold=opt.error_field_score_threshold,
                    )
                    anchor_vis_stats = None
                    if (hasattr(gaussians, "anchor_residue_seen")
                            and gaussians.anchor_residue_seen.numel() > 0):
                        anchor_vis_stats = save_anchor_error_visualization_ply(
                            error_field=error_field,
                            anchor_pos=gaussians.get_anchor,
                            residue_num_ema=gaussians.anchor_residue_num_ema,
                            residue_den_ema=gaussians.anchor_residue_den_ema,
                            residue_seen=gaussians.anchor_residue_seen,
                            bbox_min=error_field_bbox_min,
                            bbox_max=error_field_bbox_max,
                            output_dir=os.path.join(error_field_dir, "anchor_visualization"),
                            iteration=iteration,
                            eps=opt.residue_div_eps,
                        )
                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/error_field/exported_points', exported_points, iteration)
                        if anchor_vis_stats and not anchor_vis_stats.get("skipped", False):
                            tb_writer.add_scalar(
                                f'{dataset_name}/error_field/anchor_vis_corr',
                                anchor_vis_stats["corr_pred_error_vs_residual_norm_valid"],
                                iteration,
                            )
                            tb_writer.add_scalar(
                                f'{dataset_name}/error_field/anchor_vis_top10_overlap',
                                anchor_vis_stats["top10_overlap"],
                                iteration,
                            )
                    if logger and anchor_vis_stats and not anchor_vis_stats.get("skipped", False):
                        logger.info(
                            "[ITER {}] Error field anchor PLY: corr {:.4f}, top10 {:.4f}, valid {}".format(
                                iteration,
                                anchor_vis_stats["corr_pred_error_vs_residual_norm_valid"],
                                anchor_vis_stats["top10_overlap"],
                                anchor_vis_stats["valid_residual_anchor_count"],
                            )
                        )
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(
                    viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask,
                    gaussian_residue_num=None if residue_pkg is None else residue_pkg.get("gaussian_residue_num", None),
                    gaussian_residue_den=None if residue_pkg is None else residue_pkg.get("gaussian_residue_den", None),
                    neural_anchor_indices=None if residue_pkg is None else residue_pkg.get("neural_anchor_indices", None),
                )

                pending_error_field_validations = update_error_field_validation_snapshots(
                    pending_error_field_validations,
                    gaussians,
                    opt,
                    iteration,
                    tb_writer=tb_writer,
                    logger=logger,
                    dataset_name=dataset_name,
                )

                if (opt.use_error_field and error_field is not None
                        and iteration >= opt.error_field_start
                        and iteration % opt.error_field_interval == 0
                        and hasattr(gaussians, "anchor_residue_seen")
                        and gaussians.anchor_residue_seen.numel() > 0):
                    with torch.enable_grad():
                        error_stats = train_error_field_steps(
                            error_field=error_field,
                            error_optim=error_field_optim,
                            anchor_pos=gaussians.get_anchor,
                            residue_num_ema=gaussians.anchor_residue_num_ema,
                            residue_den_ema=gaussians.anchor_residue_den_ema,
                            residue_seen=gaussians.anchor_residue_seen,
                            steps=opt.error_field_steps,
                            batch_size=opt.error_field_batch_size,
                            jitter_std=opt.error_field_jitter_std,
                            lambda_sparse=opt.error_field_sparse_weight,
                            lambda_smooth=opt.error_field_smooth_weight,
                            eps=opt.residue_div_eps,
                        )
                    if error_stats and not error_stats.get("skipped", False):
                        error_field_trained = True
                        error_field_bbox_min = error_stats["bbox_min"]
                        error_field_bbox_max = error_stats["bbox_max"]
                        if tb_writer:
                            tb_writer.add_scalar(f'{dataset_name}/error_field/loss', error_stats["loss"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/loss_data', error_stats["loss_data"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/loss_sparse', error_stats["loss_sparse"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/loss_smooth', error_stats["loss_smooth"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/valid_anchors', error_stats["valid_anchors"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/target_min', error_stats["target_min"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/target_mean', error_stats["target_mean"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/target_max', error_stats["target_max"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/target_std', error_stats["target_std"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/pred_min', error_stats["pred_min"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/pred_mean', error_stats["pred_mean"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/pred_max', error_stats["pred_max"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/pred_std', error_stats["pred_std"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/corr_pred_target', error_stats["corr_pred_target"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/top10_overlap', error_stats["top10_overlap"], iteration)
                            tb_writer.add_scalar(f'{dataset_name}/error_field/weight_mean', error_stats["weight_mean"], iteration)
                        if (int(getattr(opt, "error_field_validation_interval", 100)) > 0
                                and iteration % int(getattr(opt, "error_field_validation_interval", 100)) == 0):
                            validation_snapshot = freeze_error_field_validation_snapshot(
                                error_field,
                                gaussians,
                                error_field_bbox_min,
                                error_field_bbox_max,
                                iteration,
                                opt,
                                logger=logger,
                            )
                            if validation_snapshot is not None:
                                pending_error_field_validations.append(validation_snapshot)
                    elif tb_writer and error_stats:
                        tb_writer.add_scalar(f'{dataset_name}/error_field/valid_anchors', error_stats.get("valid_anchors", 0), iteration)
                    if logger and error_stats and not error_stats.get("skipped", False):
                        logger.info(
                            "[ITER {}] Error field: loss {:.6f}, corr {:.4f}, top10 {:.4f}, "
                            "target [{:.4f}, {:.4f}, {:.4f}], pred [{:.4f}, {:.4f}, {:.4f}], valid {}".format(
                                iteration,
                                error_stats["loss"],
                                error_stats["corr_pred_target"],
                                error_stats["top10_overlap"],
                                error_stats["target_min"],
                                error_stats["target_mean"],
                                error_stats["target_max"],
                                error_stats["pred_min"],
                                error_stats["pred_mean"],
                                error_stats["pred_max"],
                                error_stats["valid_anchors"],
                            )
                        )
                
                if (opt.use_adaptive_k and iteration > opt.update_from
                        and iteration % opt.adaptive_k_update_interval == 0):
                    adaptive_stats = gaussians.adjust_adaptive_k()
                    if tb_writer and adaptive_stats:
                        tb_writer.add_scalar(f'{dataset_name}/adaptive_k/grow', adaptive_stats.get("grow", 0), iteration)
                        tb_writer.add_scalar(f'{dataset_name}/adaptive_k/shrink', adaptive_stats.get("shrink", 0), iteration)
                        tb_writer.add_scalar(f'{dataset_name}/adaptive_k/valid', adaptive_stats.get("valid", 0), iteration)

                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                if getattr(gaussians, "use_residue_tracking", False) and not opt.use_error_field:
                    del gaussians.anchor_residue_num_ema
                    del gaussians.anchor_residue_den_ema
                    del gaussians.anchor_residue_seen
                if getattr(gaussians, "use_adaptive_k", False):
                    del gaussians.anchor_visibility_ema
                    del gaussians.adaptive_residue_num_window
                    del gaussians.adaptive_residue_den_window
                    del gaussians.adaptive_seen_window
                    del gaussians.adaptive_high_count
                    del gaussians.adaptive_low_count
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if opt.use_error_field and error_field_trained:
                    save_error_field_checkpoint(
                        error_field, error_field_optim, error_field_bbox_min, error_field_bbox_max,
                        os.path.join(scene.model_path, "error_field", "error_field_latest.pth"), iteration, opt,
                    )

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
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
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
