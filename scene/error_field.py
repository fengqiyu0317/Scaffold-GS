import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, num_freqs=6):
    enc = [x]
    for k in range(num_freqs):
        freq = 2.0 ** k
        enc.append(torch.sin(freq * torch.pi * x))
        enc.append(torch.cos(freq * torch.pi * x))
    return torch.cat(enc, dim=-1)


class ErrorField(nn.Module):
    def __init__(self, num_freqs=6, hidden_dim=64):
        super().__init__()
        self.num_freqs = int(num_freqs)
        in_dim = 3 * (1 + 2 * self.num_freqs)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = positional_encoding(x, self.num_freqs)
        return self.net(h).squeeze(-1)


def _normalize_xyz(x, bbox_min, bbox_max, eps=1e-8):
    return torch.clamp((x - bbox_min) / (bbox_max - bbox_min + eps), 0.0, 1.0)


def _safe_quantile(values, q):
    q = min(max(float(q), 0.0), 1.0)
    return torch.quantile(values.view(-1), q)


def _topk_overlap(pred, target, fraction=0.10):
    count = int(target.numel())
    if count <= 0:
        return torch.tensor(float('nan'), device=target.device)
    k = max(1, int(float(fraction) * count))
    pred_top = torch.topk(pred.view(-1), k, largest=True).indices
    target_top = torch.topk(target.view(-1), k, largest=True).indices
    target_mask = torch.zeros(count, dtype=torch.bool, device=target.device)
    target_mask[target_top] = True
    return target_mask[pred_top].float().mean()


def train_error_field_steps(
    error_field,
    error_optim,
    anchor_pos,
    residue_num_ema,
    residue_den_ema,
    residue_seen,
    steps=10,
    batch_size=8192,
    num_jitter=1,
    jitter_std=0.01,
    lambda_sparse=0.02,
    lambda_smooth=0.01,
    num_bg=2048,
    eps=1e-8,
):
    valid = residue_seen.detach().view(-1) > 0
    valid_count = int(valid.sum().item())
    if valid_count < 64:
        return {'skipped': True, 'valid_anchors': valid_count}

    anchor_pos = anchor_pos.detach()
    residue_num = residue_num_ema.detach().view(-1)
    residue_den = residue_den_ema.detach().view(-1)
    residue = residue_num / residue_den.clamp_min(eps)

    X = anchor_pos[valid]
    R = residue[valid]
    W = residue_den[valid]

    r_low = _safe_quantile(R, 0.30)
    r_high = _safe_quantile(R, 0.90)
    target = torch.clamp((R - r_low) / (r_high - r_low + eps), 0.0, 1.0)

    w_norm = _safe_quantile(W.clamp_min(0.0), 0.90).clamp_min(eps)
    weight = torch.clamp(W / w_norm, 0.0, 1.0)

    bbox_min = anchor_pos.min(dim=0).values.detach()
    bbox_max = anchor_pos.max(dim=0).values.detach()
    X_norm_all = _normalize_xyz(X, bbox_min, bbox_max, eps)

    last_stats = None
    sample_count = X_norm_all.shape[0]
    batch_size = min(int(batch_size), sample_count)

    error_field.train()
    for _ in range(int(steps)):
        if batch_size < sample_count:
            idx = torch.randint(0, sample_count, (batch_size,), device=X_norm_all.device)
            X_norm = X_norm_all[idx]
            target_batch = target[idx]
            weight_batch = weight[idx]
        else:
            X_norm = X_norm_all
            target_batch = target
            weight_batch = weight

        if int(num_jitter) > 1:
            X_norm = X_norm[:, None, :].repeat(1, int(num_jitter), 1)
            X_norm = X_norm + float(jitter_std) * torch.randn_like(X_norm)
            X_norm = torch.clamp(X_norm.reshape(-1, 3), 0.0, 1.0)
            target_batch = target_batch[:, None].repeat(1, int(num_jitter)).reshape(-1)
            weight_batch = weight_batch[:, None].repeat(1, int(num_jitter)).reshape(-1)
        elif float(jitter_std) > 0.0:
            X_norm = torch.clamp(X_norm + float(jitter_std) * torch.randn_like(X_norm), 0.0, 1.0)

        pred = error_field(X_norm)
        loss_data = (
            weight_batch * F.smooth_l1_loss(pred, target_batch, reduction='none')
        ).sum() / weight_batch.sum().clamp_min(eps)

        pred_bg = error_field(torch.rand(int(num_bg), 3, device=X_norm.device))
        loss_sparse = pred_bg.mean()

        X_near = torch.clamp(X_norm + 0.01 * torch.randn_like(X_norm), 0.0, 1.0)
        pred_near = error_field(X_near)
        loss_smooth = ((pred - pred_near) ** 2).mean()

        loss = loss_data + float(lambda_sparse) * loss_sparse + float(lambda_smooth) * loss_smooth
        error_optim.zero_grad(set_to_none=True)
        loss.backward()
        error_optim.step()

        with torch.no_grad():
            pred_all = error_field(X_norm_all)
            pred_std = pred_all.std(unbiased=False)
            target_std = target.std(unbiased=False)
            if pred_all.numel() >= 2 and pred_std > eps and target_std > eps:
                corr = torch.corrcoef(torch.stack([pred_all, target]))[0, 1]
            else:
                corr = torch.tensor(float('nan'), device=pred_all.device)
            top10_overlap = _topk_overlap(pred_all, target, 0.10)

        last_stats = {
            'skipped': False,
            'valid_anchors': valid_count,
            'loss': float(loss.detach().item()),
            'loss_data': float(loss_data.detach().item()),
            'loss_sparse': float(loss_sparse.detach().item()),
            'loss_smooth': float(loss_smooth.detach().item()),
            'target_min': float(target.min().detach().item()),
            'target_mean': float(target.mean().detach().item()),
            'target_max': float(target.max().detach().item()),
            'target_std': float(target_std.detach().item()),
            'pred_min': float(pred_all.min().detach().item()),
            'pred_mean': float(pred_all.mean().detach().item()),
            'pred_max': float(pred_all.max().detach().item()),
            'pred_std': float(pred_std.detach().item()),
            'corr_pred_target': float(corr.detach().item()),
            'top10_overlap': float(top10_overlap.detach().item()),
            'weight_mean': float(weight.mean().detach().item()),
            'bbox_min': bbox_min.detach(),
            'bbox_max': bbox_max.detach(),
        }

    return last_stats


@torch.no_grad()
def query_error_field_grid(error_field, bbox_min, bbox_max, resolution=64, chunk=65536, device='cuda'):
    error_field.eval()
    xs = torch.linspace(0.0, 1.0, int(resolution), device=device)
    ys = torch.linspace(0.0, 1.0, int(resolution), device=device)
    zs = torch.linspace(0.0, 1.0, int(resolution), device=device)
    try:
        mesh = torch.meshgrid(xs, ys, zs, indexing='ij')
    except TypeError:
        mesh = torch.meshgrid(xs, ys, zs)
    grid = torch.stack(mesh, dim=-1).reshape(-1, 3)

    scores = []
    for start in range(0, grid.shape[0], int(chunk)):
        scores.append(error_field(grid[start:start + int(chunk)]))
    scores = torch.cat(scores, dim=0)
    xyz_world = bbox_min + grid * (bbox_max - bbox_min)
    return xyz_world, scores


@torch.no_grad()
def save_error_points_ply(error_field, bbox_min, bbox_max, path, resolution=64, threshold=0.7, max_points=200000):
    xyz, score = query_error_field_grid(
        error_field,
        bbox_min,
        bbox_max,
        resolution=resolution,
        device=bbox_min.device,
    )
    mask = score > float(threshold)
    xyz = xyz[mask]
    score = score[mask]
    if xyz.shape[0] > int(max_points):
        topk = torch.topk(score, int(max_points), largest=True).indices
        xyz = xyz[topk]
        score = score[topk]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz_cpu = xyz.detach().cpu()
    score_cpu = score.detach().cpu()
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(xyz_cpu.shape[0]))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float error_score\n')
        f.write('end_header\n')
        for p, s in zip(xyz_cpu.tolist(), score_cpu.tolist()):
            f.write('{:.8f} {:.8f} {:.8f} {:.8f}\n'.format(p[0], p[1], p[2], s))
    return int(xyz_cpu.shape[0])


def _scalar_to_rgb(values):
    values = torch.clamp(values.float(), 0.0, 1.0)
    red = torch.clamp(1.5 - torch.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = torch.clamp(1.5 - torch.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = torch.clamp(1.5 - torch.abs(4.0 * values - 1.0), 0.0, 1.0)
    return torch.round(torch.stack([red, green, blue], dim=1) * 255.0).to(torch.uint8)


def _quantile_summary(values):
    return {
        str(q): float(_safe_quantile(values, q).detach().item())
        for q in (0.0, 0.5, 0.9, 0.95, 1.0)
    }


def _write_anchor_scalar_ply(path, xyz, color_values, scalar_name, scalar_values,
                             residual_norm, residual_raw, pred_error, residue_den, residue_seen):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz_cpu = xyz.detach().cpu().float()
    colors = _scalar_to_rgb(color_values.detach().cpu())
    scalar_cpu = scalar_values.detach().cpu().float()
    residual_norm_cpu = residual_norm.detach().cpu().float()
    residual_raw_cpu = residual_raw.detach().cpu().float()
    pred_error_cpu = pred_error.detach().cpu().float()
    residue_den_cpu = residue_den.detach().cpu().float()
    residue_seen_cpu = residue_seen.detach().cpu().float()

    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(xyz_cpu.shape[0]))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property float {}\n'.format(scalar_name))
        f.write('property float residual_norm\n')
        f.write('property float residual_raw\n')
        f.write('property float pred_error\n')
        f.write('property float residue_den\n')
        f.write('property float residue_seen\n')
        f.write('end_header\n')
        for p, c, s, rn, rr, pe, den, seen in zip(
                xyz_cpu.tolist(), colors.tolist(), scalar_cpu.tolist(),
                residual_norm_cpu.tolist(), residual_raw_cpu.tolist(),
                pred_error_cpu.tolist(), residue_den_cpu.tolist(), residue_seen_cpu.tolist()):
            f.write(
                '{:.8f} {:.8f} {:.8f} {} {} {} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(
                    p[0], p[1], p[2], c[0], c[1], c[2], s, rn, rr, pe, den, seen
                )
            )


@torch.no_grad()
def save_anchor_error_visualization_ply(
    error_field,
    anchor_pos,
    residue_num_ema,
    residue_den_ema,
    residue_seen,
    bbox_min,
    bbox_max,
    output_dir,
    iteration,
    eps=1e-8,
    chunk=65536,
):
    anchor_pos = anchor_pos.detach()
    residue_num = residue_num_ema.detach().view(-1)
    residue_den = residue_den_ema.detach().view(-1)
    residue_seen = residue_seen.detach().view(-1)
    valid = torch.logical_and(residue_seen > 0, residue_den > 0)
    valid_count = int(valid.sum().item())
    if valid_count < 2:
        return {'skipped': True, 'valid_anchors': valid_count}

    residual_raw = torch.zeros_like(residue_den)
    residual_raw[valid] = residue_num[valid] / residue_den[valid].clamp_min(eps)
    valid_residual = residual_raw[valid]
    r_low = _safe_quantile(valid_residual, 0.30)
    r_high = _safe_quantile(valid_residual, 0.90)
    residual_norm = torch.zeros_like(residual_raw)
    residual_norm[valid] = torch.clamp(
        (residual_raw[valid] - r_low) / (r_high - r_low + eps),
        0.0,
        1.0,
    )

    error_field.eval()
    pred_chunks = []
    for start in range(0, anchor_pos.shape[0], int(chunk)):
        x_norm = _normalize_xyz(anchor_pos[start:start + int(chunk)], bbox_min, bbox_max, eps)
        pred_chunks.append(error_field(x_norm).detach())
    pred_error = torch.cat(pred_chunks, dim=0)

    pred_valid = pred_error[valid]
    target_valid = residual_norm[valid]
    pred_std = pred_valid.std(unbiased=False)
    target_std = target_valid.std(unbiased=False)
    if pred_valid.numel() >= 2 and pred_std > eps and target_std > eps:
        corr = torch.corrcoef(torch.stack([pred_valid, target_valid]))[0, 1]
    else:
        corr = torch.tensor(float('nan'), device=pred_valid.device)
    top10_overlap = _topk_overlap(pred_valid, target_valid, 0.10)
    top5_overlap = _topk_overlap(pred_valid, target_valid, 0.05)

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, 'anchors_colored_by_error_field_{}.ply'.format(iteration))
    residual_path = os.path.join(output_dir, 'anchors_colored_by_residual_{}.ply'.format(iteration))
    _write_anchor_scalar_ply(
        pred_path, anchor_pos, pred_error, 'error_field_score', pred_error,
        residual_norm, residual_raw, pred_error, residue_den, residue_seen,
    )
    _write_anchor_scalar_ply(
        residual_path, anchor_pos, residual_norm, 'residual_color_score', residual_norm,
        residual_norm, residual_raw, pred_error, residue_den, residue_seen,
    )

    summary = {
        'skipped': False,
        'iteration': int(iteration),
        'anchor_count': int(anchor_pos.shape[0]),
        'valid_residual_anchor_count': valid_count,
        'residual_raw_quantiles_valid': _quantile_summary(valid_residual),
        'residual_norm_quantiles_valid': _quantile_summary(target_valid),
        'pred_error_quantiles_valid': _quantile_summary(pred_valid),
        'corr_pred_error_vs_residual_norm_valid': float(corr.detach().item()),
        'top10_overlap': float(top10_overlap.detach().item()),
        'top5_overlap': float(top5_overlap.detach().item()),
        'bbox_min': [float(x) for x in bbox_min.detach().cpu().tolist()],
        'bbox_max': [float(x) for x in bbox_max.detach().cpu().tolist()],
        'outputs': {
            'pred_error_ply': pred_path,
            'residual_ply': residual_path,
        },
    }
    summary_path = os.path.join(output_dir, 'anchor_error_visualization_summary_{}.json'.format(iteration))
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    summary['summary_path'] = summary_path
    return summary


def save_error_field_checkpoint(error_field, error_optim, bbox_min, bbox_max, path, iteration, opt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'iteration': int(iteration),
            'model_state': error_field.state_dict(),
            'optimizer_state': error_optim.state_dict(),
            'bbox_min': bbox_min.detach(),
            'bbox_max': bbox_max.detach(),
            'num_freqs': int(getattr(opt, 'error_field_num_freqs', 6)),
            'hidden_dim': int(getattr(opt, 'error_field_hidden_dim', 64)),
        },
        path,
    )
