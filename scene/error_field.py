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
