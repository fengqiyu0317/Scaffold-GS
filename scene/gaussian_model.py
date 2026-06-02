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

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding

    
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)
        self.offset_error_accum = torch.empty(0)
        self.offset_error_denom = torch.empty(0)
        self.anchor_error_accum = torch.empty(0)
        self.anchor_error_denom = torch.empty(0)
        self.use_error_aware_refinement = False
        self.error_grow_weight = 0.5
        self.error_norm_clip = 3.0
        self.error_prune_keep_ratio = 1.0

        self.anchor_parent = torch.empty(0, dtype=torch.long)
        self.anchor_depth = torch.empty(0, dtype=torch.int32)
        self.anchor_children_count = torch.empty(0, dtype=torch.int32)
        self.use_tree_anchor_refinement = False
        self.tree_max_depth = 6
        self.tree_grow_weight = 0.5
        self.tree_error_norm_clip = 3.0
        self.tree_error_keep_ratio = 1.0
        self.tree_child_base_cap = 4
        self.tree_child_high_cap = 12
        self.tree_nonleaf_prune = False

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.anchor_parent,
            self.anchor_depth,
            self.anchor_children_count,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) == 13:
            (self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.anchor_parent,
            self.anchor_depth,
            self.anchor_children_count) = model_args
        elif len(model_args) == 10:
            (self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._init_anchor_tree()
        elif len(model_args) == 11:
            (self.active_sh_degree,
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._init_anchor_tree()
        else:
            raise ValueError(f"Unsupported checkpoint format with {len(model_args)} fields")
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self._init_anchor_tree()


    def _init_anchor_tree(self):
        n = self.get_anchor.shape[0]
        device = self.get_anchor.device if self.get_anchor.numel() > 0 else torch.device("cuda")
        self.anchor_parent = torch.full((n,), -1, dtype=torch.long, device=device)
        self.anchor_depth = torch.zeros((n,), dtype=torch.int32, device=device)
        self.anchor_children_count = torch.zeros((n,), dtype=torch.int32, device=device)

    def _ensure_anchor_tree(self):
        n = self.get_anchor.shape[0]
        device = self.get_anchor.device if self.get_anchor.numel() > 0 else torch.device("cuda")
        if self.anchor_parent.numel() != n or self.anchor_depth.numel() != n or self.anchor_children_count.numel() != n:
            self._init_anchor_tree()
            return
        self.anchor_parent = self.anchor_parent.to(device=device, dtype=torch.long)
        self.anchor_depth = self.anchor_depth.to(device=device, dtype=torch.int32)
        self.anchor_children_count = self.anchor_children_count.to(device=device, dtype=torch.int32)

    def _compute_subtree_stats(self):
        self._ensure_anchor_tree()
        n = self.get_anchor.shape[0]
        device = self.get_anchor.device
        if n == 0:
            empty = torch.empty(0, device=device)
            return {
                "subtree_error_mean": empty,
                "subtree_error_count": empty,
                "subtree_grad_mean": empty,
                "subtree_grad_count": empty,
                "subtree_opacity_mean": empty,
                "subtree_visit_count": empty,
                "subtree_size": empty,
            }

        if self.anchor_error_accum.numel() == n and self.anchor_error_denom.numel() == n:
            anchor_error_count = self.anchor_error_denom.view(-1).float()
            anchor_error_mean = (self.anchor_error_accum.view(-1) / anchor_error_count.clamp_min(1.0)).float()
        else:
            anchor_error_count = torch.zeros(n, device=device)
            anchor_error_mean = torch.zeros(n, device=device)

        offset_total = n * self.n_offsets
        if self.offset_gradient_accum.numel() >= offset_total and self.offset_denom.numel() >= offset_total:
            offset_grad_accum = self.offset_gradient_accum[:offset_total].view(n, self.n_offsets).float()
            offset_grad_denom = self.offset_denom[:offset_total].view(n, self.n_offsets).float()
            offset_grad_mean = offset_grad_accum / offset_grad_denom.clamp_min(1.0)
            anchor_grad_count = offset_grad_denom.sum(dim=1)
            anchor_grad_mean = offset_grad_mean.mean(dim=1)
        else:
            anchor_grad_count = torch.zeros(n, device=device)
            anchor_grad_mean = torch.zeros(n, device=device)

        if self.opacity_accum.numel() == n and self.anchor_demon.numel() == n:
            anchor_visit_count = self.anchor_demon.view(-1).float()
            anchor_opacity_mean = (self.opacity_accum.view(-1) / anchor_visit_count.clamp_min(1.0)).float()
        else:
            anchor_visit_count = torch.zeros(n, device=device)
            anchor_opacity_mean = torch.zeros(n, device=device)

        subtree_error_sum = anchor_error_mean * anchor_error_count
        subtree_error_count = anchor_error_count.clone()
        subtree_grad_sum = anchor_grad_mean * anchor_grad_count
        subtree_grad_count = anchor_grad_count.clone()
        subtree_opacity_sum = anchor_opacity_mean * anchor_visit_count
        subtree_visit_count = anchor_visit_count.clone()
        subtree_size = torch.ones(n, dtype=torch.float32, device=device)

        max_depth = int(self.anchor_depth.max().item()) if self.anchor_depth.numel() > 0 else 0
        for depth in range(max_depth, 0, -1):
            child_ids = torch.nonzero(self.anchor_depth == depth, as_tuple=False).squeeze(1)
            if child_ids.numel() == 0:
                continue
            parent_ids = self.anchor_parent[child_ids]
            valid = parent_ids >= 0
            if valid.sum() == 0:
                continue
            child_ids = child_ids[valid]
            parent_ids = parent_ids[valid]
            subtree_error_sum.scatter_add_(0, parent_ids, subtree_error_sum[child_ids])
            subtree_error_count.scatter_add_(0, parent_ids, subtree_error_count[child_ids])
            subtree_grad_sum.scatter_add_(0, parent_ids, subtree_grad_sum[child_ids])
            subtree_grad_count.scatter_add_(0, parent_ids, subtree_grad_count[child_ids])
            subtree_opacity_sum.scatter_add_(0, parent_ids, subtree_opacity_sum[child_ids])
            subtree_visit_count.scatter_add_(0, parent_ids, subtree_visit_count[child_ids])
            subtree_size.scatter_add_(0, parent_ids, subtree_size[child_ids])

        return {
            "subtree_error_mean": subtree_error_sum / subtree_error_count.clamp_min(1.0),
            "subtree_error_count": subtree_error_count,
            "subtree_grad_mean": subtree_grad_sum / subtree_grad_count.clamp_min(1.0),
            "subtree_grad_count": subtree_grad_count,
            "subtree_opacity_mean": subtree_opacity_sum / subtree_visit_count.clamp_min(1.0),
            "subtree_visit_count": subtree_visit_count,
            "subtree_size": subtree_size,
        }

    def _remap_anchor_tree_after_prune(self, prune_mask):
        self._ensure_anchor_tree()
        valid_points_mask = ~prune_mask
        old_to_new = torch.full((prune_mask.shape[0],), -1, dtype=torch.long, device=prune_mask.device)
        old_to_new[valid_points_mask] = torch.arange(valid_points_mask.sum(), device=prune_mask.device)

        new_parent = self.anchor_parent[valid_points_mask].clone()
        parent_valid = new_parent >= 0
        if parent_valid.sum() > 0:
            new_parent[parent_valid] = old_to_new[new_parent[parent_valid]]
            orphan = parent_valid & (new_parent < 0)
            new_parent[orphan] = -1

        self.anchor_parent = new_parent.long()
        old_depth = self.anchor_depth[valid_points_mask].clone().int()
        self.anchor_depth = torch.zeros_like(old_depth, dtype=torch.int32)
        max_old_depth = int(old_depth.max().item()) if old_depth.numel() > 0 else 0
        for depth in range(1, max_old_depth + 1):
            depth_mask = torch.logical_and(old_depth == depth, self.anchor_parent >= 0)
            node_ids = torch.nonzero(depth_mask, as_tuple=False).squeeze(1)
            if node_ids.numel() == 0:
                continue
            self.anchor_depth[node_ids] = self.anchor_depth[self.anchor_parent[node_ids]].int() + 1
        self.anchor_children_count = torch.zeros_like(self.anchor_depth, dtype=torch.int32)
        valid_child = self.anchor_parent >= 0
        if valid_child.sum() > 0:
            self.anchor_children_count.scatter_add_(
                0,
                self.anchor_parent[valid_child],
                torch.ones(valid_child.sum(), dtype=torch.int32, device=prune_mask.device),
            )

    def _save_tree_stats(self, path):
        if self.anchor_parent.numel() != self.get_anchor.shape[0]:
            self._init_anchor_tree()
        stats_path = os.path.splitext(path)[0] + "_tree_stats.json"
        if self.anchor_parent.numel() == 0:
            payload = {"anchor_count": 0}
        else:
            children = self.anchor_children_count.detach().float()
            depth = self.anchor_depth.detach()
            payload = {
                "anchor_count": int(self.get_anchor.shape[0]),
                "max_tree_depth": int(depth.max().item()),
                "mean_children_count": float(children.mean().item()),
                "max_children_count": int(self.anchor_children_count.max().item()),
                "leaf_anchor_count": int((self.anchor_children_count == 0).sum().item()),
                "root_anchor_count": int((self.anchor_parent < 0).sum().item()),
            }
        with open(stats_path, "w") as fp:
            json.dump(payload, fp, indent=2)


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.use_error_aware_refinement = getattr(training_args, "use_error_aware_refinement", False)
        self.error_grow_weight = getattr(training_args, "error_grow_weight", 0.5)
        self.error_norm_clip = getattr(training_args, "error_norm_clip", 3.0)
        self.error_prune_keep_ratio = getattr(training_args, "error_prune_keep_ratio", 1.0)
        self.use_tree_anchor_refinement = getattr(training_args, "use_tree_anchor_refinement", False)
        self.tree_max_depth = getattr(training_args, "tree_max_depth", 6)
        self.tree_grow_weight = getattr(training_args, "tree_grow_weight", 0.5)
        self.tree_error_norm_clip = getattr(training_args, "tree_error_norm_clip", 3.0)
        self.tree_error_keep_ratio = getattr(training_args, "tree_error_keep_ratio", 1.0)
        self.tree_child_base_cap = getattr(training_args, "tree_child_base_cap", 4)
        self.tree_child_high_cap = getattr(training_args, "tree_child_high_cap", 12)
        self.tree_nonleaf_prune = getattr(training_args, "tree_nonleaf_prune", False)
        self._ensure_anchor_tree()

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_error_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_error_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_error_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.anchor_error_denom = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        self._save_tree_stats(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._init_anchor_tree()


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask,
                        neural_errors=None, neural_error_filter=None, neural_offset_indices=None, neural_anchor_indices=None):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        if self.use_error_aware_refinement and neural_errors is not None and neural_offset_indices is not None and neural_anchor_indices is not None:
            error_filter = update_filter if neural_error_filter is None else torch.logical_and(update_filter, neural_error_filter)
            if error_filter.sum() > 0:
                error_values = neural_errors[error_filter].detach().view(-1, 1)
                offset_indices = neural_offset_indices[error_filter].detach().long().view(-1, 1)
                anchor_indices = neural_anchor_indices[error_filter].detach().long().view(-1, 1)
                ones = torch.ones_like(error_values)
                self.offset_error_accum.scatter_add_(0, offset_indices, error_values)
                self.offset_error_denom.scatter_add_(0, offset_indices, ones)
                self.anchor_error_accum.scatter_add_(0, anchor_indices, error_values)
                self.anchor_error_denom.scatter_add_(0, anchor_indices, ones)

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask, source_scores=None, tree_stats=None):
        ## 
        self._ensure_anchor_tree()
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            if candidate_mask.sum() == 0:
                continue

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            selected_offset_ids = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
            selected_parent_ids = torch.div(selected_offset_ids, self.n_offsets, rounding_mode="floor").long()
            score_source = grads if source_scores is None else source_scores
            if score_source.shape[0] < candidate_mask.shape[0]:
                score_source = torch.cat([
                    score_source,
                    torch.full((candidate_mask.shape[0] - score_source.shape[0],), -1e9, dtype=score_source.dtype, device=score_source.device),
                ], dim=0)
            selected_scores = score_source[:candidate_mask.shape[0]][candidate_mask].float()
            _, best_local = scatter_max(selected_scores, inverse_indices, dim=0)
            best_parent = selected_parent_ids[best_local]
            best_scores = selected_scores[best_local]


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for chunk_idx in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size
            new_parent = best_parent[remove_duplicates]
            candidate_scores = best_scores[remove_duplicates]

            if self.use_tree_anchor_refinement and candidate_anchor.shape[0] > 0:
                parent_depth = self.anchor_depth[new_parent].long()
                depth_ok = parent_depth + 1 <= int(self.tree_max_depth)
                if tree_stats is not None and tree_stats["subtree_error_mean"].numel() == self.anchor_parent.numel():
                    subtree_error = tree_stats["subtree_error_mean"]
                    subtree_visit = tree_stats["subtree_visit_count"]
                    observed = subtree_visit > 0
                    if observed.sum() > 0:
                        global_error = subtree_error[observed].mean().clamp_min(1e-6)
                        parent_high_error = subtree_error[new_parent] > global_error
                    else:
                        parent_high_error = torch.zeros_like(depth_ok)
                else:
                    parent_high_error = torch.zeros_like(depth_ok)
                child_cap = torch.where(
                    parent_high_error,
                    torch.full_like(new_parent, int(self.tree_child_high_cap), dtype=torch.int32),
                    torch.full_like(new_parent, int(self.tree_child_base_cap), dtype=torch.int32),
                )
                budget_ok = self.anchor_children_count[new_parent] < child_cap
                keep = torch.logical_and(depth_ok, budget_ok)
                if keep.sum() > 0:
                    final_keep = torch.zeros_like(keep)
                    for parent in torch.unique(new_parent[keep]):
                        ids = torch.nonzero(torch.logical_and(keep, new_parent == parent), as_tuple=False).squeeze(1)
                        cap = int(child_cap[ids[0]].item())
                        current = int(self.anchor_children_count[parent].item())
                        remaining = cap - current
                        if remaining <= 0:
                            continue
                        if ids.numel() > remaining:
                            top_ids = torch.topk(candidate_scores[ids], remaining).indices
                            ids = ids[top_ids]
                        final_keep[ids] = True
                    keep = final_keep
                candidate_anchor = candidate_anchor[keep]
                new_parent = new_parent[keep]

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                if self.use_tree_anchor_refinement:
                    new_feat = self._anchor_feat[new_parent]
                else:
                    new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                    new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                if self.use_tree_anchor_refinement:
                    new_depth = (self.anchor_depth[new_parent].long() + 1).int()
                    self.anchor_parent = torch.cat([self.anchor_parent, new_parent.long()], dim=0)
                    self.anchor_depth = torch.cat([self.anchor_depth, new_depth], dim=0)
                    self.anchor_children_count = torch.cat([
                        self.anchor_children_count,
                        torch.zeros(candidate_anchor.shape[0], dtype=torch.int32, device="cuda"),
                    ], dim=0)
                    self.anchor_children_count.scatter_add_(
                        0,
                        new_parent.long(),
                        torch.ones(candidate_anchor.shape[0], dtype=torch.int32, device="cuda"),
                    )

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        grow_scores = grads_norm
        if self.use_error_aware_refinement and self.offset_error_denom.numel() == self.offset_gradient_accum.numel():
            offset_error_mean = self.offset_error_accum / self.offset_error_denom.clamp_min(1.0)
            observed_error = (self.offset_error_denom > 0).squeeze(dim=1)
            if observed_error.sum() > 0:
                mean_error = offset_error_mean[observed_error].mean().clamp_min(1e-6)
                error_norm = (offset_error_mean.squeeze(dim=1) / mean_error).clamp(max=self.error_norm_clip)
                grow_scores = grads_norm * (1.0 + self.error_grow_weight * error_norm)

        tree_stats = None
        if self.use_tree_anchor_refinement:
            tree_stats = self._compute_subtree_stats()
            subtree_error = tree_stats["subtree_error_mean"]
            subtree_visit = tree_stats["subtree_visit_count"]
            observed = subtree_visit > 0
            if observed.sum() > 0:
                global_error = subtree_error[observed].mean().clamp_min(1e-6)
                subtree_error_norm = (subtree_error / global_error).clamp(max=self.tree_error_norm_clip)
                anchor_factor = 1.0 + self.tree_grow_weight * subtree_error_norm
            else:
                anchor_factor = torch.ones_like(subtree_error)
            max_depth = max(int(self.tree_max_depth), 1)
            depth_factor = 1.0 - self.anchor_depth.float().clamp(max=max_depth) / max_depth
            anchor_factor = anchor_factor * depth_factor.clamp_min(0.1)
            offset_anchor_ids = torch.arange(self.get_anchor.shape[0], device="cuda").repeat_interleave(self.n_offsets)
            grow_scores = grow_scores * anchor_factor[offset_anchor_ids]
        
        self.anchor_growing(grow_scores, grad_threshold, offset_mask, source_scores=grow_scores, tree_stats=tree_stats)

        if self.use_error_aware_refinement:
            self.offset_error_accum[offset_mask] = 0
            self.offset_error_denom[offset_mask] = 0
            padding_offset_error = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_error_accum.shape[0], 1],
                                               dtype=self.offset_error_accum.dtype,
                                               device=self.offset_error_accum.device)
            self.offset_error_accum = torch.cat([self.offset_error_accum, padding_offset_error], dim=0)
            padding_offset_error_denom = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_error_denom.shape[0], 1],
                                                     dtype=self.offset_error_denom.dtype,
                                                     device=self.offset_error_denom.device)
            self.offset_error_denom = torch.cat([self.offset_error_denom, padding_offset_error_denom], dim=0)
            padding_anchor_error = torch.zeros([self.get_anchor.shape[0] - self.anchor_error_accum.shape[0], 1],
                                               dtype=self.anchor_error_accum.dtype,
                                               device=self.anchor_error_accum.device)
            self.anchor_error_accum = torch.cat([self.anchor_error_accum, padding_anchor_error], dim=0)
            padding_anchor_error_denom = torch.zeros([self.get_anchor.shape[0] - self.anchor_error_denom.shape[0], 1],
                                                     dtype=self.anchor_error_denom.dtype,
                                                     device=self.anchor_error_denom.device)
            self.anchor_error_denom = torch.cat([self.anchor_error_denom, padding_anchor_error_denom], dim=0)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 

        if self.use_error_aware_refinement and self.anchor_error_denom.numel() == self.opacity_accum.numel():
            anchor_error_mean = self.anchor_error_accum / self.anchor_error_denom.clamp_min(1.0)
            observed_anchor_error = (self.anchor_error_denom > 0).squeeze(dim=1)
            if observed_anchor_error.sum() > 0:
                mean_anchor_error = anchor_error_mean[observed_anchor_error].mean().clamp_min(1e-6)
                high_error_anchor = anchor_error_mean.squeeze(dim=1) > mean_anchor_error * self.error_prune_keep_ratio
                prune_mask = torch.logical_and(prune_mask, ~high_error_anchor)

        if self.use_tree_anchor_refinement:
            tree_stats = self._compute_subtree_stats()
            subtree_error = tree_stats["subtree_error_mean"]
            subtree_grad = tree_stats["subtree_grad_mean"]
            subtree_visit = tree_stats["subtree_visit_count"]
            error_observed = tree_stats["subtree_error_count"] > 0
            grad_observed = tree_stats["subtree_grad_count"] > 0
            if error_observed.sum() > 0:
                global_error = subtree_error[error_observed].mean().clamp_min(1e-6)
                high_subtree_error = error_observed & (subtree_error > global_error * self.tree_error_keep_ratio)
            else:
                high_subtree_error = torch.zeros_like(prune_mask)
            if grad_observed.sum() > 0:
                global_grad = subtree_grad[grad_observed].mean().clamp_min(1e-12)
                active_subtree_grad = grad_observed & (subtree_grad > global_grad)
            else:
                active_subtree_grad = torch.zeros_like(prune_mask)
            mature_subtree = subtree_visit > check_interval * success_threshold
            protect_by_tree = mature_subtree & (high_subtree_error | active_subtree_grad)
            prune_mask = torch.logical_and(prune_mask, ~protect_by_tree)
            if not self.tree_nonleaf_prune:
                prune_mask = torch.logical_and(prune_mask, self.anchor_children_count == 0)
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        if self.use_error_aware_refinement:
            offset_error_accum = self.offset_error_accum.view([-1, self.n_offsets])[~prune_mask]
            offset_error_accum = offset_error_accum.view([-1, 1])
            offset_error_denom = self.offset_error_denom.view([-1, self.n_offsets])[~prune_mask]
            offset_error_denom = offset_error_denom.view([-1, 1])
            del self.offset_error_accum
            del self.offset_error_denom
            self.offset_error_accum = offset_error_accum
            self.offset_error_denom = offset_error_denom
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
            if self.use_error_aware_refinement:
                self.anchor_error_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.anchor_error_denom[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if self.use_error_aware_refinement:
            temp_anchor_error_accum = self.anchor_error_accum[~prune_mask]
            temp_anchor_error_denom = self.anchor_error_denom[~prune_mask]
            del self.anchor_error_accum
            del self.anchor_error_denom
            self.anchor_error_accum = temp_anchor_error_accum
            self.anchor_error_denom = temp_anchor_error_denom

        if self.use_tree_anchor_refinement and prune_mask.shape[0] > 0:
            self._remap_anchor_tree_after_prune(prune_mask)

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
