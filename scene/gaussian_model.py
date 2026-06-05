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
        self.use_residue_tracking = False
        self.residue_ema = 0.9
        self.residue_min_den = 1.0
        self.residue_div_eps = 1e-8
        self.anchor_residue_num_ema = torch.empty(0)
        self.anchor_residue_den_ema = torch.empty(0)
        self.anchor_residue_seen = torch.empty(0)
        self._anchor_uid = torch.empty(0, dtype=torch.long)
        self._next_anchor_uid = 0

        self.use_adaptive_k = False
        self.use_error_field = False
        self.adaptive_k_min = 1
        self.adaptive_k_init = n_offsets
        self.adaptive_min_seen_grow = 1
        self.adaptive_min_seen_shrink = 5
        self.adaptive_min_contrib_grow = 1e-8
        self.adaptive_residue_low_percentile = 0.30
        self.adaptive_residue_high_percentile = 0.80
        self.adaptive_den_low_percentile = 0.20
        self.adaptive_den_high_percentile = 0.80
        self.adaptive_grow_threshold = 0.75
        self.adaptive_den_shrink_threshold = -1.0
        self.adaptive_shrink_threshold = 0.2
        self.adaptive_grow_hysteresis = 2
        self.adaptive_shrink_hysteresis = 5
        self.anchor_active_offsets = torch.empty(0, dtype=torch.long)
        self.anchor_visibility_ema = torch.empty(0)
        self.adaptive_residue_num_window = torch.empty(0)
        self.adaptive_residue_den_window = torch.empty(0)
        self.adaptive_seen_window = torch.empty(0)
        self.adaptive_high_count = torch.empty(0, dtype=torch.long)
        self.adaptive_low_count = torch.empty(0, dtype=torch.long)

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

    def _ensure_anchor_uid(self):
        anchor_count = int(self.get_anchor.shape[0])
        device = self.get_anchor.device
        if self._anchor_uid.numel() != anchor_count:
            self._anchor_uid = torch.arange(anchor_count, dtype=torch.long, device=device).view(-1, 1)
            self._next_anchor_uid = anchor_count
        else:
            self._anchor_uid = self._anchor_uid.to(device=device, dtype=torch.long).view(-1, 1)
            if self._anchor_uid.numel() > 0:
                self._next_anchor_uid = max(self._next_anchor_uid, int(self._anchor_uid.max().item()) + 1)

    @property
    def get_anchor_uid(self):
        self._ensure_anchor_uid()
        return self._anchor_uid

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        state = {
            "anchor": self._anchor,
            "offset": self._offset,
            "anchor_feat": self._anchor_feat,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer": self.optimizer.state_dict(),
            "mlp_opacity": self.mlp_opacity.state_dict(),
            "mlp_cov": self.mlp_cov.state_dict(),
            "mlp_color": self.mlp_color.state_dict(),
        }
        if self.use_feat_bank:
            state["mlp_feature_bank"] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0 and self.embedding_appearance is not None:
            state["embedding_appearance"] = self.embedding_appearance.state_dict()
        if self._anchor_uid.numel() == self.get_anchor.shape[0]:
            state["anchor_uid"] = self._anchor_uid
            state["next_anchor_uid"] = torch.tensor(self._next_anchor_uid, device=self._anchor_uid.device)
        for name in [
            "anchor_residue_num_ema",
            "anchor_residue_den_ema",
            "anchor_residue_seen",
            "anchor_active_offsets",
            "anchor_visibility_ema",
            "adaptive_residue_num_window",
            "adaptive_residue_den_window",
            "adaptive_seen_window",
            "adaptive_high_count",
            "adaptive_low_count",
        ]:
            value = getattr(self, name, None)
            if value is not None and value.numel() > 0:
                state[name] = value
        return state
    
    def restore(self, model_args, training_args):
        if not isinstance(model_args, dict):
            raise ValueError("Unsupported legacy checkpoint format: expected dict model state")

        self._anchor = nn.Parameter(model_args["anchor"].requires_grad_(True))
        self._offset = nn.Parameter(model_args["offset"].requires_grad_(True))
        self._anchor_feat = nn.Parameter(model_args["anchor_feat"].requires_grad_(True))
        self._scaling = nn.Parameter(model_args["scaling"].requires_grad_(True))
        self._rotation = nn.Parameter(model_args["rotation"].requires_grad_(True))
        self._opacity = nn.Parameter(model_args["opacity"].requires_grad_(True))
        self.max_radii2D = model_args["max_radii2D"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]

        self.mlp_opacity.load_state_dict(model_args["mlp_opacity"])
        self.mlp_cov.load_state_dict(model_args["mlp_cov"])
        self.mlp_color.load_state_dict(model_args["mlp_color"])
        if self.use_feat_bank and "mlp_feature_bank" in model_args:
            self.mlp_feature_bank.load_state_dict(model_args["mlp_feature_bank"])
        if self.appearance_dim > 0 and "embedding_appearance" in model_args:
            self.embedding_appearance.load_state_dict(model_args["embedding_appearance"])

        self.training_setup(training_args)
        for name in [
            "anchor_residue_num_ema",
            "anchor_residue_den_ema",
            "anchor_residue_seen",
            "anchor_active_offsets",
            "anchor_visibility_ema",
            "adaptive_residue_num_window",
            "adaptive_residue_den_window",
            "adaptive_seen_window",
            "adaptive_high_count",
            "adaptive_low_count",
        ]:
            if name in model_args:
                setattr(self, name, model_args[name])
        if "anchor_uid" in model_args and model_args["anchor_uid"].numel() == self.get_anchor.shape[0]:
            self._anchor_uid = model_args["anchor_uid"].to(device=self.get_anchor.device, dtype=torch.long).view(-1, 1)
            if "next_anchor_uid" in model_args:
                self._next_anchor_uid = int(model_args["next_anchor_uid"].item())
            elif self._anchor_uid.numel() > 0:
                self._next_anchor_uid = int(self._anchor_uid.max().item()) + 1
            else:
                self._next_anchor_uid = 0
        else:
            self._ensure_anchor_uid()
        self.optimizer.load_state_dict(model_args["optimizer"])

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
    def get_anchor_residue(self):
        if self.anchor_residue_num_ema.numel() == 0:
            return torch.empty(0, device=self.get_anchor.device)
        return self.anchor_residue_num_ema / self.anchor_residue_den_ema.clamp_min(self.residue_div_eps)

    def get_active_offset_mask(self, anchor_mask=None):
        device = self.get_anchor.device
        if (not self.use_adaptive_k) or self.anchor_active_offsets.numel() == 0:
            count = int(anchor_mask.sum().item()) if anchor_mask is not None else self.get_anchor.shape[0]
            return torch.ones((count, self.n_offsets), dtype=torch.bool, device=device)

        active_offsets = self.anchor_active_offsets
        if anchor_mask is not None:
            active_offsets = active_offsets[anchor_mask]
        offset_ids = torch.arange(self.n_offsets, device=device).view(1, -1)
        return offset_ids < active_offsets.view(-1, 1)
    
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


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.use_adaptive_k = getattr(training_args, "use_adaptive_k", False)
        self.use_error_field = getattr(training_args, "use_error_field", False)
        self.use_residue_tracking = (
            getattr(training_args, "use_residue_tracking", False)
            or self.use_adaptive_k
            or self.use_error_field
        )
        self.residue_ema = getattr(training_args, "residue_ema", 0.9)
        self.residue_min_den = getattr(training_args, "residue_min_den", 1.0)
        self.residue_div_eps = getattr(training_args, "residue_div_eps", 1e-8)
        self.adaptive_k_min = max(1, min(int(getattr(training_args, "adaptive_k_min", 1)), self.n_offsets))
        self.adaptive_k_init = max(self.adaptive_k_min, min(int(getattr(training_args, "adaptive_k_init", self.n_offsets)), self.n_offsets))
        self.adaptive_min_seen_grow = max(1, int(getattr(training_args, "adaptive_min_seen_grow", 1)))
        self.adaptive_min_seen_shrink = max(1, int(getattr(training_args, "adaptive_min_seen_shrink", 5)))
        self.adaptive_min_contrib_grow = float(getattr(training_args, "adaptive_min_contrib_grow", 1e-8))
        self.adaptive_residue_low_percentile = float(getattr(training_args, "adaptive_residue_low_percentile", 0.30))
        self.adaptive_residue_high_percentile = float(getattr(training_args, "adaptive_residue_high_percentile", 0.80))
        self.adaptive_den_low_percentile = float(getattr(training_args, "adaptive_den_low_percentile", 0.20))
        self.adaptive_den_high_percentile = float(getattr(training_args, "adaptive_den_high_percentile", 0.80))
        self.adaptive_grow_threshold = float(getattr(training_args, "adaptive_grow_threshold", 0.75))
        self.adaptive_den_shrink_threshold = float(getattr(training_args, "adaptive_den_shrink_threshold", -1.0))
        self.adaptive_shrink_threshold = float(getattr(training_args, "adaptive_shrink_threshold", 0.2))
        self.adaptive_grow_hysteresis = max(1, int(getattr(training_args, "adaptive_grow_hysteresis", 2)))
        self.adaptive_shrink_hysteresis = max(1, int(getattr(training_args, "adaptive_shrink_hysteresis", 5)))

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self._ensure_anchor_uid()
        if self.use_residue_tracking:
            self.anchor_residue_num_ema = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.anchor_residue_den_ema = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.anchor_residue_seen = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        else:
            self.anchor_residue_num_ema = torch.empty(0, device="cuda")
            self.anchor_residue_den_ema = torch.empty(0, device="cuda")
            self.anchor_residue_seen = torch.empty(0, device="cuda")

        if self.use_adaptive_k:
            self.anchor_active_offsets = torch.full(
                (self.get_anchor.shape[0], 1), self.adaptive_k_init, dtype=torch.long, device="cuda")
            self.anchor_visibility_ema = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.adaptive_residue_num_window = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.adaptive_residue_den_window = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.adaptive_seen_window = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.adaptive_high_count = torch.zeros((self.get_anchor.shape[0], 1), dtype=torch.long, device="cuda")
            self.adaptive_low_count = torch.zeros((self.get_anchor.shape[0], 1), dtype=torch.long, device="cuda")
        else:
            self.anchor_active_offsets = torch.empty(0, dtype=torch.long, device="cuda")
            self.anchor_visibility_ema = torch.empty(0, device="cuda")
            self.adaptive_residue_num_window = torch.empty(0, device="cuda")
            self.adaptive_residue_den_window = torch.empty(0, device="cuda")
            self.adaptive_seen_window = torch.empty(0, device="cuda")
            self.adaptive_high_count = torch.empty(0, dtype=torch.long, device="cuda")
            self.adaptive_low_count = torch.empty(0, dtype=torch.long, device="cuda")

        
        
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
        l.append('active_offsets')
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
        if getattr(self, 'use_adaptive_k', False) and self.anchor_active_offsets.numel() > 0:
            active_offsets = self.anchor_active_offsets.detach().float().cpu().numpy()
        else:
            active_offsets = np.full((anchor.shape[0], 1), self.n_offsets, dtype=np.float32)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, active_offsets, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

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

        property_names = {p.name for p in plydata.elements[0].properties}
        if 'active_offsets' in property_names:
            active_offsets = np.asarray(plydata.elements[0]['active_offsets']).astype(np.float32)
            active_offsets = np.rint(active_offsets).astype(np.int64)
            active_offsets = np.clip(active_offsets, 1, self.n_offsets)[..., np.newaxis]
            self.anchor_active_offsets = torch.tensor(active_offsets, dtype=torch.long, device='cuda')
            self.use_adaptive_k = bool((self.anchor_active_offsets < self.n_offsets).any().item())
        else:
            self.anchor_active_offsets = torch.empty((0, 1), dtype=torch.long, device='cuda')
            self.use_adaptive_k = False
        self.anchor_visibility_ema = torch.empty((0, 1), dtype=torch.float, device='cuda')
        self.adaptive_high_count = torch.empty((0, 1), dtype=torch.long, device='cuda')
        self.adaptive_low_count = torch.empty((0, 1), dtype=torch.long, device='cuda')
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device='cuda').requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


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
                        gaussian_residue_num=None, gaussian_residue_den=None, neural_anchor_indices=None):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1
        visible_anchor_mask = anchor_visible_mask

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        if (self.use_residue_tracking and gaussian_residue_num is not None
                and gaussian_residue_den is not None and neural_anchor_indices is not None):
            gaussian_residue_num = gaussian_residue_num.detach().view(-1, 1)
            gaussian_residue_den = gaussian_residue_den.detach().view(-1, 1)
            observed_gaussian = gaussian_residue_den.squeeze(1) > 0
            if observed_gaussian.sum() > 0:
                anchor_indices = neural_anchor_indices[observed_gaussian].detach().long().view(-1, 1)
                batch_num = torch.zeros_like(self.anchor_residue_num_ema)
                batch_den = torch.zeros_like(self.anchor_residue_den_ema)
                batch_num.scatter_add_(0, anchor_indices, gaussian_residue_num[observed_gaussian])
                batch_den.scatter_add_(0, anchor_indices, gaussian_residue_den[observed_gaussian])
                if self.use_adaptive_k and self.adaptive_residue_num_window.numel() > 0:
                    self.adaptive_residue_num_window += batch_num
                    self.adaptive_residue_den_window += batch_den
                    self.adaptive_seen_window[visible_anchor_mask] += 1

                reliable_anchor = batch_den.squeeze(1) > self.residue_min_den
                if reliable_anchor.sum() > 0:
                    m = float(self.residue_ema)
                    self.anchor_residue_num_ema[reliable_anchor] = (
                        m * self.anchor_residue_num_ema[reliable_anchor]
                        + (1.0 - m) * batch_num[reliable_anchor]
                    )
                    self.anchor_residue_den_ema[reliable_anchor] = (
                        m * self.anchor_residue_den_ema[reliable_anchor]
                        + (1.0 - m) * batch_den[reliable_anchor]
                    )
                    self.anchor_residue_seen[reliable_anchor] += 1

        

    def _reset_adaptive_window_stats(self):
        if self.adaptive_residue_num_window.numel() == 0:
            return
        self.adaptive_residue_num_window.zero_()
        self.adaptive_residue_den_window.zero_()
        self.adaptive_seen_window.zero_()

    def adjust_adaptive_k(self):
        if (not self.use_adaptive_k) or self.anchor_active_offsets.numel() == 0:
            return {}
        if (self.adaptive_residue_num_window.numel() == 0
                or self.adaptive_residue_den_window.numel() == 0
                or self.adaptive_seen_window.numel() == 0):
            return {}

        den = self.adaptive_residue_den_window.squeeze(1)
        seen_count = self.adaptive_seen_window.squeeze(1)
        residue = (self.adaptive_residue_num_window / self.adaptive_residue_den_window.clamp_min(self.residue_div_eps)).squeeze(1)
        active = self.anchor_active_offsets.squeeze(1)
        observed = seen_count >= 1
        valid_count = int(observed.sum().item())
        if valid_count < 2:
            return {"valid": valid_count, "grow": 0, "shrink": 0}

        residue_norm = torch.zeros_like(residue)
        residue_valid = torch.logical_and(observed, den > self.adaptive_min_contrib_grow)
        if residue_valid.sum() >= 2:
            valid_residue = residue[residue_valid]
            q_low = min(max(self.adaptive_residue_low_percentile, 0.0), 1.0)
            q_high = min(max(self.adaptive_residue_high_percentile, 0.0), 1.0)
            r_low = torch.quantile(valid_residue, q_low)
            r_high = torch.quantile(valid_residue, q_high)
            residue_norm = torch.clamp((residue - r_low) / (r_high - r_low + self.residue_div_eps), 0.0, 1.0)

        den_norm = torch.zeros_like(den)
        observed_den = den[observed]
        if observed_den.numel() >= 2:
            q_low = min(max(self.adaptive_den_low_percentile, 0.0), 1.0)
            q_high = min(max(self.adaptive_den_high_percentile, 0.0), 1.0)
            d_low = torch.quantile(observed_den, q_low)
            d_high = torch.quantile(observed_den, q_high)
            den_norm = torch.clamp((den - d_low) / (d_high - d_low + self.residue_div_eps), 0.0, 1.0)

        grow_valid = (
            (seen_count >= self.adaptive_min_seen_grow)
            & (den > self.adaptive_min_contrib_grow)
            & (active < self.n_offsets)
        )
        high = grow_valid & (residue_norm > self.adaptive_grow_threshold)

        shrink_enabled = self.adaptive_den_shrink_threshold >= 0.0
        shrink_valid = (
            shrink_enabled
            & (seen_count >= self.adaptive_min_seen_shrink)
            & (active > self.adaptive_k_min)
        )
        low = shrink_valid & (den_norm < self.adaptive_den_shrink_threshold)

        self.adaptive_high_count[high, 0] += 1
        self.adaptive_high_count[grow_valid & ~high, 0] = 0
        self.adaptive_low_count[low, 0] += 1
        self.adaptive_low_count[shrink_valid & ~low, 0] = 0

        grow = (self.adaptive_high_count.squeeze(1) >= self.adaptive_grow_hysteresis) & (active < self.n_offsets)
        shrink = (self.adaptive_low_count.squeeze(1) >= self.adaptive_shrink_hysteresis) & (active > self.adaptive_k_min)

        grow_count = int(grow.sum().item())
        shrink_count = int(shrink.sum().item())
        if grow_count > 0:
            grow_idx = torch.nonzero(grow, as_tuple=False).squeeze(1)
            new_slot = self.anchor_active_offsets[grow_idx, 0]
            src_slot = torch.clamp(new_slot - 1, min=0)
            base_offset = self._offset.data[grow_idx, src_slot].clone()
            noise_scale = 0.01
            self._offset.data[grow_idx, new_slot] = base_offset + torch.randn_like(base_offset) * noise_scale
            self.anchor_active_offsets[grow_idx, 0] += 1
            self.adaptive_high_count[grow_idx, 0] = 0
            self.adaptive_low_count[grow_idx, 0] = 0

        if shrink_count > 0:
            shrink_idx = torch.nonzero(shrink, as_tuple=False).squeeze(1)
            self.anchor_active_offsets[shrink_idx, 0] -= 1
            self.adaptive_high_count[shrink_idx, 0] = 0
            self.adaptive_low_count[shrink_idx, 0] = 0

        stats = {
            "valid": valid_count,
            "grow": grow_count,
            "shrink": shrink_count,
            "mean_k": float(self.anchor_active_offsets.float().mean().item()),
            "mean_den": float(den[observed].mean().item()),
            "mean_seen": float(seen_count[observed].mean().item()),
        }
        self._reset_adaptive_window_stats()
        return stats

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

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
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

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

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

                self._ensure_anchor_uid()
                new_uid = torch.arange(
                    self._next_anchor_uid,
                    self._next_anchor_uid + new_opacities.shape[0],
                    dtype=torch.long,
                    device=self._anchor_uid.device,
                ).view(-1, 1)
                self._next_anchor_uid += int(new_opacities.shape[0])
                self._anchor_uid = torch.cat([self._anchor_uid, new_uid], dim=0)

                if self.use_residue_tracking:
                    zeros = torch.zeros([new_opacities.shape[0], 1], device="cuda").float()
                    self.anchor_residue_num_ema = torch.cat([self.anchor_residue_num_ema, zeros.clone()], dim=0)
                    self.anchor_residue_den_ema = torch.cat([self.anchor_residue_den_ema, zeros.clone()], dim=0)
                    self.anchor_residue_seen = torch.cat([self.anchor_residue_seen, zeros], dim=0)

                if self.use_adaptive_k:
                    new_active = torch.full([new_opacities.shape[0], 1], self.adaptive_k_init, dtype=torch.long, device="cuda")
                    self.anchor_active_offsets = torch.cat([self.anchor_active_offsets, new_active], dim=0)
                    zeros = torch.zeros([new_opacities.shape[0], 1], device="cuda").float()
                    self.anchor_visibility_ema = torch.cat([self.anchor_visibility_ema, zeros.clone()], dim=0)
                    self.adaptive_residue_num_window = torch.cat([self.adaptive_residue_num_window, zeros.clone()], dim=0)
                    self.adaptive_residue_den_window = torch.cat([self.adaptive_residue_den_window, zeros.clone()], dim=0)
                    self.adaptive_seen_window = torch.cat([self.adaptive_seen_window, zeros.clone()], dim=0)
                    zero_count = torch.zeros([new_opacities.shape[0], 1], dtype=torch.long, device="cuda")
                    self.adaptive_high_count = torch.cat([self.adaptive_high_count, zero_count.clone()], dim=0)
                    self.adaptive_low_count = torch.cat([self.adaptive_low_count, zero_count], dim=0)

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
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
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
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            if self.use_residue_tracking:
                self.anchor_residue_num_ema[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.anchor_residue_den_ema[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.anchor_residue_seen[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
            if self.use_adaptive_k:
                self.anchor_visibility_ema[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.adaptive_residue_num_window[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.adaptive_residue_den_window[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.adaptive_seen_window[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.adaptive_high_count[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], dtype=torch.long, device="cuda")
                self.adaptive_low_count[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], dtype=torch.long, device="cuda")
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if self.use_residue_tracking:
            temp_residue_num = self.anchor_residue_num_ema[~prune_mask]
            temp_residue_den = self.anchor_residue_den_ema[~prune_mask]
            temp_residue_seen = self.anchor_residue_seen[~prune_mask]
            del self.anchor_residue_num_ema
            del self.anchor_residue_den_ema
            del self.anchor_residue_seen
            self.anchor_residue_num_ema = temp_residue_num
            self.anchor_residue_den_ema = temp_residue_den
            self.anchor_residue_seen = temp_residue_seen

        if self._anchor_uid.numel() == prune_mask.shape[0]:
            temp_anchor_uid = self._anchor_uid[~prune_mask]
            del self._anchor_uid
            self._anchor_uid = temp_anchor_uid

        if self.use_adaptive_k:
            temp_active_offsets = self.anchor_active_offsets[~prune_mask]
            temp_visibility = self.anchor_visibility_ema[~prune_mask]
            temp_adaptive_num = self.adaptive_residue_num_window[~prune_mask]
            temp_adaptive_den = self.adaptive_residue_den_window[~prune_mask]
            temp_adaptive_seen = self.adaptive_seen_window[~prune_mask]
            temp_high_count = self.adaptive_high_count[~prune_mask]
            temp_low_count = self.adaptive_low_count[~prune_mask]
            del self.anchor_active_offsets
            del self.anchor_visibility_ema
            del self.adaptive_residue_num_window
            del self.adaptive_residue_den_window
            del self.adaptive_seen_window
            del self.adaptive_high_count
            del self.adaptive_low_count
            self.anchor_active_offsets = temp_active_offsets
            self.anchor_visibility_ema = temp_visibility
            self.adaptive_residue_num_window = temp_adaptive_num
            self.adaptive_residue_den_window = temp_adaptive_den
            self.adaptive_seen_window = temp_adaptive_seen
            self.adaptive_high_count = temp_high_count
            self.adaptive_low_count = temp_low_count

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
