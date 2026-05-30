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
import math
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

    def _encoded_dim(self, input_dim, num_freqs, enabled=None):
        if enabled is None:
            enabled = self.use_viewdist_pe
        if not enabled:
            return input_dim
        encoded_dim = input_dim if self.pe_include_input else 0
        encoded_dim += 2 * input_dim * max(0, int(num_freqs))
        return encoded_dim

    def _positional_encoding(self, values, num_freqs, include_input=True, enabled=None):
        if enabled is None:
            enabled = self.use_viewdist_pe
        if not enabled:
            return values

        encoded = []
        if include_input:
            encoded.append(values)

        num_freqs = max(0, int(num_freqs))
        if num_freqs > 0:
            freq_bands = (2.0 ** torch.arange(num_freqs, dtype=values.dtype, device=values.device)) * math.pi
            scaled = values.unsqueeze(-1) * freq_bands.view(*([1] * values.dim()), num_freqs)
            scaled = scaled.flatten(start_dim=1)
            encoded.extend([torch.sin(scaled), torch.cos(scaled)])

        if not encoded:
            return values.new_zeros((values.shape[0], 0))
        return torch.cat(encoded, dim=1)

    def encode_view(self, view):
        return self._positional_encoding(view, self.view_pe_freqs, self.pe_include_input)

    def encode_color_view(self, view):
        if self.use_color_view_pe:
            return self._positional_encoding(view, self.color_view_pe_freqs, self.pe_include_input, enabled=True)
        return self.encode_view(view)

    def encode_dist(self, dist):
        if not self.use_viewdist_pe:
            return dist
        scene_scale = max(float(self.spatial_lr_scale), 1e-6) if self.spatial_lr_scale else 1.0
        normalized_dist = torch.log1p(dist / scene_scale)
        return self._positional_encoding(normalized_dist, self.dist_pe_freqs, self.pe_include_input)


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
                 use_viewdist_pe : bool = False,
                 view_pe_freqs : int = 4,
                 dist_pe_freqs : int = 3,
                 use_color_view_pe : bool = False,
                 color_view_pe_freqs : int = 1,
                 pe_include_input : bool = True,
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
        self.use_viewdist_pe = use_viewdist_pe
        self.view_pe_freqs = max(0, int(view_pe_freqs))
        self.dist_pe_freqs = max(0, int(dist_pe_freqs))
        self.use_color_view_pe = use_color_view_pe
        self.color_view_pe_freqs = max(0, int(color_view_pe_freqs))
        self.pe_include_input = pe_include_input
        self.view_dim = self._encoded_dim(3, self.view_pe_freqs)
        self.dist_dim = self._encoded_dim(1, self.dist_pe_freqs)
        self.color_view_dim = self._encoded_dim(3, self.color_view_pe_freqs, enabled=True) if self.use_color_view_pe else self.view_dim
        self.featurebank_input_dim = self.view_dim + self.dist_dim

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)
        self.offset_error_accum = torch.empty(0)
        self.offset_error_denom = torch.empty(0)
        self.anchor_error_accum = torch.empty(0)
        self.anchor_error_denom = torch.empty(0)
        self.use_error_aware_refinement = False
        self.error_grow_weight = 1.0
        self.error_norm_clip = 4.0
        self.error_prune_keep_ratio = 1.0
        self.error_score_add_weight = 0.25
        self.error_visit_threshold_scale = 0.35
        self.use_component_refinement = False
        self.component_score_add_weight = 0.35
        self.component_norm_clip = 4.0
        self.component_budget_ratio = 0.01
        self.component_max_anchor_ratio = 2.80
        self.component_min_views = 2
        self.component_visit_threshold_scale = 0.25
        self.component_proposal_level = 0
        self.component_candidate_max_per_interval = 512
        self.initial_anchor_count = 0
        self.offset_component_accum = torch.empty(0)
        self.offset_component_denom = torch.empty(0)
        self.offset_component_proposal_accum = torch.empty(0)
        self.offset_component_proposal_denom = torch.empty(0)
        self.anchor_component_accum = torch.empty(0)
        self.anchor_component_denom = torch.empty(0)
        self.anchor_component_views = torch.empty(0)
        self.component_candidate_xyz = torch.empty(0)
        self.use_add_gaussian = False
        self.add_gaussian_budget_per_interval = 256
        self.add_gaussian_max_anchor_ratio = 1.10
        self.add_gaussian_jitter_voxels = 1.0
        self.add_gaussian_min_votes = 2
        self.add_gaussian_pending_limit = 4096
        self.add_gaussian_anchor_reference_count = 0
        self.add_gaussian_inserted_count = 0
        self.add_gaussian_pending_grid = torch.empty(0)
        self.add_gaussian_pending_votes = torch.empty(0)
        self.add_gaussian_pending_last_uid = torch.empty(0)

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
                nn.Linear(self.featurebank_input_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = self.dist_dim if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+self.view_dim+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = self.dist_dim if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+self.view_dim+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = self.dist_dim if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+self.color_view_dim+self.color_dist_dim+self.appearance_dim, feat_dim),
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
        )
    
    def restore(self, model_args, training_args):
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


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.use_error_aware_refinement = getattr(training_args, "use_error_aware_refinement", False)
        self.error_grow_weight = getattr(training_args, "error_grow_weight", 1.0)
        self.error_norm_clip = getattr(training_args, "error_norm_clip", 4.0)
        self.error_prune_keep_ratio = getattr(training_args, "error_prune_keep_ratio", 1.0)
        self.error_score_add_weight = getattr(training_args, "error_score_add_weight", 0.25)
        self.error_visit_threshold_scale = getattr(training_args, "error_visit_threshold_scale", 0.35)
        self.use_component_refinement = getattr(training_args, "use_component_refinement", False)
        self.component_score_add_weight = getattr(training_args, "component_score_add_weight", 0.35)
        self.component_norm_clip = getattr(training_args, "component_norm_clip", 4.0)
        self.component_budget_ratio = getattr(training_args, "component_budget_ratio", 0.01)
        self.component_max_anchor_ratio = getattr(training_args, "component_max_anchor_ratio", 2.80)
        self.component_min_views = getattr(training_args, "component_min_views", 2)
        self.component_visit_threshold_scale = getattr(training_args, "component_visit_threshold_scale", 0.25)
        self.component_proposal_level = getattr(training_args, "component_proposal_level", 0)
        self.component_candidate_max_per_interval = getattr(training_args, "component_candidate_max_per_interval", 512)
        self.use_add_gaussian = getattr(training_args, "use_add_gaussian", False)
        self.add_gaussian_budget_per_interval = getattr(training_args, "add_gaussian_budget_per_interval", 256)
        self.add_gaussian_max_anchor_ratio = getattr(training_args, "add_gaussian_max_anchor_ratio", 1.10)
        self.add_gaussian_jitter_voxels = getattr(training_args, "add_gaussian_jitter_voxels", 1.0)
        self.add_gaussian_min_votes = getattr(training_args, "add_gaussian_min_votes", 2)
        self.add_gaussian_pending_limit = getattr(training_args, "add_gaussian_pending_limit", 4096)
        self.add_gaussian_anchor_reference_count = 0
        self.add_gaussian_inserted_count = 0
        self.initial_anchor_count = self.get_anchor.shape[0]

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_error_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_error_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_error_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.anchor_error_denom = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_component_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_component_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_component_proposal_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_component_proposal_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_component_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.anchor_component_denom = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.anchor_component_views = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.component_candidate_xyz = torch.empty((0, 3), device="cuda")
        self.add_gaussian_pending_grid = torch.empty((0, 3), dtype=torch.int32, device="cuda")
        self.add_gaussian_pending_votes = torch.empty((0, 1), dtype=torch.int32, device="cuda")
        self.add_gaussian_pending_last_uid = torch.empty((0, 1), dtype=torch.long, device="cuda")

        
        
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
                        neural_errors=None, neural_error_filter=None, neural_offset_indices=None, neural_anchor_indices=None,
                        component_scores=None, component_score_filter=None,
                        component_proposal_scores=None, component_proposal_filter=None,
                        component_candidate_xyz=None,
                        add_gaussian_candidate_xyz=None, add_gaussian_view_uid=None):
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

        if self.use_component_refinement and component_scores is not None and neural_offset_indices is not None and neural_anchor_indices is not None:
            component_filter = update_filter if component_score_filter is None else torch.logical_and(update_filter, component_score_filter)
            if component_filter.sum() > 0:
                component_values = component_scores[component_filter].detach().view(-1, 1)
                positive_component = component_values.squeeze(1) > 0
                if positive_component.sum() > 0:
                    component_values = component_values[positive_component]
                    offset_indices = neural_offset_indices[component_filter][positive_component].detach().long().view(-1, 1)
                    anchor_indices = neural_anchor_indices[component_filter][positive_component].detach().long().view(-1, 1)
                    ones = torch.ones_like(component_values)
                    self.offset_component_accum.scatter_add_(0, offset_indices, component_values)
                    self.offset_component_denom.scatter_add_(0, offset_indices, ones)
                    self.anchor_component_accum.scatter_add_(0, anchor_indices, component_values)
                    self.anchor_component_denom.scatter_add_(0, anchor_indices, ones)
                    unique_anchor_indices = torch.unique(anchor_indices.view(-1)).view(-1, 1)
                    unique_ones = torch.ones_like(unique_anchor_indices, dtype=self.anchor_component_views.dtype)
                    self.anchor_component_views.scatter_add_(0, unique_anchor_indices, unique_ones)

        if self.use_component_refinement and self.component_proposal_level >= 1 and component_proposal_scores is not None and neural_offset_indices is not None:
            proposal_filter = update_filter if component_proposal_filter is None else torch.logical_and(update_filter, component_proposal_filter)
            if proposal_filter.sum() > 0:
                proposal_values = component_proposal_scores[proposal_filter].detach().view(-1, 1)
                positive_proposal = proposal_values.squeeze(1) > 0
                if positive_proposal.sum() > 0:
                    proposal_values = proposal_values[positive_proposal]
                    offset_indices = neural_offset_indices[proposal_filter][positive_proposal].detach().long().view(-1, 1)
                    ones = torch.ones_like(proposal_values)
                    self.offset_component_proposal_accum.scatter_add_(0, offset_indices, proposal_values)
                    self.offset_component_proposal_denom.scatter_add_(0, offset_indices, ones)

        if self.use_component_refinement and self.component_proposal_level >= 2 and component_candidate_xyz is not None and component_candidate_xyz.numel() > 0:
            candidate_xyz = component_candidate_xyz.detach().view(-1, 3)
            if self.component_candidate_xyz.numel() == 0:
                self.component_candidate_xyz = candidate_xyz
            else:
                self.component_candidate_xyz = torch.cat([self.component_candidate_xyz, candidate_xyz], dim=0)
            max_candidates = max(1, int(self.component_candidate_max_per_interval) * 4)
            if self.component_candidate_xyz.shape[0] > max_candidates:
                keep = torch.randperm(self.component_candidate_xyz.shape[0], device=self.component_candidate_xyz.device)[:max_candidates]
                self.component_candidate_xyz = self.component_candidate_xyz[keep]

        if self.use_add_gaussian and add_gaussian_candidate_xyz is not None and add_gaussian_view_uid is not None:
            self.accumulate_add_gaussian_candidates(add_gaussian_candidate_xyz, add_gaussian_view_uid)

        

        
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

    
    def anchor_growing(self, grads, threshold, offset_mask, max_new_anchors=None):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        total_added = 0
        for i in range(self.update_depth):
            if max_new_anchors is not None and total_added >= max_new_anchors:
                break
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

                if max_new_anchors is not None:
                    remaining = max_new_anchors - total_added
                    if remaining <= 0:
                        continue
                    if candidate_anchor.shape[0] > remaining:
                        keep_indices = torch.randperm(candidate_anchor.shape[0], device=candidate_anchor.device)[:remaining]
                        candidate_anchor = candidate_anchor[keep_indices]
                        new_scaling = new_scaling[keep_indices]
                        new_rotation = new_rotation[keep_indices]
                        new_opacities = new_opacities[keep_indices]
                        new_feat = new_feat[keep_indices]

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

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                total_added += candidate_anchor.shape[0]
                
        return total_added


    def add_component_candidate_anchors(self, candidate_xyz, max_new_anchors=None):
        if candidate_xyz is None or candidate_xyz.numel() == 0:
            return 0
        if max_new_anchors is not None and max_new_anchors <= 0:
            return 0

        finest_factor = max(self.update_init_factor // (self.update_hierachy_factor ** max(self.update_depth - 1, 0)), 1)
        cur_size = self.voxel_size * finest_factor
        candidate_grid = torch.round(candidate_xyz.detach() / cur_size).int()
        candidate_grid_unique = torch.unique(candidate_grid, dim=0)
        if candidate_grid_unique.shape[0] == 0:
            return 0

        grid_coords = torch.round(self.get_anchor / cur_size).int()
        chunk_size = 4096
        remove_duplicates_list = []
        for i in range(grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)):
            cur_remove_duplicates = (candidate_grid_unique.unsqueeze(1) == grid_coords[i * chunk_size:(i + 1) * chunk_size, :]).all(-1).any(-1).view(-1)
            remove_duplicates_list.append(cur_remove_duplicates)
        if remove_duplicates_list:
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            candidate_grid_unique = candidate_grid_unique[~remove_duplicates]
        if candidate_grid_unique.shape[0] == 0:
            return 0

        if max_new_anchors is not None and candidate_grid_unique.shape[0] > max_new_anchors:
            keep = torch.randperm(candidate_grid_unique.shape[0], device=candidate_grid_unique.device)[:max_new_anchors]
            candidate_grid_unique = candidate_grid_unique[keep]

        candidate_anchor = candidate_grid_unique.float() * cur_size
        nearest_feat = []
        for start in range(0, candidate_anchor.shape[0], 32):
            chunk = candidate_anchor[start:start + 32]
            distances = torch.cdist(chunk, self.get_anchor.detach())
            nearest = distances.argmin(dim=1)
            nearest_feat.append(self._anchor_feat.detach()[nearest])
        new_feat = torch.cat(nearest_feat, dim=0) if nearest_feat else torch.zeros((0, self.feat_dim), device='cuda')
        if new_feat.shape[0] == 0:
            return 0

        new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
        new_scaling = torch.log(new_scaling)
        new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
        new_rotation[:, 0] = 1.0
        new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device='cuda'))
        new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

        d = {
            'anchor': candidate_anchor,
            'scaling': new_scaling,
            'rotation': new_rotation,
            'anchor_feat': new_feat,
            'offset': new_offsets,
            'opacity': new_opacities,
        }

        self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
        self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
        torch.cuda.empty_cache()

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._anchor = optimizable_tensors['anchor']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        self._anchor_feat = optimizable_tensors['anchor_feat']
        self._offset = optimizable_tensors['offset']
        self._opacity = optimizable_tensors['opacity']
        return candidate_anchor.shape[0]

    def get_finest_voxel_size(self):
        finest_factor = max(self.update_init_factor // (self.update_hierachy_factor ** max(self.update_depth - 1, 0)), 1)
        return self.voxel_size * finest_factor

    def accumulate_add_gaussian_candidates(self, candidate_xyz, view_uid):
        if not self.use_add_gaussian or candidate_xyz is None or candidate_xyz.numel() == 0:
            return

        cur_size = self.get_finest_voxel_size()
        candidate_grid = torch.round(candidate_xyz.detach() / cur_size).int()
        candidate_grid = torch.unique(candidate_grid, dim=0)
        if candidate_grid.shape[0] == 0:
            return

        existing_grid = torch.round(self.get_anchor.detach() / cur_size).int()
        chunk_size = 4096
        duplicate_list = []
        for start in range(0, existing_grid.shape[0], chunk_size):
            duplicate = (candidate_grid.unsqueeze(1) == existing_grid[start:start + chunk_size]).all(-1).any(-1)
            duplicate_list.append(duplicate)
        if duplicate_list:
            duplicate_existing = reduce(torch.logical_or, duplicate_list)
            candidate_grid = candidate_grid[~duplicate_existing]
        if candidate_grid.shape[0] == 0:
            return

        view_uid_tensor = torch.full((candidate_grid.shape[0], 1), int(view_uid), dtype=torch.long, device=candidate_grid.device)
        new_votes = torch.ones((candidate_grid.shape[0], 1), dtype=torch.int32, device=candidate_grid.device)

        if self.add_gaussian_pending_grid.numel() == 0:
            self.add_gaussian_pending_grid = candidate_grid
            self.add_gaussian_pending_votes = new_votes
            self.add_gaussian_pending_last_uid = view_uid_tensor
        else:
            pending = self.add_gaussian_pending_grid
            matches = (candidate_grid.unsqueeze(1) == pending.unsqueeze(0)).all(-1)
            has_match = matches.any(dim=1)
            if has_match.any():
                candidate_ids = torch.nonzero(has_match, as_tuple=False).view(-1)
                pending_ids = matches[has_match].float().argmax(dim=1).long()
                different_view = self.add_gaussian_pending_last_uid[pending_ids].view(-1) != int(view_uid)
                if different_view.any():
                    pending_ids = pending_ids[different_view]
                    self.add_gaussian_pending_votes[pending_ids] += 1
                    self.add_gaussian_pending_last_uid[pending_ids] = int(view_uid)

            if (~has_match).any():
                self.add_gaussian_pending_grid = torch.cat([self.add_gaussian_pending_grid, candidate_grid[~has_match]], dim=0)
                self.add_gaussian_pending_votes = torch.cat([self.add_gaussian_pending_votes, new_votes[~has_match]], dim=0)
                self.add_gaussian_pending_last_uid = torch.cat([self.add_gaussian_pending_last_uid, view_uid_tensor[~has_match]], dim=0)

        max_pending = max(1, int(self.add_gaussian_pending_limit))
        if self.add_gaussian_pending_grid.shape[0] > max_pending:
            keep = torch.randperm(self.add_gaussian_pending_grid.shape[0], device=self.add_gaussian_pending_grid.device)[:max_pending]
            self.add_gaussian_pending_grid = self.add_gaussian_pending_grid[keep]
            self.add_gaussian_pending_votes = self.add_gaussian_pending_votes[keep]
            self.add_gaussian_pending_last_uid = self.add_gaussian_pending_last_uid[keep]

    def add_pending_add_gaussian_anchors(self, max_new_anchors=None):
        if (not self.use_add_gaussian or self.add_gaussian_pending_grid.numel() == 0 or
                max_new_anchors is not None and max_new_anchors <= 0):
            return 0

        ready = self.add_gaussian_pending_votes.view(-1) >= int(self.add_gaussian_min_votes)
        if ready.sum() == 0:
            return 0

        ready_indices = torch.nonzero(ready, as_tuple=False).view(-1)
        if max_new_anchors is not None and ready_indices.shape[0] > max_new_anchors:
            keep = torch.randperm(ready_indices.shape[0], device=ready_indices.device)[:max_new_anchors]
            ready_indices = ready_indices[keep]

        cur_size = self.get_finest_voxel_size()
        candidate_xyz = self.add_gaussian_pending_grid[ready_indices].float() * cur_size
        jitter_scale = float(cur_size) * float(self.add_gaussian_jitter_voxels)
        if jitter_scale > 0.0:
            candidate_xyz = candidate_xyz + (torch.rand_like(candidate_xyz) * 2.0 - 1.0) * jitter_scale
        added = self.add_component_candidate_anchors(candidate_xyz, max_new_anchors=max_new_anchors)

        keep_pending = torch.ones((self.add_gaussian_pending_grid.shape[0],), dtype=torch.bool, device=self.add_gaussian_pending_grid.device)
        keep_pending[ready_indices] = False
        self.add_gaussian_pending_grid = self.add_gaussian_pending_grid[keep_pending]
        self.add_gaussian_pending_votes = self.add_gaussian_pending_votes[keep_pending]
        self.add_gaussian_pending_last_uid = self.add_gaussian_pending_last_uid[keep_pending]
        return added


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        visit_threshold_scale = self.error_visit_threshold_scale if self.use_error_aware_refinement else 0.5
        if self.use_component_refinement:
            visit_threshold_scale = min(visit_threshold_scale, self.component_visit_threshold_scale)
        offset_mask = (self.offset_denom > check_interval * success_threshold * visit_threshold_scale).squeeze(dim=1)

        grow_scores = grads_norm
        if self.use_error_aware_refinement and self.offset_error_denom.numel() == self.offset_gradient_accum.numel():
            offset_error_mean = self.offset_error_accum / self.offset_error_denom.clamp_min(1.0)
            observed_error = (self.offset_error_denom > 0).squeeze(dim=1)
            if observed_error.sum() > 0:
                mean_error = offset_error_mean[observed_error].mean().clamp_min(1e-6)
                error_norm = (offset_error_mean.squeeze(dim=1) / mean_error).clamp(max=self.error_norm_clip)
                multiplicative_score = grads_norm * (1.0 + self.error_grow_weight * error_norm)
                additive_score = grad_threshold * self.error_score_add_weight * error_norm
                grow_scores = multiplicative_score + additive_score

        reliable_component_count = 0
        proposal_component_count = 0
        if self.use_component_refinement and self.offset_component_denom.numel() == self.offset_gradient_accum.numel():
            offset_component_mean = self.offset_component_accum / self.offset_component_denom.clamp_min(1.0)
            observed_component = (self.offset_component_denom > 0).squeeze(dim=1)
            reliable_component_count = int(observed_component.sum().item())
            if observed_component.sum() > 0:
                mean_component = offset_component_mean[observed_component].mean().clamp_min(1e-6)
                component_norm = (offset_component_mean.squeeze(dim=1) / mean_component).clamp(max=self.component_norm_clip)
                anchor_component_views = self.anchor_component_views.repeat_interleave(self.n_offsets, dim=0).squeeze(dim=1)
                consistent_component = anchor_component_views >= self.component_min_views
                component_add = grad_threshold * self.component_score_add_weight * component_norm
                grow_scores = grow_scores + component_add * consistent_component.float()
                offset_mask = torch.logical_or(offset_mask, torch.logical_and(consistent_component, observed_component))

        if self.use_component_refinement and self.component_proposal_level >= 1 and self.offset_component_proposal_denom.numel() == self.offset_gradient_accum.numel():
            proposal_mean = self.offset_component_proposal_accum / self.offset_component_proposal_denom.clamp_min(1.0)
            observed_proposal = (self.offset_component_proposal_denom > 0).squeeze(dim=1)
            proposal_component_count = int(observed_proposal.sum().item())
            if observed_proposal.sum() > 0:
                mean_proposal = proposal_mean[observed_proposal].mean().clamp_min(1e-6)
                proposal_norm = (proposal_mean.squeeze(dim=1) / mean_proposal).clamp(max=self.component_norm_clip)
                proposal_add = grad_threshold * self.component_score_add_weight * proposal_norm
                grow_scores = grow_scores + proposal_add
                offset_mask = torch.logical_or(offset_mask, observed_proposal)

        max_new_anchors = None
        if self.use_component_refinement and self.component_budget_ratio > 0:
            max_total_anchors = max(self.get_anchor.shape[0], int(self.initial_anchor_count * self.component_max_anchor_ratio))
            remaining_budget = max_total_anchors - self.get_anchor.shape[0]
            interval_budget = max(1, int(self.get_anchor.shape[0] * self.component_budget_ratio))
            max_new_anchors = max(0, min(interval_budget, remaining_budget))
        
        added_from_growth = self.anchor_growing(grow_scores, grad_threshold, offset_mask, max_new_anchors=max_new_anchors)
        added_from_candidates = 0
        if self.use_component_refinement and self.component_proposal_level >= 2 and self.component_candidate_xyz.numel() > 0:
            candidate_budget = int(self.component_candidate_max_per_interval)
            if max_new_anchors is not None:
                candidate_budget = min(candidate_budget, max(0, max_new_anchors - added_from_growth))
            added_from_candidates = self.add_component_candidate_anchors(self.component_candidate_xyz, max_new_anchors=candidate_budget)

        added_from_add_gaussian = 0
        if self.use_add_gaussian and self.add_gaussian_pending_grid.numel() > 0:
            if self.add_gaussian_anchor_reference_count <= 0:
                self.add_gaussian_anchor_reference_count = int(self.get_anchor.shape[0])
            add_ratio_budget = max(0, int(self.add_gaussian_anchor_reference_count * (self.add_gaussian_max_anchor_ratio - 1.0)))
            remaining_add_budget = max(0, add_ratio_budget - int(self.add_gaussian_inserted_count))
            interval_add_budget = min(int(self.add_gaussian_budget_per_interval), remaining_add_budget)
            added_from_add_gaussian = self.add_pending_add_gaussian_anchors(max_new_anchors=interval_add_budget)
            self.add_gaussian_inserted_count += int(added_from_add_gaussian)
            if added_from_add_gaussian > 0 or self.add_gaussian_pending_grid.numel() > 0:
                print(f"[add_gaussian] pending={self.add_gaussian_pending_grid.shape[0]} added={added_from_add_gaussian} add_total={self.add_gaussian_inserted_count} anchors={self.get_anchor.shape[0]}")

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

        if self.use_component_refinement:
            self.offset_component_accum[offset_mask] = 0
            self.offset_component_denom[offset_mask] = 0
            padding_offset_component = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_component_accum.shape[0], 1],
                                                   dtype=self.offset_component_accum.dtype,
                                                   device=self.offset_component_accum.device)
            self.offset_component_accum = torch.cat([self.offset_component_accum, padding_offset_component], dim=0)
            padding_offset_component_denom = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_component_denom.shape[0], 1],
                                                         dtype=self.offset_component_denom.dtype,
                                                         device=self.offset_component_denom.device)
            self.offset_component_denom = torch.cat([self.offset_component_denom, padding_offset_component_denom], dim=0)
            self.offset_component_proposal_accum[offset_mask] = 0
            self.offset_component_proposal_denom[offset_mask] = 0
            padding_offset_component_proposal = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_component_proposal_accum.shape[0], 1],
                                                            dtype=self.offset_component_proposal_accum.dtype,
                                                            device=self.offset_component_proposal_accum.device)
            self.offset_component_proposal_accum = torch.cat([self.offset_component_proposal_accum, padding_offset_component_proposal], dim=0)
            padding_offset_component_proposal_denom = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_component_proposal_denom.shape[0], 1],
                                                                  dtype=self.offset_component_proposal_denom.dtype,
                                                                  device=self.offset_component_proposal_denom.device)
            self.offset_component_proposal_denom = torch.cat([self.offset_component_proposal_denom, padding_offset_component_proposal_denom], dim=0)
            padding_anchor_component = torch.zeros([self.get_anchor.shape[0] - self.anchor_component_accum.shape[0], 1],
                                                   dtype=self.anchor_component_accum.dtype,
                                                   device=self.anchor_component_accum.device)
            self.anchor_component_accum = torch.cat([self.anchor_component_accum, padding_anchor_component], dim=0)
            padding_anchor_component_denom = torch.zeros([self.get_anchor.shape[0] - self.anchor_component_denom.shape[0], 1],
                                                         dtype=self.anchor_component_denom.dtype,
                                                         device=self.anchor_component_denom.device)
            self.anchor_component_denom = torch.cat([self.anchor_component_denom, padding_anchor_component_denom], dim=0)
            padding_anchor_component_views = torch.zeros([self.get_anchor.shape[0] - self.anchor_component_views.shape[0], 1],
                                                         dtype=self.anchor_component_views.dtype,
                                                         device=self.anchor_component_views.device)
            self.anchor_component_views = torch.cat([self.anchor_component_views, padding_anchor_component_views], dim=0)
            self.component_candidate_xyz = torch.empty((0, 3), device="cuda")
            if self.component_proposal_level > 0:
                print(f"[component] level={self.component_proposal_level} reliable_offsets={reliable_component_count} proposal_offsets={proposal_component_count} growth_added={added_from_growth} candidate_added={added_from_candidates} anchors={self.get_anchor.shape[0]}")
        
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

        if self.use_component_refinement and self.anchor_component_denom.numel() == self.opacity_accum.numel():
            anchor_component_mean = self.anchor_component_accum / self.anchor_component_denom.clamp_min(1.0)
            observed_anchor_component = (self.anchor_component_denom > 0).squeeze(dim=1)
            if observed_anchor_component.sum() > 0:
                mean_anchor_component = anchor_component_mean[observed_anchor_component].mean().clamp_min(1e-6)
                high_component_anchor = torch.logical_and(
                    anchor_component_mean.squeeze(dim=1) > mean_anchor_component,
                    self.anchor_component_views.squeeze(dim=1) >= self.component_min_views,
                )
                prune_mask = torch.logical_and(prune_mask, ~high_component_anchor)
        
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

        if self.use_component_refinement:
            offset_component_accum = self.offset_component_accum.view([-1, self.n_offsets])[~prune_mask]
            offset_component_accum = offset_component_accum.view([-1, 1])
            offset_component_denom = self.offset_component_denom.view([-1, self.n_offsets])[~prune_mask]
            offset_component_denom = offset_component_denom.view([-1, 1])
            offset_component_proposal_accum = self.offset_component_proposal_accum.view([-1, self.n_offsets])[~prune_mask]
            offset_component_proposal_accum = offset_component_proposal_accum.view([-1, 1])
            offset_component_proposal_denom = self.offset_component_proposal_denom.view([-1, self.n_offsets])[~prune_mask]
            offset_component_proposal_denom = offset_component_proposal_denom.view([-1, 1])
            del self.offset_component_accum
            del self.offset_component_denom
            del self.offset_component_proposal_accum
            del self.offset_component_proposal_denom
            self.offset_component_accum = offset_component_accum
            self.offset_component_denom = offset_component_denom
            self.offset_component_proposal_accum = offset_component_proposal_accum
            self.offset_component_proposal_denom = offset_component_proposal_denom
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
            if self.use_error_aware_refinement:
                self.anchor_error_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.anchor_error_denom[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
            if self.use_component_refinement:
                self.anchor_component_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.anchor_component_denom[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
                self.anchor_component_views[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
        
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

        if self.use_component_refinement:
            temp_anchor_component_accum = self.anchor_component_accum[~prune_mask]
            temp_anchor_component_denom = self.anchor_component_denom[~prune_mask]
            temp_anchor_component_views = self.anchor_component_views[~prune_mask]
            del self.anchor_component_accum
            del self.anchor_component_denom
            del self.anchor_component_views
            self.anchor_component_accum = temp_anchor_component_accum
            self.anchor_component_denom = temp_anchor_component_denom
            self.anchor_component_views = temp_anchor_component_views

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.color_view_dim+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, self.featurebank_input_dim).cuda()))
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
