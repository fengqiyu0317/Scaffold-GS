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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        # Optional error-aware anchor refinement. Disabled by default to keep
        # the original Scaffold-GS baseline unchanged.
        self.use_error_aware_refinement = False
        self.error_grow_weight = 0.5
        self.error_norm_clip = 3.0
        self.error_prune_keep_ratio = 1.0

        # Optional tree-aware refinement. This keeps the renderer unchanged and
        # uses parent/subtree metadata only during training-time density control.
        self.use_tree_anchor_refinement = False
        self.tree_max_depth = 6
        self.tree_grow_weight = 0.5
        self.tree_error_norm_clip = 3.0
        self.tree_error_keep_ratio = 1.0
        self.tree_child_base_cap = 4
        self.tree_child_high_cap = 12
        self.tree_nonleaf_prune = False

        # Optional V1.2 highlight-aware tree growing. This only affects
        # refinement attribution and tree growing; pruning stays V1.1-style.
        self.use_highlight_aware_refinement = False
        self.highlight_grow_weight = 0.5
        self.highlight_error_norm_clip = 3.0
        self.highlight_tree_weight = 0.5
        self.highlight_luma_threshold = 0.65
        self.highlight_local_contrast = 0.08

        # Optional V1 hotspot-aware density control. The hotspot field is a
        # training-only sparse 3D statistic and does not change rendering.
        self.use_hotspot_field = False
        self.hotspot_mode = "score_only"
        self.hotspot_start = 3000
        self.hotspot_until = 18000
        self.hotspot_update_interval = 100
        self.hotspot_grow_interval = 500
        self.hotspot_voxel_multiplier = 4.0
        self.hotspot_min_support_views = 2
        self.hotspot_min_view_angle_deg = 5.0
        self.hotspot_error_mean_multiplier = 1.5
        self.hotspot_max_pixels_per_view = 768
        self.hotspot_reproj_radius_px = 4.0
        self.hotspot_attribution_mode = "footprint_depth"
        self.hotspot_depth_radius_cap_px = 4
        self.hotspot_depth_min_weight = 1e-4
        self.hotspot_depth_min_radii = 1.0
        self.hotspot_thin_min_support_views = 2
        self.hotspot_highlight_min_support_views = 4
        self.hotspot_general_min_support_views = 3
        self.hotspot_thin_reproj_radius_px = 4.0
        self.hotspot_highlight_reproj_radius_px = 2.5
        self.hotspot_general_reproj_radius_px = 3.5
        self.hotspot_weight = 0.5
        self.hotspot_score_clip = 3.0
        self.hotspot_high_error_percentile = 95.0
        self.hotspot_highlight_luma_threshold = 0.65
        self.hotspot_highlight_deficit_threshold = 0.05
        self.hotspot_thin_luma_threshold = 0.45
        self.hotspot_thin_chroma_max = 0.12
        self.hotspot_edge_threshold = 0.08
        self.hotspot_min_anchor_count = 8
        self.hotspot_density_ratio_thresh = 0.7
        self.hotspot_add_budget_per_interval = 256
        self.hotspot_add_min_votes = 2
        self.hotspot_add_candidate_multiplier = 6
        self.hotspot_add_max_anchor_ratio = 1.10

        # Compatibility aliases for the experiment command naming used in notes.
        self.use_add_gaussian = False
        self.add_gaussian_mode = "hotspot"
        self.add_gaussian_budget_per_interval = -1
        self.add_gaussian_candidate_multiplier = -1
        self.add_gaussian_min_votes = -1
        self.add_gaussian_max_anchor_ratio = -1.0

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    args = Namespace(**merged_dict)
    if getattr(args, "use_add_gaussian", False):
        args.use_hotspot_field = True
        if str(getattr(args, "add_gaussian_mode", "hotspot")).lower() in ("hotspot", "add_gaussian"):
            args.hotspot_mode = "add_gaussian"
    if int(getattr(args, "add_gaussian_budget_per_interval", -1)) >= 0:
        args.hotspot_add_budget_per_interval = int(args.add_gaussian_budget_per_interval)
    if int(getattr(args, "add_gaussian_candidate_multiplier", -1)) >= 0:
        args.hotspot_add_candidate_multiplier = int(args.add_gaussian_candidate_multiplier)
    if int(getattr(args, "add_gaussian_min_votes", -1)) >= 0:
        args.hotspot_add_min_votes = int(args.add_gaussian_min_votes)
    if float(getattr(args, "add_gaussian_max_anchor_ratio", -1.0)) > 0:
        args.hotspot_add_max_anchor_ratio = float(args.add_gaussian_max_anchor_ratio)
    return args
