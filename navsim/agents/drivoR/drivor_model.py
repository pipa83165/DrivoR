from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .vggt_geometry import FrozenVggtGeometryTeacher, VggtGeometryProjector, cfg_get, tokens_per_camera
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)

        self.b2d = config.b2d

        vggt_geometry_cfg = cfg_get(config, "vggt_geometry", None)
        self.vggt_geometry_cfg = vggt_geometry_cfg
        self.vggt_geometry_enabled = bool(vggt_geometry_cfg and cfg_get(vggt_geometry_cfg, "enabled", False))
        self.vggt_geometry_mode = "normal"
        self.vggt_geometry_source = "cache"
        self.__dict__["_vggt_geometry_teacher"] = None
        if self.vggt_geometry_enabled:
            self.vggt_geometry_mode = str(cfg_get(vggt_geometry_cfg, "mode", "normal"))
            self.vggt_geometry_source = str(cfg_get(vggt_geometry_cfg, "source", "cache"))
            if self.vggt_geometry_mode not in ("normal", "shuffle", "noise", "drop"):
                raise ValueError(f"Unknown VGGT geometry mode: {self.vggt_geometry_mode}")
            if self.vggt_geometry_source not in ("cache", "online"):
                raise ValueError(f"Unknown VGGT geometry source: {self.vggt_geometry_source}")

            self.vggt_geometry_tokens_per_cam = tokens_per_camera(vggt_geometry_cfg)
            self.vggt_geometry_projector = VggtGeometryProjector(
                int(cfg_get(vggt_geometry_cfg, "vggt_dim", 2048)),
                config.tf_d_model,
                num_cams=4,
                tokens_per_cam=self.vggt_geometry_tokens_per_cam,
            )

            if self.vggt_geometry_source == "online":
                self.__dict__["_vggt_geometry_teacher"] = FrozenVggtGeometryTeacher(
                    cfg_get(vggt_geometry_cfg, "checkpoint_path"),
                    use_camera_token=bool(cfg_get(vggt_geometry_cfg, "use_camera_token", False)),
                    joint_forward=bool(cfg_get(vggt_geometry_cfg, "joint_forward", True)),
                )

    def _extend_memory_with_vggt_geometry(self, scene_features: torch.Tensor, features: Dict[str, Any]) -> torch.Tensor:
        # Drop mode physically removes geometry tokens in eval instead of appending zero keys.
        if self.vggt_geometry_mode == "drop" and not self.training:
            return scene_features

        if "vggt_geometry_tokens" in features:
            geo = features["vggt_geometry_tokens"].to(scene_features.device)
        elif self.vggt_geometry_source == "online":
            teacher_images = features.get("vggt_teacher_images", features.get("vggt_images"))
            if teacher_images is None:
                raise RuntimeError("VGGT geometry online source needs features['vggt_teacher_images']")
            teacher = self.__dict__.get("_vggt_geometry_teacher")
            if teacher is None:
                raise RuntimeError("VGGT geometry online source is enabled, but the frozen teacher was not built")
            if next(teacher.parameters()).device != scene_features.device:
                teacher.to(scene_features.device)
            geo = teacher(teacher_images.to(device=scene_features.device, dtype=torch.float32, non_blocking=True))
        else:
            raise RuntimeError("VGGT geometry source=cache is enabled, but features['vggt_geometry_tokens'] is missing")

        geo = self.vggt_geometry_projector(geo).to(scene_features.dtype)
        return torch.cat([scene_features, geo], dim=1)


    def forward(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        scene_features = []
        # image features
        if self.num_cams > 0:
            
            if "image" in features :
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)

        # Append frozen VGGT-Omega geometry tokens only at decoder memory.
        if self.vggt_geometry_enabled:
            scene_features = self._extend_memory_with_vggt_geometry(scene_features, features)
        vggt_geometry_memory_len = scene_features.shape[1]
        log.debug(f"Scene features - {scene_features.shape}")

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list
        if self.vggt_geometry_enabled:
            output["vggt_geometry_memory_len"] = vggt_geometry_memory_len

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(embedded_traj, scene_features)  # (B, N, d_model)
        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        pdm_score = (
        self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
        self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
        self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +    
        (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
        self._config.ep * pred_logit['ego_progress'].sigmoid()  
        + self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output



