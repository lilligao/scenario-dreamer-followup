import torch
import torch.nn as nn

import numpy as np
from utils.dit_layers import FactorizedDiTBlock, FinalLayer, LabelEmbedder, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid, TwoLayerResMLP, FactorizedDiTBlockCond
from utils.pyg_helpers import get_indices_within_scene


class DiT3DCond(nn.Module):

    def __init__(self, cfg):
        super(DiT3DCond, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset

        
        self.emb_drop = nn.Dropout(self.cfg_model.dropout)
        # Condition on scene type
        self.scene_type_embedder = LabelEmbedder(self.cfg_dataset.num_map_ids * 2, self.cfg_model.hidden_dim, self.cfg_model.label_dropout)

        # Condition on number of agents and lanes
        self.num_agents_embedder = LabelEmbedder(self.cfg_dataset.max_num_agents + 1, self.cfg_model.hidden_dim, 0)
        self.num_lanes_embedder = LabelEmbedder(self.cfg_dataset.max_num_lanes + 1, self.cfg_model.hidden_dim, 0)
        
        # Diffusion timestep embedding
        self.t_embedder = TimestepEmbedder(self.cfg_model.hidden_dim)
        # Used because agent embedding is smaller than lane embedding
        self.downsample_c = nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.agent_hidden_dim)
        
        # Embed agent and lane latents
        self.lane_embedder = TwoLayerResMLP(self.cfg_model.lane_latent_dim, self.cfg_model.hidden_dim)
        self.agent_embedder = TwoLayerResMLP(self.cfg_model.agent_latent_dim, self.cfg_model.agent_hidden_dim)
        
        # These will be overwritten by sin/cos positional encodings
        self.pos_emb_lane = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_lanes, self.cfg_model.hidden_dim), requires_grad=False)
        self.pos_emb_agent = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_agents, self.cfg_model.agent_hidden_dim), requires_grad=False)
        
        # image projection: map img_feats dim -> lane hidden dim
        self.img_proj_lane = nn.Linear(self.cfg_model.img_feat_dim, self.cfg_model.hidden_dim)
        self.img_proj_agent = nn.Linear(self.cfg_model.img_feat_dim, self.cfg_model.agent_hidden_dim)

        self.pos_emb_img_lane = nn.Parameter(torch.zeros(self.cfg_model.max_img_tokens, self.cfg_model.hidden_dim), requires_grad=False)
        self.pos_emb_img_agent = nn.Parameter(torch.zeros(self.cfg_model.max_img_tokens, self.cfg_model.agent_hidden_dim), requires_grad=False)

        # factorized dit blocks
        self.blocks = nn.ModuleList([
            FactorizedDiTBlockCond(
                self.cfg_model.hidden_dim, 
                self.cfg_model.agent_hidden_dim, 
                self.cfg_model.num_heads, 
                self.cfg_model.agent_num_heads, 
                self.cfg_model.dropout, 
                mlp_ratio=4, 
                num_l2l_blocks=self.cfg_model.num_l2l_blocks,
                use_img_cross=self.cfg_model.image_conditioning,
                num_cross_attn = self.cfg_model.num_cross_attn,
                ) for _ in range(self.cfg_model.num_factorized_dit_blocks)
        ])

        # noise prediction heads
        self.pred_agent_noise = FinalLayer(self.cfg_model.agent_hidden_dim, self.cfg_model.agent_latent_dim)
        self.pred_lane_noise = FinalLayer(self.cfg_model.hidden_dim, self.cfg_model.lane_latent_dim)
        self.initialize_weights()


    def initialize_weights(self):
        """ Custom initialization for DiT model"""
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) lane and agent pos_embed by sin-cos embedding:
        pos_emb_lane = get_1d_sincos_pos_embed_from_grid(self.pos_emb_lane.shape[-1], np.arange(self.pos_emb_lane.shape[0]))
        self.pos_emb_lane.data.copy_(torch.from_numpy(pos_emb_lane).float())
        pos_emb_agent = get_1d_sincos_pos_embed_from_grid(self.pos_emb_agent.shape[-1], self.cfg_dataset.max_num_lanes + np.arange(self.pos_emb_agent.shape[0]))
        self.pos_emb_agent.data.copy_(torch.from_numpy(pos_emb_agent).float())

        # initialize BEV 1D sin/cos pos-emb over flattened grid
        pos_img_lane = get_1d_sincos_pos_embed_from_grid(self.pos_emb_img_lane.shape[-1], np.arange(self.cfg_model.max_img_tokens))
        self.pos_emb_img_lane.data.copy_(torch.from_numpy(pos_img_lane).float())
        pos_img_agent = get_1d_sincos_pos_embed_from_grid(self.pos_emb_img_agent.shape[-1], np.arange(self.cfg_model.max_img_tokens))
        self.pos_emb_img_agent.data.copy_(torch.from_numpy(pos_img_agent).float())

        # Initialize label embedding table:
        nn.init.normal_(self.scene_type_embedder.embedding_table.weight, std=0.02)

        # Initialize num lane and num agent embedding tables:
        nn.init.normal_(self.num_agents_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.num_lanes_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            for l2l_block in block.l2l_blocks:
                nn.init.constant_(l2l_block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(l2l_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a2l_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2l_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_agent_noise.linear.weight, 0)
        nn.init.constant_(self.pred_agent_noise.linear.bias, 0)

        nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_lane_noise.linear.weight, 0)
        nn.init.constant_(self.pred_lane_noise.linear.bias, 0)


    def _prep_batched_bev_tokens(self, img_feats: torch.Tensor):
        """
        img_feats: (B, 128, H=25, W=25)
        returns:
        bev_lane:  (B, L=H*W, D_lane)
        bev_agent: (B, L,       D_agent)
        pos_lane:  (1, L,       D_lane)  # broadcast to each scene
        """
        B, C, H, W = img_feats.shape
        # (B, H, W, C) -> (B, L, C)
        bev = img_feats.permute(0, 2, 3, 1).reshape(B, H*W, C)
        bev_lane  = self.img_proj_lane(bev)            # (B, L, D_lane)
        bev_agent = self.img_proj_agent(bev)           # (B, L, D_agent)
        # add pos emb (broadcast across batch)
        pos_lane = self.pos_emb_img_lane[: H * W, :].unsqueeze(0)   # (1, L, D_lane)
        pos_agent = self.pos_emb_img_agent[: H * W, :].unsqueeze(0) # (1, L, D_agent)
        bev_lane = bev_lane + pos_lane
        bev_agent = bev_agent + pos_agent
        return bev_lane, bev_agent
    
    def forward(self, 
                x_agent, 
                x_lane, 
                data, 
                agent_timestep, 
                lane_timestep, 
                img_feats=None,
                unconditional=False,):
        """ Forward pass of the DiT model."""
        # TODO: implement image conditioning here
        # Process BEV features if available
        if img_feats is not None:
            bev_lane_kv, bev_agent_kv = self._prep_batched_bev_tokens(img_feats)   # (B, L, D_lane), (B, L, D_agent)

        lane_idx_batch = get_indices_within_scene(data['lane'].batch)
        agent_idx_batch = get_indices_within_scene(data['agent'].batch)
        
        # add positional embeddings
        pos_emb_lane = self.pos_emb_lane[lane_idx_batch]
        pos_emb_agent = self.pos_emb_agent[agent_idx_batch]
        x_lane = self.lane_embedder(x_lane[:, 0]) + pos_emb_lane
        x_agent = self.agent_embedder(x_agent[:, 0]) + pos_emb_agent
        
        scene_idx = self.cfg_dataset.num_map_ids * data['lg_type'].long() + data['map_id'].long()
        scene_type = self.scene_type_embedder(scene_idx.long(), train=self.training, force_drop_ids=torch.ones_like(scene_idx) if unconditional else None)
        
        agent_batch = data['agent'].batch 
        lane_batch = data['lane'].batch
        agent_scene_type = scene_type[agent_batch]
        lane_scene_type = scene_type[lane_batch] 
        
        num_agents = data['num_agents'].long()
        num_lanes = data['num_lanes'].long()
        num_agents_emb = self.num_agents_embedder(num_agents, train=self.training)[agent_batch]
        num_lanes_emb = self.num_lanes_embedder(num_lanes, train=self.training)[lane_batch] 
        
        # embedding of timestep
        t = self.t_embedder(torch.cat([lane_timestep, agent_timestep], dim=-1))
        # embedding of number of agents and lanes
        n = torch.cat([num_lanes_emb, num_agents_emb], dim=0)
        # embedding of scene type
        y = torch.cat([lane_scene_type, agent_scene_type], dim=0)

        l2l_edge_index = data['lane', 'to', 'lane'].edge_index
        a2a_edge_index = data['agent', 'to', 'agent'].edge_index
        l2a_edge_index = data['lane', 'to', 'agent'].edge_index.clone()
        l2a_edge_index[1] = l2a_edge_index[1] + x_lane.shape[0]
        
        # conditioning vector for DiT block
        c = t + y + n
        # necessary for A2A and L2A attention
        c_small = self.downsample_c(c)
        
        # apply dropout
        x_lane = self.emb_drop(x_lane)
        x_agent = self.emb_drop(x_agent)
        
        # factorized dit block processing
        for block in self.blocks:
            x_lane, x_agent = block(
                x_lane, 
                x_agent, 
                c, 
                c_small, 
                l2l_edge_index, 
                a2a_edge_index, 
                l2a_edge_index,
                img_kv=bev_lane_kv,
                img_kv_agent=bev_agent_kv,
                lane_batch=data['lane'].batch,
                agent_batch=data['agent'].batch,
                )

        # decode the noise as in the original DiT paper
        c_lane = c[:x_lane.shape[0]]
        c_agent = c_small[x_lane.shape[0]:]
        x_lane = self.pred_lane_noise(x_lane, c_lane).unsqueeze(1)    
        x_agent = self.pred_agent_noise(x_agent, c_agent).unsqueeze(1)
        
        return x_agent, x_lane
        