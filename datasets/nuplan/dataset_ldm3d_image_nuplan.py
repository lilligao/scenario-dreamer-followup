import os
import sys
import glob
import hydra
import torch
import pickle
import random
import sys
from tqdm import tqdm
from cfgs.config import CONFIG_PATH
from typing import Any, Dict
import torch.multiprocessing as mp

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
torch.set_printoptions(threshold=100000)
import numpy as np
np.set_printoptions(suppress=True, threshold=sys.maxsize)

from utils.data_container import ScenarioDreamerData
from utils.torch_helpers import from_numpy
from utils.data_helpers import sample_latents, reorder_indices

from utils.cam_img_utils import trans_matrix, transform, project_cam_to_image, project_cam_to_image_nodrop, plot_projection_all_views, plot_topdown_lanes_and_agents, load_cam_views, get_3d_box_corners, transform_heading


class NuplanDatasetLDM3D(Dataset):
    def __init__(self, cfg: Any, split_name: str = "train") -> None:
        """Instantiate a :class:`NuplanDatasetLDM3D`.

        Parameters
        ----------
        cfg
            Hydra configuration object containing dataset configs (cfg.dataset in global config)
        split_name
            One of ``{"train", "val", "test"}`` selecting which split
            to load from ``cfg.dataset.dataset_path``.
        """
        super(NuplanDatasetLDM3D, self).__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.load_images = cfg.get('load_images', False)
        self.dataset_dir = os.path.join(self.cfg.dataset_path, f"{self.split_name}")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)

        self.files = sorted(glob.glob(self.dataset_dir + "/*.pkl"))
        self.dset_len = len(self.files)

        if self.load_images:
            self.image_root = f"{self.cfg.nuplan_data_root}/sensor_blobs"  # this should be the base path before filename_jpg
            self.cam_order = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']
            self.image_height_og = self.cfg.get('image_height_og', 256)
            self.image_width_og = self.cfg.get('image_width_og', 256)
            self.downsample_factor = self.cfg.get('downsample_factor', 1.0)
            self.image_height = self.cfg.get('image_height', int(self.image_height_og // self.downsample_factor))
            self.image_width = self.cfg.get('image_width', int(self.image_width_og // self.downsample_factor))


    def get_data(self, data, idx):
        """Return a sample for ldm training"""
        idx = data['idx']
        agent_states = data['agent_states']
        road_points = data['road_points']
        lane_mu = data['lane_mu']
        agent_mu = data['agent_mu']
        lane_log_var = data['lane_log_var']
        agent_log_var = data['agent_log_var']
        edge_index_lane_to_lane = data['edge_index_lane_to_lane']
        edge_index_lane_to_agent = data['edge_index_lane_to_agent']
        edge_index_agent_to_agent = data['edge_index_agent_to_agent']
        scene_type = data['scene_type']
        ego_state_og = data['ego_state_og']  # [ego_translation, ego_rotation, ego_dim, z_coord]
        if self.load_images:
            cam_infos = data['cam_infos']
        map_id = np.array([data['map_id']], dtype=int)
        num_lanes = lane_mu.shape[0]
        num_agents = agent_mu.shape[0]

        # apply recursive ordering
        agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane, agent_partition_mask, lane_partition_mask = reorder_indices(
            agent_mu, 
            agent_log_var, 
            lane_mu, 
            lane_log_var, 
            edge_index_lane_to_lane, 
            agent_states, 
            road_points, 
            scene_type,
            dataset='nuplan')
        edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)

        # sample for ldm training
        d = dict()
        d = ScenarioDreamerData()
        d['idx'] = idx
        d['num_lanes'] = num_lanes 
        d['num_agents'] = num_agents
        d['lg_type'] = scene_type
        d['map_id'] = from_numpy(map_id)
        d['agent'].x = from_numpy(agent_mu)
        d['lane'].x = from_numpy(lane_mu)
        d['agent'].partition_mask = from_numpy(agent_partition_mask)
        d['lane'].partition_mask = from_numpy(lane_partition_mask)
        d['agent'].log_var = from_numpy(agent_log_var)
        d['lane'].log_var = from_numpy(lane_log_var)
        d['agent'].latents, d['lane'].latents = sample_latents(
            d, 
            self.cfg.agent_latents_mean,
            self.cfg.agent_latents_std,
            self.cfg.lane_latents_mean,
            self.cfg.lane_latents_std,
            normalize=True) # sample normalized latents for training

        d['lane', 'to', 'lane'].edge_index = from_numpy(edge_index_lane_to_lane)
        d['agent', 'to', 'agent'].edge_index = from_numpy(edge_index_agent_to_agent)
        d['lane', 'to', 'agent'].edge_index = from_numpy(edge_index_lane_to_agent)
        d['ego_state_og'] = from_numpy(ego_state_og)
        if self.load_images:
            d['cam_infos'] = cam_infos
            if len(cam_infos) != 8:
                raise ValueError(f"Expected 8 cameras, but got {len(cam_infos)}. Please check the data extraction script.")

            cam_img_stack, T_cam_tf_stack, T_cam_tf_inv_stack, T_cam_ego_inv_stack, intrinsics_stack, widths, heights  = load_cam_views(
                cam_infos=cam_infos,
                cam_order= self.cam_order,
                image_root=self.image_root,
                do_undistortion=False,
                downsample_factor=self.downsample_factor
            )
            if scene_type == 1:
                cam_img_stack[:5, :, :, :] = 0 # [N, H, W, 3], overwrite front pixel values to 0 to keep same shape of cam_img_stack
            
            T_local2global = trans_matrix(ego_state_og[1], ego_state_og[0])

            # Precompute extrinsics for each camera
            T_extrinsics = [
                T_cam_tf_inv_stack[i] @ T_cam_ego_inv_stack[i] @ T_local2global
                for i in range(len(self.cam_order))
            ]
            
            d['cam_img_stack'] = from_numpy(cam_img_stack)
            d['T_cam_tf_stack'] = from_numpy(T_cam_tf_stack)
            d['T_cam_tf_inv_stack'] = from_numpy(T_cam_tf_inv_stack)
            d['T_cam_ego_inv_stack'] = from_numpy(T_cam_ego_inv_stack)
            d['T_extrinsics'] = from_numpy(np.stack(T_extrinsics))
            d['intrinsics_stack'] = from_numpy(intrinsics_stack)
            d['img_widths'] = from_numpy(widths)
            d['img_heights'] = from_numpy(heights)
            # TODO:
            # - apply augmentations (resizing, random crop, ect). Maybe we can add it in the dataloader (self.transforms, adn pass it to it)
            # here /mnt/efs/users/samuele.ruffino/dev/scenario-dreamer-followup/datamodules/nuplan/nuplan_datamodule_ldm3d_image.py
            # - add vizualization function
        return d


    def get(self, idx: int):
        raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
        raw_path = os.path.join(self.dataset_dir, f'{raw_file_name}.pkl')
        with open(raw_path, 'rb') as f:
            data = pickle.load(f)
        
        d = self.get_data(data, idx)
        
        return d


    def len(self):
        return self.dset_len

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    cfg = cfg.ldm
    dset = NuplanDatasetLDM3D(cfg.dataset, split_name='train')

    print(cfg.dataset.dataset_path)
    
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    print(len(dset))

    if not os.path.exists(cfg.dataset.latent_stats_path):
        cfg.dataset.agent_latents_mean = 0.0
        cfg.dataset.agent_latents_std = 1.0
        cfg.dataset.lane_latents_mean = 0.0
        cfg.dataset.lane_latents_std = 1.0
    
    dloader = DataLoader(dset, 
               batch_size=1024, 
               shuffle=True, 
               num_workers=0,
               pin_memory=True,
               drop_last=True)

    agent_latents_all = []
    lane_latents_all = []
    for i, d in enumerate(tqdm(dloader)):
        agent_latents, lane_latents = sample_latents(
            d, 
            cfg.dataset.agent_latents_mean,
            cfg.dataset.agent_latents_std,
            cfg.dataset.lane_latents_mean,
            cfg.dataset.lane_latents_std,
            normalize=False)
        
        agent_latents_all.append(agent_latents)
        lane_latents_all.append(lane_latents)

        if i == 5:
            break
    
    agent_latents_all = torch.cat(agent_latents_all, dim=0)
    lane_latents_all = torch.cat(lane_latents_all, dim=0)

    print(agent_latents_all.mean(), agent_latents_all.std())
    print(lane_latents_all.mean(), lane_latents_all.std())



if __name__ == "__main__":
    main()