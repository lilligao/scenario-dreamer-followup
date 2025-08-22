from pathlib import Path
import pickle
import hydra
import os
import sys
sys.path.append("/mnt/efs/users/lili.gao/Repos/scenario-dreamer-followup")
from cfgs.config import CONFIG_PATH
import yaml
import shutil
from tqdm import tqdm
from typing import List
import random
import gzip

cam_order = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']
image_root = "/data_nuplan/nuplan/dataset/nuplan-v1.1/sensor_blobs"
sledge_preprocessed_data_folder = "autoencoder_cache_with_all_sam"
out_data_folder = "scenario_dreamer_nuplan_with_all_sam"
def find_feature_paths(root_path, feature_name):
    """Find all paths to the specified feature files in the given root path."""
    file_paths: List[Path] = []
    for log_path in root_path.iterdir():
        if log_path.name == "metadata":
            continue
        for scenario_type_path in log_path.iterdir():
            for token_path in scenario_type_path.iterdir():
                feature_path = token_path / f"{feature_name}.gz"
                if feature_path.is_file():
                    file_paths.append(token_path / feature_name)

    return file_paths

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    print("Creating train/val/test splits for SLEDGE autoencoder cache files...")
    autoencoder_cache_path = f"/opt/dlami/nvme/sledge_workspace/caches/{sledge_preprocessed_data_folder}" # os.path.join(cfg.scratch_root, 'exp/caches/autoencoder_cache')
    path = Path(autoencoder_cache_path)
    file_paths = find_feature_paths(path, 'sledge_raw')
    print("Number of files: ", len(file_paths))
    
    # we use the same train/val/test split as SLEDGE.
    yaml_file_path = os.path.join(cfg.project_root, 'metadata/sledge_files/nuplan.yaml')
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    
    # Extract the directories
    log_splits = yaml_data.get('log_splits', {})
    train_dirs = log_splits.get('train', [])
    val_dirs = log_splits.get('val', [])
    test_dirs = log_splits.get('test', [])

    # Define destination directories
    train_dest_sledge_raw = os.path.join(cfg.dataset_root, f'{out_data_folder}/sledge_raw/train')
    val_dest_sledge_raw = os.path.join(cfg.dataset_root, f'{out_data_folder}/sledge_raw/val')
    test_dest_sledge_raw = os.path.join(cfg.dataset_root, f'{out_data_folder}/sledge_raw/test')
    train_dest_map_id = os.path.join(cfg.dataset_root, f'{out_data_folder}/map_id/train')
    val_dest_map_id = os.path.join(cfg.dataset_root, f'{out_data_folder}/map_id/val')
    test_dest_map_id = os.path.join(cfg.dataset_root, f'{out_data_folder}/map_id/test')

    # Create the destination directories if they don't exist
    print(f"Saving files to {train_dest_sledge_raw}, {val_dest_sledge_raw}, {test_dest_sledge_raw}, {train_dest_map_id}, {val_dest_map_id}, {test_dest_map_id}")
    os.makedirs(train_dest_sledge_raw, exist_ok=True)
    os.makedirs(val_dest_sledge_raw, exist_ok=True)
    os.makedirs(test_dest_sledge_raw, exist_ok=True)
    os.makedirs(train_dest_map_id, exist_ok=True)
    os.makedirs(val_dest_map_id, exist_ok=True)
    os.makedirs(test_dest_map_id, exist_ok=True)

    train_files = []
    val_files = []
    test_files = []
    for file_path in tqdm(file_paths):
        path_partitioned = str(file_path).split("/")
        split_id_index = path_partitioned.index(sledge_preprocessed_data_folder) + 1
        split_id = path_partitioned[split_id_index]

        raw_file_path = str(file_path) + ".gz"
        with gzip.open(raw_file_path, 'rb') as f:
            data = pickle.load(f)

        file_exists = []
        cam_infos = data.get('cam_infos', {})
        for key, value in cam_infos.items():
            filename_jpg = value.get('filename_jpg', '')
            file_exists.append(os.path.exists(os.path.join(image_root, filename_jpg)))
        if False in file_exists:
            #print(f"Skipping {file_path} due to missing camera images.")
            continue
        if len(cam_infos) != 8:
            print(f"Skipping {file_path} due to incorrect number of cameras: {len(cam_infos)}. Expected 8.")
            continue

        if split_id in train_dirs:
            train_files.append(file_path)
        elif split_id in val_dirs:
            val_files.append(file_path)
        elif split_id in test_dirs:
            test_files.append(file_path)
    
    def copy_files(file_list, sledge_raw_destination, map_id_destination):
        for i, file_name in enumerate(tqdm(file_list)):
            sledge_raw_gz_path = str(file_name) + '.gz'
            map_id_gz_path = str(file_name)[:-len('sledge_raw')] + 'map_id.gz'
            
            identifier = sledge_raw_gz_path.split("/")[-2]
            
            sledge_raw_destination_path = os.path.join(sledge_raw_destination, identifier + '.gz')
            map_id_destination_path = os.path.join(map_id_destination, identifier + '.gz')
            shutil.copy(sledge_raw_gz_path, sledge_raw_destination_path)
            shutil.copy(map_id_gz_path, map_id_destination_path)

    # Copy train, val, and test files
    copy_files(train_files, train_dest_sledge_raw, train_dest_map_id)
    copy_files(val_files, val_dest_sledge_raw, val_dest_map_id)
    copy_files(test_files, test_dest_sledge_raw, test_dest_map_id)

    nuplan_dataset_dict = {
        'train_files': [os.path.relpath( str(i), autoencoder_cache_path) for i in train_files],
        'val_files': [os.path.relpath(str(i), autoencoder_cache_path) for i in val_files],
        'test_files': [os.path.relpath(str(i), autoencoder_cache_path) for i in test_files],
        'len_train_files': len(train_files),
        'len_val_files': len(val_files),
        'len_test_files': len(test_files),
    }
    with open(os.path.join(cfg.project_root, 'metadata', 'nuplan_dataset.pkl'), 'wb') as f:
        pickle.dump(nuplan_dataset_dict, f)

    print("Done.")
    # Create nuplan eval set
    print("Creating nuplan eval set (for computing metrics)...")
    random.seed(42)

    map_id_path = os.path.join(cfg.dataset_root, 'scenario_dreamer_nuplan', 'map_id', 'test')
    test_files = os.listdir(map_id_path)

    test_files_dict = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    for test_file in tqdm(test_files):
        test_file_path = os.path.join(map_id_path, test_file)
        with gzip.open(test_file_path, 'rb') as f:
            map_id_dict = pickle.load(f)

        test_files_dict[map_id_dict['id'].item()].append(test_file)

    for i in range(4):
        random.shuffle(test_files_dict[i])

    print("Number of files in each city:", 
          [len(test_files_dict[i]) for i in range(4)])

    # list of 12500 files from each city that forms the nuplan test set
    nuplan_test_files = test_files_dict[0][:12500] + test_files_dict[1][:12500] + test_files_dict[2][:12500] + test_files_dict[3][:12500]
    # This ensures files are not ordered by city
    random.shuffle(nuplan_test_files)

    assert len(nuplan_test_files) == 50000

    nuplan_test_dict = {
        'files': nuplan_test_files
    }

    with open(os.path.join(cfg.project_root, 'metadata', 'nuplan_eval_set.pkl'), 'wb') as f:
        pickle.dump(nuplan_test_dict, f)

    print("Done.")

main()