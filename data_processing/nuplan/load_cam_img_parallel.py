import os
import json
import pickle
from tqdm import tqdm
from multiprocessing import Process, Manager
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper

# Constants
THREAD_COUNT = 2
split = "mini" 
NUPLAN_DATA_ROOT = "/data_nuplan/nuplan/dataset"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
NUPLAN_MAPS_ROOT = "/data_nuplan/nuplan/dataset/maps"
NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/{split}/"
OUTPUT_DIR = f"/opt/dlami/nvme/scenario_dreamer_data/lidartoken2cam_caminfos_backup/{split}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON
with open('/opt/dlami/nvme/sledge_workspace/identifier_to_scene.json', 'r') as f:
    data = json.load(f)

unique_log_db_names = list(set(data.values()))
unique_log_db_names.reverse()
set_lidartokens = set(data.keys())

wrapper = NuPlanDBWrapper(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_version=NUPLAN_MAP_VERSION
    )

included_logs = wrapper.log_names

def process_logs(log_db_names, set_lidartokens, progress_queue, wrapper):
    for log_db_name in log_db_names:
        output_path = os.path.join(OUTPUT_DIR, f"{log_db_name}.pkl")
        if os.path.exists(output_path):
            progress_queue.put(1)
            print(f"Skipping {log_db_name} as output file already exists.")
            continue
        if log_db_name not in included_logs:
            print(f"Skipping {log_db_name} as it is not in the included logs.")
            progress_queue.put(1)
            continue
        try:
            log_db = wrapper.get_log_db(log_db_name)
        except Exception as e:
            print(f"[ERROR] {log_db_name}: {e}")
            progress_queue.put(1)
            continue

        lidartoken_to_image = {}
        for img in log_db.image:
            if img.camera is None:
                continue
            token = img.lidar_pc.token
            if token not in set_lidartokens:
                continue
            if token not in lidartoken_to_image:
                lidartoken_to_image[token] = {
                    'timestamp': img.lidar_pc.timestamp,
                    'ego_pose_token': img.lidar_pc.ego_pose_token,
                    'scene_token': img.lidar_pc.scene_token,
                    'next_token': img.lidar_pc.next_token,
                    'prev_token': img.lidar_pc.prev_token,
                    'cams': {}
                }

            cam = img.camera.channel
            cam_i = img.camera
            lidartoken_to_image[token]['cams'][cam] = {
                'token': img.token,
                'timestamp': img.timestamp,
                'filename_jpg': img.filename_jpg,
                'ego_pose_token': img.ego_pose_token,
                'next_token': img.next_token,
                'prev_token': img.prev_token,
                'camera_token': cam_i.token,
                'width': cam_i.width,
                'height': cam_i.height,
                'camera_model': cam_i.model,
                'intrinsic': list(cam_i.intrinsic),
                'translation': list(cam_i.translation),
                'rotation': list(cam_i.rotation),
                'distortion': list(cam_i.distortion),
                'model': cam_i.model,
                'log_token': cam_i.log_token,
                'channel': cam_i.channel
            }

        with open(output_path, 'wb') as f:
            pickle.dump(lidartoken_to_image, f)

        progress_queue.put(1)  # Report progress

def run_with_progress():
    manager = Manager()
    progress_queue = manager.Queue()

    # Partition workload
    partitions = [[] for _ in range(THREAD_COUNT)]
    for i, log_db in enumerate(unique_log_db_names):
        partitions[i % THREAD_COUNT].append(log_db)

    processes = []
    for part in partitions:
        p = Process(target=process_logs, args=(part, set_lidartokens, progress_queue, wrapper))
        p.start()
        processes.append(p)

    # TQDM progress monitor
    with tqdm(total=len(unique_log_db_names), desc="Processing NuPlan logs") as pbar:
        completed = 0
        while completed < len(unique_log_db_names):
            progress_queue.get()
            pbar.update(1)
            completed += 1

    for p in processes:
        p.join()

if __name__ == "__main__":
    run_with_progress()
