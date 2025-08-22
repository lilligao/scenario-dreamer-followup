import json
import pandas as pd

path_root = "/opt/dlami/nvme/sledge_workspace/caches/autoencoder_cache_with_all_lili_temp_r100"
json_file_name = "sequence_token_mapping.json"
path_train_json = f"{path_root}/{json_file_name}"
path_test_json = f"{path_root}_test/{json_file_name}"

path_json_output = f"{path_root}_train_test/{json_file_name}"


# Load JSON files
with open(path_train_json, "r") as f1, open(path_test_json, "r") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Merge (data2 overwrites data1 if keys overlap)
merged = {**data1, **data2}

# Save merged JSON
with open(path_json_output, "w") as f:
    json.dump(merged, f, indent=4)

