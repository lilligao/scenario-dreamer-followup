import pickle

# Replace with the path to your pickle file
file_path = '/opt/dlami/nvme/scenario_dreamer_data/scenario_dreamer_ae_preprocess_waymo/test/testing.tfrecord-00000-of-00150_0_0_6.pkl'

# Open and load the pickle file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Print the loaded data
print(data)
