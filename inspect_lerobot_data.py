import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Replace with the name of your LeRobot dataset repository.
repo_id = "physical-intelligence/aloha_pen_uncap_diverse"

# Load the dataset from the hub or your local cache.
dataset = LeRobotDataset(repo_id)

print(f"Successfully loaded dataset: {repo_id}")
print("-" * 30)

# The dataset is a list of episodes. Let's inspect the first one.
# An episode is a dictionary where keys are the names of the data fields.
episode = dataset[0]

print("Fields available in an episode:")
for key, value in episode.items():
    # Convert to numpy array to easily access shape and dtype.
    print("Value: ", data_array := np.array(value))
    print(f"  - Key: '{key}'")
    print(f"    - Type: {type(data_array)}")
    print(f"    - Dtype: {data_array.dtype}")
    print(f"    - Shape: {data_array.shape}")

print("-" * 30)