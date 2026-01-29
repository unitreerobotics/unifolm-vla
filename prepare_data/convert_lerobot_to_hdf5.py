"""
Script lerobot to h5.
# --repo-id     Your unique repo ID on Hugging Face Hub
# --output_dir  Save path to h5 file

python unitree_lerobot/utils/convert_lerobot_to_h5.py \
    --repo-id 401_g1_robot_shortest_150/bag_lerobot \
    --root /home/jiang/datasets/unitree_vla/lerobot2.0_format_dataset/401_g1_robot_shortest_150/bag_lerobot \
    --output_dir /home/jiang/datasets/unitree_vla/hdf5_format_dataset/filter_401_g1/bag
"""

import os
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotDataProcessor:
    def __init__(self, repo_id: str, root: str = None, image_dtype: str = "to_unit8") -> None:
        self.image_dtype = image_dtype
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")

    def process_episode(self, episode_index: int) -> dict:
        """Process a single episode to extract camera images, state, and action."""
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()

        episode = defaultdict(list)
        cameras = defaultdict(list)

        for step_idx in tqdm(
            range(from_idx, to_idx), desc=f"Episode {episode_index}", position=1, leave=False, dynamic_ncols=True
        ):

            step = self.dataset[step_idx]

            image_dict = {
                key.split(".")[2]: np.transpose(
                    (value.numpy() * 255).astype(np.uint8), (1, 2, 0)
                )
                for key, value in step.items()
                if key.startswith("observation.image") and len(key.split(".")) >= 3
            }


            for key, value in image_dict.items():
                if self.image_dtype == "to_unit8":
                    cameras[key].append(value)
                elif self.image_dtype == "to_bytes":
                    success, encoded_img = cv2.imencode(".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    if not success:
                        raise ValueError(f"Image encoding failed for key: {key}")
                    cameras[key].append(np.void(encoded_img.tobytes()))

            cam_height, cam_width = next(iter(image_dict.values())).shape[:2]

            obs_left_gripper = step['observation.left_gripper'].unsqueeze(0)
            obs_right_gripper = step['observation.right_gripper'].unsqueeze(0)
            action_left_gripper = step['action.left_gripper'].unsqueeze(0)
            action_right_gripper = step['action.right_gripper'].unsqueeze(0)


            state_list = [step['observation.left_arm'], step['observation.right_arm'], obs_right_gripper, obs_left_gripper, step['observation.body'][12:15]]
            state = np.concatenate(state_list)

            ee_state_list = [step['observation.left_ee'], step['observation.right_ee'], obs_right_gripper, obs_left_gripper, step['observation.body'][12:15]]
            ee_state = np.concatenate(ee_state_list)

            action_list = [step['action.left_arm'], step['action.right_arm'], action_right_gripper, action_left_gripper, step['action.body'][3:6]]
            action = np.concatenate(action_list)

            ee_action_list = [step['action.left_ee'], step['action.right_ee'], action_right_gripper, action_left_gripper, step['action.body'][3:6]]
            ee_action = np.concatenate(ee_action_list)
            
            episode["state"].append(state)
            episode["action"].append(action)
            episode["ee_state"].append(ee_state)
            episode['ee_action'].append(ee_action)

        episode["cameras"] = cameras
        episode["task"] = step["task"]
        episode["episode_length"] = to_idx - from_idx

        # Data configuration for later use
        episode["data_cfg"] = {
            "camera_names": list(image_dict.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": np.squeeze(state.shape),
            "ee_state_dim": np.squeeze(ee_state.shape),
            "action_dim": np.squeeze(action.shape),
            "ee_action_dim": np.squeeze(ee_action.shape),
        }
        episode["episode_index"] = episode_index

        return episode


class H5Writer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_to_h5(self, episode: dict) -> None:
        """Write episode data to HDF5 file."""

        episode_length = episode["episode_length"]
        episode_index = episode["episode_index"]
        state = episode["state"]
        action = episode["action"]
        ee_state = episode["ee_state"]
        ee_action = episode["ee_action"]
        qvel = np.zeros_like(episode["state"])
        cameras = episode["cameras"]
        task = episode["task"]
        data_cfg = episode["data_cfg"]

        # Prepare data dictionary
        data_dict = {
            "/observations/qpos": [state],
            "/observations/ee_qpos": [ee_state],
            "/observations/qvel": [qvel],
            "/action": [action],
            "ee_action": [ee_action],
            **{f"/observations/images/{k}": [v] for k, v in cameras.items()},
        }

        h5_path = os.path.join(self.output_dir, f"episode_{episode_index}.hdf5")

        with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2, libver="latest") as root:
            # Set attributes
            root.attrs["sim"] = False

            # Create datasets
            obs = root.create_group("observations")
            image = obs.create_group("images")

            # Write camera images
            for cam_name, images in cameras.items():
                image.create_dataset(
                    cam_name,
                    shape=(episode_length, data_cfg["cam_height"], data_cfg["cam_width"], 3),
                    dtype="uint8",
                    chunks=(1, data_cfg["cam_height"], data_cfg["cam_width"], 3),
                    compression="gzip",
                )
                # root[f'/observations/images/{cam_name}'][...] = images

            # Write state and action data
            obs.create_dataset("qpos", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            obs.create_dataset("ee_qpos", (episode_length, data_cfg["ee_state_dim"]), dtype="float32", compression="gzip")
            obs.create_dataset("qvel", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            root.create_dataset("action", (episode_length, data_cfg["action_dim"]), dtype="float32", compression="gzip")
            root.create_dataset("ee_action", (episode_length, data_cfg["ee_action_dim"]), dtype="float32", compression="gzip")
            # Write metadata
            root.create_dataset("is_edited", (1,), dtype="uint8")
            substep_reasonings = root.create_dataset(
                "substep_reasonings", (episode_length,), dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip"
            )
            root.create_dataset("language_raw", data=task)
            substep_reasonings[:] = [task] * episode_length

            # Write additional data
            for name, array in data_dict.items():
                root[name][...] = array



def lerobot_to_h5(repo_id: str, output_dir: Path, root: str = None) -> None:
    """Main function to process and write LeRobot data to HDF5 format."""

    # Initialize data processor and H5 writer
    data_processor = LeRobotDataProcessor(
        repo_id, root, image_dtype="to_unit8"
    )  # image_dtype Options: "to_unit8", "to_bytes"
    h5_writer = H5Writer(output_dir)

    # Process each episode
    for episode_index in tqdm(
        range(data_processor.dataset.num_episodes), desc="Episodes", position=0, dynamic_ncols=True
    ):
        if os.path.exists(os.path.join(output_dir, f"episode_{episode_index}.hdf5")):
            print(f"Episode {episode_index} already exists")
            continue
        episode = data_processor.process_episode(episode_index)
        h5_writer.write_to_h5(episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--target_path", type=str, default="")
    args = parser.parse_args()
    repo_id = os.path.basename(args.data_path)
    root_path = args.data_path
    output_dir = args.target_path
    lerobot_to_h5(repo_id, output_dir, root_path)
