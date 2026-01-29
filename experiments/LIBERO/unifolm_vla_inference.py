from collections import deque
from typing import Optional, Sequence
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Dict,Any
import numpy as np
from pathlib import Path
from unifolm_vla.rlds_dataloader.constants import NormalizationType,ACTION_PROPRIO_NORMALIZATION_TYPE
from unifolm_vla.model.framework.share_tools import read_mode_config
from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Unifolm_VLA_Inference:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        image_size: list[int] = [224, 224],
        use_bf16: bool = True,
        vlm_pretrained_path: str = None,
    ) -> None:
        
        # build client to connect server policy
        self.vla = Unifolm_VLA.from_pretrained(policy_ckpt_path, vlm_pretrained_path=vlm_pretrained_path)
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        self.vla = self.vla.to(DEVICE).eval()
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.image_size = image_size
        self.horizon = horizon 
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None


    def step( self, obs_inputs: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

        qwen_inputs = {key: value.to(DEVICE) for key, value in obs_inputs.items() if key != "state" and key != "image"}
        
        qwen_inputs["state"] = self.normalize_proprio(obs_inputs["state"], self.state_norm_stats)
        qwen_inputs["state"] = torch.from_numpy(qwen_inputs["state"]).unsqueeze(0).to(DEVICE)
        # qwen_inputs["image"] = [obs_inputs["image"]]
        # import pdb; pdb.set_trace()
        normalized_actions = self.vla.predict_action(qwen_inputs)
        
        normalized_actions = normalized_actions["normalized_actions"][0]
        
        actions = self.unnormalize_action(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        # actions[:,-1] = 1 - actions[:,-1]


        return actions

    @staticmethod
    def unnormalize_action(normalized_actions: np.ndarray, action_norm_stats: Dict[str, Any]) -> np.ndarray:
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

        actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
                normalized_actions,
            )

        return actions

    @staticmethod
    def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
            proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
            proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        normalized_proprio = np.clip(
            np.where(
                mask,
                2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                proprio,
            ),
            a_min=-1.0,
            a_max=1.0,
        )

        return normalized_proprio
    
    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = Unifolm_VLA_Inference._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = Unifolm_VLA_Inference._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["proprio"]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key