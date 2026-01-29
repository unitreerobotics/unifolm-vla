import os
import sys
import logging
import argparse
import torch
import json
import time
import base64
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import json_numpy
json_numpy.patch()
import tensorflow as tf
from qwen_vl_utils import process_vision_info
import traceback
from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
unifolm_vla_IMAGE_SIZE = 224  


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()

def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (unifolm_vla_IMAGE_SIZE, unifolm_vla_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")

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

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
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

class Unifolm_VLA_Server:
    """FastAPI服务器, 用于VLA模型推理"""
    
    def __init__(self, args):
        self.args = args
        logging.info("Loading VLA model from: %s", args.ckpt_path)
        
        # TODO: should auto detect framework from model path
        vla = baseframework.from_pretrained(args.ckpt_path, vlm_pretrained_path=args.vlm_pretrained_path)

        if args.use_bf16:
            logging.info("Converting model to bfloat16")
            vla = vla.to(torch.bfloat16)
        
        vla = vla.to("cuda").eval()
        self.vla = vla        
        self.norm_stats_action = vla.norm_stats[self.args.unnorm_key]['action']
        self.norm_stats_proprio = vla.norm_stats[self.args.unnorm_key]['proprio']
        self.processor = vla.qwen_vl_interface.processor
        logging.info("Model loaded successfully")

    def prepare_images_for_vla(self, images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
        """
        Prepare images for VLA input by resizing and cropping as needed.

        Args:
            images: List of input images as numpy arrays
            cfg: Configuration object with parameters

        Returns:
            List[Image.Image]: Processed images ready for the model
        """
        processed_images = []

        for image in images:
            # Validate format
            check_image_format(image)

            # Resize if needed
            if image.shape != (unifolm_vla_IMAGE_SIZE, unifolm_vla_IMAGE_SIZE, 3):
                image = resize_image_for_policy(image, unifolm_vla_IMAGE_SIZE)

            # Convert to PIL image
            pil_image = Image.fromarray(image).convert("RGB")

            # Apply center crop if configured
            if cfg.center_crop:
                pil_image = center_crop_image(pil_image)

            processed_images.append(pil_image)

        return processed_images
        
    
    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            t1 = time.time()
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observations = payload['observations']
            all_images = []
            for observation in observations:    
                all_images.append(observation["full_image"])
            for observation in observations:
                all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])
            instruction = observations[0]["instruction"]
            
            if observations[0].get("task_name", None) is not None:
                task_name = observations[0].get("task_name", None)
                self.norm_stats_action = self.vla.norm_stats[task_name]['action']
                self.norm_stats_proprio = self.vla.norm_stats[task_name]['proprio']

            # Process images
            all_images = self.prepare_images_for_vla(all_images, self.args)
            lang = instruction.lower()
            text = f"The task is \"{lang}\"."
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image", "image": img}
                            for img in all_images
                        ],
                        {"type": "text", "text": text},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            batch_input = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            proprios = []
            for observation in observations:
                proprios.append(observation["state"])
                
            batch_input["state"] = torch.from_numpy(normalize_proprio(np.stack(proprios, axis=0), self.norm_stats_proprio)).unsqueeze(0).to(DEVICE)

            batch_input["input_ids"] = batch_input["input_ids"].to(DEVICE)
            batch_input["attention_mask"] = batch_input["attention_mask"].to(DEVICE)
            batch_input["pixel_values"] = batch_input["pixel_values"].to(DEVICE)
            batch_input["image_grid_thw"] = batch_input["image_grid_thw"].to(DEVICE)
            
            action = self.vla.predict_action(
                qwen_inputs=batch_input,
            )

            action = unnormalize_action(action['normalized_actions'][0], self.norm_stats_action)
            inference_time = time.time() - t1
            logging.info(f"VLA inference time: {inference_time:.3f}s")
            
            print(f"get_vla_action time: {time.time() - t1}")
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"
        
    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        """启动FastAPI服务器"""
        logging.info("Creating FastAPI server...")
        self.app = FastAPI(
            title="VLA Model Server",
            description="VLA (Vision-Language-Action) Model Inference API",
            version="1.0.0"
        )

        self.app.post("/act")(self.get_server_action)
        
        logging.info(f"Starting server on http://{host}:{port}")
        logging.info(f"API endpoint: POST http://{host}:{port}/act")
        logging.info("Press Ctrl+C to stop the server")

        uvicorn.run(self.app, host=host, port=port, log_level="info")


def deploy(args):
    """部署VLA模型服务器"""
    server = Unifolm_VLA_Server(args)
    server.run(host=args.host, port=args.port)

def build_argparser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="部署VLA模型为FastAPI服务器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/path/to/your/ckpt.pt",
        help="模型检查点路径或HuggingFace模型名称"
    )
    parser.add_argument(
        "--vlm_pretrained_path",
        type=str,
        default=None,
        help="VLM模型检查点路径或HuggingFace模型名称"
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default="new_embodiment",
        help="数据集名称"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器监听地址"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8777,
        help="服务器监听端口"
    )
    parser.add_argument(
        "--use_bf16", 
        action="store_true",
        default=True,
        help="是否使用bfloat16精度"
    )
    parser.add_argument(
        "--center_crop", 
        action="store_true",
        help="是否使用双重编码"
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    parser = build_argparser()
    args = parser.parse_args()
    
    deploy(args)
