# UnifoLM-VLA-0: A Vision-Language-Action (VLA) Framework under UnifoLM Family
 <p style="font-size: 1.2em;">
    <a href="https://unigen-x.github.io/unifolm-world-model-action.github.io"><strong>Project Page</strong></a> | 
    <a href="https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c"><strong>Models</strong></a> |
    <a href="https://huggingface.co/unitreerobotics/datasets"><strong>Datasets</strong></a> 
  </p>
<div align="center">
  <p align="right">
    <span> 🌎English </span> | <a href="README_cn.md"> 🇨🇳中文 </a>
  </p>
</div>

**UnifoLM-VLA-0** is a Vision–Language–Action (VLA) large model in the UnifoLM series, designed for general-purpose humanoid robot manipulation. It goes beyond the limitations of conventional Vision–Language Models (VLMs) in physical interaction. Through continued pre-training on robot manipulation data, the model evolves from "vision-language understanding" to an "embodied brain" equipped with physical common sense.
<div style="display:flex; gap:24px; margin-top:24px;"> <div style="flex:1; padding:20px; border:1px solid #e5e7eb; border-radius:12px;"> <h3 style="margin-top:0;">
Spatial Semantic Enhancement</h3> <p> To address the requirements for instruction comprehension and spatial understanding in manipulation tasks, the model deeply integrates textual instructions with 2D/3D spatial details through continued pre-training, <strong>substantially strengthening its spatial perception and geometric understanding capabilities..</strong></p> </div> 
<div style="flex:1; padding:20px; border:1px solid #e5e7eb; border-radius:12px;"> <h3 style="margin-top:0;">
Manipulation Generalization</h3> <p> By leveraging full dynamics prediction data, the model achieves strong generalization across diverse manipulation tasks. In real-robot validation, <strong>it can complete 12 categories of complex manipulation tasks with high quality using only a single policy.</strong> </p> </div> </div>


<div align="center">
  <img 
    src="assets/gif/UnifoLM-VLA-0.gif"
    style="width:100%; max-width:1000px; height:auto;"
  />
</div>

## 🔥 News
* Jan 29, 2026: 🚀 We released the training and inference code, along with the model weights for [**UnifoLM-VLA-0**](https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c).

## 📑 Open-Source Plan
- [x] Training 
- [x] Inference
- [x] Checkpoints

## ⚙️  Installation
This project is built on **CUDA 12.4**, and using the same version is strongly recommended to ensure compatibility.
```
conda create -n unifolm-vla python==3.10.18
conda activate unifolm-vla

git clone https://github.com/unitreerobotics/unifolm-vla.git

# If you already downloaded the repo:
cd unifolm-vla
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@0878c68"
pip install -e .

# Install FlashAttention2
pip install "flash-attn==2.5.6" --no-build-isolation

```
## 🧰 Model Checkpoints
| Model | Description | Link|
|---------|-------|------|
|`UnifoLM-VLM-Base` | Fine-tuned on general-purpose image–text VQA data、 [Open-X](https://robotics-transformer-x.github.io/) dataset、 [Galaxea Open-World Dataset](https://opengalaxea.github.io/GalaxeaVLA/). | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLM-Base)|
|`UnifoLM-VLA-Base` | Fine-tuned on [Unitree opensource](https://huggingface.co/collections/unitreerobotics/g1-dex1-datasets-68bae98bf0a26d617f9983ab) dataset. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Base)|
|`UnifoLM-VLA-LIBERO`| Fine-tuned on [Libero](https://huggingface.co/collections/unitreerobotics/g1-dex1-datasets-68bae98bf0a26d617f9983ab) dataset. | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Libero)|

## 🛢️ Dataset
In our experiments, we consider the following twelve open-source dataset:
| Dataset | Robot | Link |
|---------|-------|------|
|G1_Stack_Block| [Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Stack_Block)|
|G1_Bag_Insert|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Bag_Insert)|
|G1_Erase_Board|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Erase_Board)|
|G1_Clean_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Clean_Table)|
|G1_Pack_PencilBox|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PencilBox)|
|G1_Pour_Medicine|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pour_Medicine)|
|G1_Pack_PingPong|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Pack_PingPong)|
|G1_Prepare_Fruit|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Prepare_Fruit)|
|G1_Organize_Tools|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Organize_Tools)|
|G1_Fold_Towel|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Fold_Towel)|
|G1_Wipe_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_Wipe_Table)|
|G1_DualRobot_Clean_Table|[Unitree G1](https://www.unitree.com/g1)|[Huggingface](https://huggingface.co/datasets/unitreerobotics/G1_DualRobot_Clean_Table)|

To train on your own dataset, ensure the data follows the [Huggingface LeRobot V2.1](https://github.com/huggingface/lerobot) dataset format. Assume the source directory structure of the dataset is as follows:
```
source_dir/
    ├── dataset1_name
    ├── dataset2_name
    ├── dataset3_name
    └── ...
```
Then, run the following command to convert the dataset from LeRobot format to HDF5 format:
```python
cd prepare_data
python convert_lerobot_to_hdf5.py \
    --data_path /path/to/your/source_dir/dataset1_name \
    --target_path /path/to/save/the/converted/data/directory
```
Finally, run the following command to convert the HDF5 format into the RLDS dataset format required for training. Be sure to update the path ([here](prepare_data/lerobot_to_rlds/lerobot_dataset/lerobot_dataset_dataset_builder.py#L232)) to the correct location of the HDF5 data.
```
cd prepare_data/lerobot_to_rlds/lerobot_dataset
tfds build  --data_dir  /path/to/save/the/converted/data/directory
```
The directory structure of the converted RLDS dataset is as follows:
```
source_dir/
├── downloads
├── rlds_dataset
         └── 1.0.0
```
The `1.0.0` directory is the final RLDS dataset version that can be used for training. The final directory should be kept as `source_dir/1.0.0`(e.g., `g1_stack_block/1.0.0`).

## 🚴‍♂️ Training
To train on a single dataset or multiple datasets, follow the steps below:
- **Step 1**: Assuming you have already prepared the RLDS dataset, register the dataset (e.g., the Unitree open-source dataset `G1_StackBox`) with our dataloader by adding an entry for it in `configs.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py#L58)), `transforms.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/transforms.py#L948)), and `mixtures.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py#L366)) and `datasets.py`([here](src/unifolm_vla/rlds_dataloader/datasets/datasets.py#L106)). For reference, in each of these files, there are sample entries for the G1 datasets that we used in experiments.
- **Step 2**: Before starting fine-tuning, configure the size of the action chunks predicted by the model, the action and state degrees of freedom in the dataset, and the data normalization scheme in `constants.py` ([here](src/unifolm_vla/rlds_dataloader/constants.py#L70)). Refer to `NUM_ACTIONS_CHUNK`, `ACTION_DIM`, `PROPRIO_DIM`, and `ACTION_PROPRIO_NORMALIZATION_TYPE` in `G1_CONSTANTS`.
- **Step 3**:  please complete the configuration in the following order (see [here](scripts/run_scripts/run_unifolm_vla_train.sh)):
1. **Model Initialization**: Set `base_vlm` to the local path or the corresponding model weight URL of **UnifoLM-VLM-Base**, which will be used to initialize the vision–language backbone model.
2. **Dataset Path Configuration**: After configuring the model path, set `oxe_data_root` to the root directory of the dataset to ensure that the training script can correctly load the RLDS data.
3. **Dataset Mixture Specification**: Based on the configured data root, set `data_mix` to the name of the dataset(s) to be used for training or to the desired dataset mixture.
4. **Model Checkpoint Saving**: Specify the paths for saving model checkpoints and logs, which will store the model weights and training states generated during fine-tuning for later recovery, evaluation, and inference.
5. **Parallelism Configuration**: Finally, adjust `num_processes` according to the number of available GPUs to match the scale of distributed training.
- **Step 4**: You can now start fine-tuning. Before running the script [`run_unifolm_vla_train.sh`](scripts/run_scripts/run_unifolm_vla_train.sh),

## 🌏 Simulation Inference Evaluation
To evaluate the **UnifoLM-VLA-Libero** model in the `LIBERO` simulation environment ([here](https://huggingface.co/datasets/openvla/modified_libero_rlds)), follow the steps below:
- **Step 1**: Install the LIBERO simulation environment and its dependencies::
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/LIBERO/libero_requirements.txt  # Run from the UnifoLM-VLA project root directory
```
- **Step 2**: In `run_eval_libero.sh` ([here](scripts/eval_scripts/run_eval_libero.sh)), modify the following fields:  ```your_ckpt```, ```task_suite_name```, ```unnorm_key```, and ```LIBERO_HOME``` and  ```vlm_pretrained_path```.
- **Step 3**: Launch the evaluation:
```
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_eval_libero.sh
```

## 🤖 Real-World Inference Evaluation

In our system, inference is executed on the server side. The robot client collects observations from the real robot and sends them to the server for action inference. The full pipeline can be completed by following the steps below.

### Server Setup
- **Step 1**: In `run_real_eval_server.sh` ([here](scripts/eval_scripts/run_real_eval_server.sh)), modify the following fields:  ```ckpt_path```, ```port```, and ```unnorm_key``` and  ```vlm_pretrained_path```.
- **Step 2**: Launch the server:
```
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_real_eval_server.sh
```

### Client Setup

- **Step 1**: Refer to [unitree_deploy/README.md](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/unitree_deploy/README.md) to create the ```unitree_deploy``` conda environment, install the required dependencies, and start the controller or service on the real robot.

- **Step 2**: Open a new terminal and establish a tunnel connection from the client to the server:
```
ssh user_name@remote_server_IP -CNg -L port:127.0.0.1:port
```
- **Step 3**: Modify and run the script ```unitree_deploy/robot_client.py``` as a reference.

## 📝 Codebase Architecture
Here's a high-level overview of the project's code structure and core components:
```
unifolm-vla/
    ├── assets                      # Media assets such as GIFs
    ├── experiments                 # Libero datasets for running inference
    ├── deployment                  # Deployment server code
    ├── prepare_data                # Scripts for dataset preprocessing and format conversion
    ├── scripts                     # Main scripts for training, evaluation, and deployment
    ├── src
    │    ├──unifolm_vla             # Core Python package for the Unitree world model
    │    │      ├── config          # Configuration files for training
    │    │      ├── model           # Model architectures and backbone definitions
    │    │      ├── rlds_dataloader # Dataset loading, transformations, and dataloaders
    │    │      └── training        # Model Training
```

## 🙏 Acknowledgement
Lots of code are inherited from [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),  [Open-X](https://robotics-transformer-x.github.io/), [Galaxea Open-World Dataset](https://opengalaxea.github.io/GalaxeaVLA/), [openvla-oft](https://github.com/moojink/openvla-oft), [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1).

## 📝 Citation
```
@misc{unifolm-vla-0,
  author       = {Unitree},
  title        = {UnifoLM-VLA-0: A Vision-Language-Action (VLA) Framework under UnifoLM Family},
  year         = {2026},
}
```




