# UnifoLM-VLA-0: A Vision-Language-Action (VLA) Framework under UnifoLM Family
 <p style="font-size: 1.2em;">
    <a href="https://unigen-x.github.io/unifolm-world-model-action.github.io"><strong>项目主页</strong></a> | 
    <a href="https://huggingface.co/collections/unitreerobotics/unifolm-wma-0-68ca23027310c0ca0f34959c"><strong>开源模型</strong></a> |
    <a href="https://huggingface.co/unitreerobotics/datasets"><strong>开源数据</strong></a> 
  </p>
<div align="center">
  <p align="right">
    <span> 🌎English </span> | <a href="README_cn.md"> 🇨🇳中文 </a>
  </p>
</div>

**UnifoLM-VLA-0** 是 UnifoLM 系列下面向通用人形机器人操作的视觉-语言-动作（VLA）大模型。该模型旨在突破传统 VLM 在物理交互中的局限，通过在机器人操作数据上的继续预训练，实现了从通用"图文理解"向具备物理常识的"具身大脑"的进化。

<table width="100%">
  <tr>
    <th width="50%">空间语义增强</th>
    <th width="50%">通用操作泛化</th>
  </tr>
  <tr>
    <td valign="top">
      针对操作类任务中对指令理解与空间感知的高要求，模型通过继续预训练深度融合了文本指令与2D/3D空间细节, <strong>增强了模型的空间感知能力</strong>。
    </td>
    <td valign="top">
      构建了全链路动力学预测数据，模型具备更好的任务泛化性。在真机验证中, <strong>仅需单一策略即可高质量完成 12 类复杂的操作任务</strong>。
    </td>
  </tr>
</table>

<div align="center">
  <img 
    src="assets/gif/UnifoLM-VLA-0.gif"
    style="width:100%; max-width:1000px; height:auto;"
  />
</div>



## 🔥 新闻
* 2026年1月29日: 🚀 我们发布了 **UnifoLM-VLA-0** 的训练与推理代码，以及对应的模型权重。


## 📑 开源计划
- [x] 训练代码 
- [x] 推理代码 
- [x] 模型 Checkpoints

## ⚙️  安装
本项目基于**CUDA 12.4**构建,建议使用同样的版本
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
## 🧰 模型 Checkpoints

| 模型 | 描述 | 链接 |
|---|---|---|
| `UnifoLM-VLM-Base` | 在通用图文VQA数据、[Open-X](https://robotics-transformer-x.github.io/) 数据集、[星海图](https://opengalaxea.github.io/GalaxeaVLA/)数据集上微调后的模型 | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLM-Base) |
| `UnifoLM-VLA-Base` | 在 [宇树科技开源数据集](https://huggingface.co/collections/unitreerobotics/unifolm-vla-0)上微调后的模型 | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Base) |
| `UnifoLM-VLA-LIBERO` | 在 [LIBERO](https://libero-project.github.io/) 数据集上微调后的模型 | [HuggingFace](https://huggingface.co/unitreerobotics/Unifolm-VLA-Libero) |


## 🛢️ 数据集
在实验中，我们使用并评估了以下十二个开源数据集：
| 数据集 | 机器人 | 链接 |
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

要在自定义数据集上训练，请首先确保数据符合 [Huggingface LeRobot V2.1](https://github.com/huggingface/lerobot)数据集格式，假设下载后的数据目录结构如下：
```
source_dir/
    ├── dataset1_name
    ├── dataset2_name
    ├── dataset3_name
    └── ...
```
随后执行以下命令将lerobot格式的数据集转化为hdf5格式的数据集:
```python
cd prepare_data
python convert_lerobot_to_hdf5.py \
    --data_path /path/to/your/source_dir/dataset1_name \
    --target_path /path/to/save/the/converted/data/directory
```
最后执行以下命令将hdf5格式转化为训练需要的RLDS数据集格式，记得修改hdf5数据的地址([here](prepare_data/lerobot_to_rlds/lerobot_dataset/lerobot_dataset_dataset_builder.py#L232)),  `data_dir`表示rlds数据集存放地址
```
cd prepare_data/lerobot_to_rlds/lerobot_dataset
tfds build  --data_dir  /path/to/save/the/converted/data/directory
```
转完的RLDS数据目录结构如下
```
source_dir/
├── downloads
├── rlds_dataset
│       └── 1.0.0
```
其中，`1.0.0` 目录即为最终可用于训练的RLDS数据集版本，最后的目录保留为 `source_dir/1.0.0`（例如：`g1_stack_block/1.0.0`）。

## 🚴 模型训练
 在单个或多个数据集上进行训练，请按照以下步骤操作：
- **步骤1**：假设你已经准备好RLDS数据集，通过添加该数据集的条目（例如宇树开源数据集G1_Stack_Block）注册到我们的数据加载器，添加 `configs.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py#L58))、`transforms.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/transforms.py#L948)) 和 `mixtures.py` ([here](src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/mixtures.py#L366))、`datasets.py`([here](src/unifolm_vla/rlds_dataloader/datasets/datasets.py#L106)) 的条目。
- **步骤2**：开始微调前，需要设置模型预测的动作块的大小、数据集中动作和状态的自由度大小、数据归一化的方式  `constants.py` ([here](src/unifolm_vla/rlds_dataloader/constants.py#L70)) (参考G1_CONSTANTS 中的 `NUM_ACTIONS_CHUNK`、 `ACTION_DIM`、 `PROPRIO_DIM`、 `ACTION_PROPRIO_NORMALIZATION_TYPE`)。
- **步骤3**：请按以下顺序完成配置(参考 [here](scripts/run_scripts/run_unifolm_vla_train.sh)):
  1. **模型初始化**：将 `base_vlm` 修改为 **UnifoLM-VLM-Base** 的本地路径或对应的模型权重地址，用于初始化视觉-语言主干模型；
  2. **数据路径设置**：在完成模型路径配置后，将 `oxe_data_root` 设置为数据集所在的根目录，确保训练脚本能够正确加载 RLDS 数据；
  3. **数据组合指定**：基于已配置的数据根目录，将 `data_mix` 配置为需要参与训练的数据集名称或其组合方式；
  4. **模型权重保存**：设置模型Checkpoint与日志的保存路径，用于存储微调过程中生成的模型权重与训练状态，便于后续模型恢复、评估与推理使用；
  5. **并行规模调整**：最后，根据实际可用的 GPU 数量，将 `num_processes` 调整为对应的值，以匹配当前的分布式训练规模。
- **步骤4**：现在可以开始进行微调了，运行脚本 [`run_unifolm_vla_train.sh`](scripts/run_scripts/run_unifolm_vla_train.sh)。
## 🌏 仿真推理测试
要在 `LIBERO`([here](https://huggingface.co/datasets/openvla/modified_libero_rlds)) 仿真环境中测试 **UnifoLM-VLA-Libero** 模型，请按以下步骤操作：

- **步骤 1**：在进行仿真测试前，需要先安装 LIBERO 仿真环境及其相关依赖。请依次执行以下命令：
  ```bash
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
  pip install -e LIBERO
  pip install -r experiments/LIBERO/libero_requirements.txt  # 从 UnifoLM-VLA 项目根目录执行
  ```
- **步骤2**： 在 `run_eval_libero.sh` ([here](scripts/eval_scripts/run_eval_libero.sh)) 中修改指定 ```your_ckpt```、```task_suite_name```、```unnorm_key```、 ```LIBERO_HOME```、 ```vlm_pretrained_path```;
- **步骤3**： 启动服务器：
```
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_eval_libero.sh
```
  
## 🤖 真机推理测试
在我们的系统中，推理在服务器端执行；机器人客户端从真实机器人收集观测信息并发送至服务器，进行动作推理。可通过如下步骤实现整个过程：

### 服务器端设置
- **步骤1**： 在 `run_real_eval_server.sh` ([here](scripts/eval_scripts/run_real_eval_server.sh)) 中修改指定 ```ckpt_path```、```port```、```unnorm_key```、 ```vlm_pretrained_path```;
- **步骤2**： 启动服务器：
```
conda activate unifolm-vla
cd unifolm-vla
bash scripts/eval_scripts/run_real_eval_server.sh
```

### 客户端设置
- **步骤1**： 参考 [unitree_deploy/README.md](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/unitree_deploy/README.md)，创建 ```unitree_deploy``` conda 环境，安装所需依赖包，并在真实机器人端启动控制器或服务;
- **步骤2**: 打开一个新的终端，从客户端到服务器建立隧道连接：  
```
ssh user_name@remote_server_IP -CNg -L port:127.0.0.1:port
```
- **步骤3**： 参考 ```unitree_deploy/robot_client.py``` 脚本进行修改并运行。


## 📝 代码架构
以下是本项目代码结构设计及核心组件说明：
```
unifolm-vla/
    ├── assets                      # GIF动图、静态图片和演示视频等媒体素材
    ├── experiments                 # 仿真测试
    ├── deployment                  # 示例数据
    ├── prepare_data                # 数据处理
    ├── scripts                     # 主程序脚本
    ├── src
    │    ├──unifolm_vla             # 核心库
    │    │      ├── config          # 参数配置
    │    │      ├── model           # 模型架构
    │    │      ├── rlds_dataloader # 数据加载
    |    │      └── training        # 模型训练
```

## 🙏 致谢声明
本项目代码基于以下优秀开源项目构建，特此致谢：[Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),  [Open-X](https://robotics-transformer-x.github.io/), [Galaxea Open-World Dataset](https://opengalaxea.github.io/GalaxeaVLA/), [openvla-oft](https://github.com/moojink/openvla-oft), [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1)。
