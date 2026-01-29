export LIBERO_HOME=/jfs/jiang/code/unitree/LIBERO
# export LIBERO_HOME=/path/to/your/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} 
export PYTHONPATH=$(pwd):${PYTHONPATH}


# your_ckpt=/path/to/your/Unifolm-VLA-Libero/checkpoints/pytorch_model.pt
# vlm_pretrained_path=/path/to/your/Unifolm-VLM-Base
your_ckpt=/DATA/disk2/unitree_vla/unitreevla_libero_4_task_window_size_2/checkpoints/pytorch_model.pt
vlm_pretrained_path=/root/Unifolm-VLM-0
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $5}')
step_name=$(echo "$your_ckpt" | awk -F'/' '{print $6}')
task_suite_name=libero_spatial   # libero_goal, libero_object, libero_10, libero_90
num_trials_per_task=50
window_size=2
unnorm_key="libero_spatial_no_noops"  # libero_goal_no_noops, libero_object_no_noops, libero_10_no_noops, libero_90_no_noops

video_out_path="results/${task_suite_name}/${folder_name}/${step_name}"

DEVICE=0

CUDA_VISIBLE_DEVICES=${DEVICE} python ./experiments/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.vlm-pretrained-path ${vlm_pretrained_path} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.unnorm-key "$unnorm_key" \
    --args.window-size "$window_size"