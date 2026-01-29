python deployment/model_server/run_real_eval_server.py \
    --ckpt_path /path/to/your/Unifolm-VLA-Base/checkpoints/pytorch_model.pt \
    --port 8777 \
    --unnorm_key g1_stack_block \
    --vlm_pretrained_path /path/to/your/Unifolm-VLM-Base