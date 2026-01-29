run_id: qwenact
run_root_dir: results/Checkpoints
seed: 42
trackers: [jsonl, wandb]
wandb_entity: your_wandb_entity
wandb_project: your_wandb_project
is_debug: false

framework:
  name: unifolm_vla
  qwenvl:
    base_vlm: /root/unitree_jiang/Unifolm-VLM-0
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
    model_type: qwen2_5_vl

  action_model:
    input_embedding_dim: 1536
    hidden_size: 1024     
    add_pos_embed: True
    max_seq_len: 1024
    action_dim: 16
    state_dim: 16
    future_action_window_size: 15
    action_horizon: 16
    past_action_window_size: 0
    repeated_diffusion_steps: 8
    noise_beta_alpha: 1.5
    noise_beta_beta: 1.0
    noise_s: 0.999
    num_timestep_buckets: 1000
    num_inference_timesteps: 4
    num_target_vision_tokens: 32
    diffusion_model_cfg:   
      cross_attention_dim: 2048 
      attention_head_dim: 48
      num_attention_heads: 32
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: "ada_norm"
      num_layers: 16
      output_dim: 1024
      positional_embeddings: null

datasets:
  vla_data:
    data_root_dir: /path/to/your/data
    data_mix: your_data_mix  
    per_device_batch_size: 16
    load_all_data_for_training: true
    window_size: 1
    image_size: [224,224]

trainer:
  epochs: 100
  max_train_steps: 100000
  num_warmup_steps: 5000
  save_interval: 5000
  eval_interval: 100
  learning_rate:
    base: 1e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 5.0e-07
  freeze_modules: ''
  loss_scale:
    vla: 1.0
  repeated_diffusion_steps: 4
  max_grad_norm: 1.0
  warmup_ratio: 0.1
  weight_decay: 0.0
  logging_frequency: 10
  gradient_clipping: 1.0
  gradient_accumulation_steps: 1

  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08

  # parameters to be determined
  is_resume: false
  resume_epoch: null
  resume_step: null
  enable_gradient_checkpointing: true
  enable_mixed_precision_training: true
