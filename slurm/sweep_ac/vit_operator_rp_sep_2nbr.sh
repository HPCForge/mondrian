python src/quadrature/train.py \
    --multirun \
    experiment=phase_field/vit_operator \
    experiment.train_cfg.lr=0.001 \
    experiment.train_cfg.weight_decay=0.01 \
    experiment.model_cfg.embed_dim=128 \
    experiment.model_cfg.qkv_config.n=256 \
    experiment.model_cfg.ff_config.n=256 \
    experiment.model_cfg.channel_heads=4 \
    experiment.model_cfg.domain_size=[8,8] \
    experiment.model_cfg.attn_neighborhood_radius=2 \
    experiment.logger_version='vit_operator_sweep' \
    hydra.launcher.name='vit_operator_sweep' \
    hydra.launcher.account='amowli_lab_gpu' \
    hydra.launcher.partition='gpu' \
    hydra.launcher.nodes=1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.gres='gpu:A30:1' \
    hydra.launcher.timeout_min=360