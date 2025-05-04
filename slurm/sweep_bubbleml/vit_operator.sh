
NAME='bubblme_vit_sweep'

python src/bubbleml/train.py \
    --multirun \
    experiment=bubbleml/vit_operator \
    experiment.logger_version=${NAME} \
    experiment.train_cfg.lr=0.0001 \
    experiment.model_cfg.embed_dim=256,384 \
    experiment.model_cfg.num_layers=6 \
    experiment.model_cfg.qkv_config.n=256 \
    experiment.model_cfg.ff_config.n=256 \
    experiment.model_cfg.channel_heads=4 \
    experiment.model_cfg.attn_neighborhood_radius=2,4 \
    hydra.launcher.name=${NAME} \
    hydra.launcher.account='amowli_lab_gpu' \
    hydra.launcher.partition='free-gpu' \
    hydra.launcher.nodes=1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.cpus_per_task=20 \
    hydra.launcher.gres='gpu:A30:1' \
    hydra.launcher.timeout_min=1440