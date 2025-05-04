
NAME='ns_sweep'

python src/ns/train.py \
    --multirun \
    experiment=ns/vit_operator \
    experiment.logger_version=${NAME} \
    experiment.model_cfg.embed_dim=128 \
    experiment.model_cfg.ff_config.n=256 \
    experiment.model_cfg.qkv_config.n=256 \
    hydra.launcher.name=${NAME} \
    hydra.launcher.account='amowli_lab_gpu' \
    hydra.launcher.partition='gpu' \
    hydra.launcher.nodes=1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.cpus_per_task=16 \
    hydra.launcher.gres='gpu:A30:1' \
    hydra.launcher.timeout_min=360