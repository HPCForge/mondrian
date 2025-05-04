python src/ns/train.py \
    --multirun \
    experiment=phase_field/ffno \
    experiment.train_cfg.lr=0.001 \
    experiment.train_cfg.weight_decay=0.01 \
    experiment.model_cfg.n_modes=16,32,48 \
    experiment.model_cfg.hidden_channels=64,128,256 \
    experiment.model_cfg.num_layers=4,6,8 \
    experiment.logger_version='ffno_sweep' \
    hydra.launcher.name='ffno_sweep' \
    hydra.launcher.account='amowli_lab_gpu' \
    hydra.launcher.partition='gpu' \
    hydra.launcher.nodes=1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.gres='gpu:A30:1' \
    hydra.launcher.timeout_min=360