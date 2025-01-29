python src/quadrature/train.py \
    --multirun \
    experiment=phase_field/fno \
    experiment.train_cfg.lr=0.001,0.0005 \
    experiment.train_cfg.weight_decay=0.01,0.0001 \
    experiment.model_cfg.n_modes=[16,16],[24,24] \
    experiment.model_cfg.hidden_channels=16,32,64 \
    experiment.model_cfg.num_layers=4 \
    experiment.logger_version='fno_sweep' \
    hydra.launcher.name='fno_sweep' \
    hydra.launcher.account='amowli_lab_gpu' \
    hydra.launcher.partition='gpu' \
    hydra.launcher.nodes=1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.gres='gpu:A30:1' \
    hydra.launcher.timeout_min=360