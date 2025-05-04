python src/bubbleml/train.py \
    --multirun \
    experiment=bubbleml/ffno \
    experiment.train_cfg.lr=0.0005 \
    experiment.model_cfg.n_modes=96,128 \
    experiment.model_cfg.hidden_channels=32,64,128 \
    experiment.model_cfg.num_layers=5 \
    experiment.logger_version='ffno_sweep' \
    hydra.launcher.name='ffno_sweep' \
    hydra.launcher.account='amowli_lab_gpu' \
    hydra.launcher.partition='free-gpu' \
    hydra.launcher.nodes=1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.cpus_per_task=20 \
    hydra.launcher.gres='gpu:A30:1' \
    hydra.launcher.timeout_min=1440