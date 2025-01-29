python src/quadrature/train.py \
    --multirun \
    experiment=phase_field/galerkin_transformer \
    experiment.train_cfg.lr=0.001,0.0005 \
    experiment.model_cfg.embed_dim=128,256,512 \
    experiment.model_cfg.num_heads=4,8 \
    experiment.model_cfg.pos_method='add','concat' \
    experiment.logger_version='gt_sweep' \
    hydra.launcher.name='galerkin_sweep' \
    hydra.launcher.account='amowli_lab_gpu' \
    hydra.launcher.partition='gpu' \
    hydra.launcher.nodes=1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.gres='gpu:A30:1' \
    hydra.launcher.timeout_min=360