# Mondrian

## Experiment Setup

Create a conda/mamba environment with the needed dependencies.

```console
conda env create -n mondrian -f conda/pytorch-2.5.1-cuda-11.7.yaml
conda activate mondrian
```

Next, you need to install the utility library, `mondrian`. At the moment,
this is only really a library so it's easy to reuse modules in different experiments. Some parts of it could be reused in other projects, but a some parts depends on the config files.

```console
python -m pip install --editable .
```

To run the unit tests, execute

```console
python -m pytest tests/
```

## Running Experiments

The experiments use the hydra config manage (this should be installed when setting up the conda environment). You need to point the config file to the where you downloaded the data on your machine. This can be done by directly editing the config file, or by overriding it on the command line:

```console
python src/quadrature/train.py \
    experiment=phase_field/vit_operator \
    experiment.data_path='/path/to/your/data.hdf5'
```

What data paths need to be overriden depends on the experiment. I.e, the allen-cahn
test uses four separate hdf5 files. One for training / val, and three more with data 
at different resolutions.

## Citation

If you find this repo useful for your work, consider citing our arXiv preprint:

```bibtex
@misc{feeney2025mondriantransformeroperatorsdomain,
      title={Mondrian: Transformer Operators via Domain Decomposition}, 
      author={Arthur Feeney and Kuei-Hsiang Huang and Aparna Chandramowlishwaran},
      year={2025},
      eprint={2506.08226},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.08226}, 
}
```