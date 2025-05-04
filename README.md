# Mondrian

## Environment Setup

Create a conda environment with the dependencies.

```console
conda env create -n mondrian -f conda/pytorch-2.5.1-cuda-11.7.yaml
conda activate mondrian
```

Next, you need to install the utility library, `mondrian`. At the moment,
this is only really a library so it's easy to reuse across our experiments.
Some parts of it could be reused in other projects, but others cannot be.

```console
python -m pip install --editable .
```

To run the unit tests, execute

```console
python -m pytest tests/
```

## Running Experiments

The experiments use the hydra config manager. You need to point the config file
to the correct data location on your machine. This can be done by directly editing
the config file, or by overriding it on the command line:

```console
python src/quadrature/train.py \
    experiment=phase_field/vit_operator \
    experiment.data_path='/path/to/your/data.hdf5'
```

What data paths need to be overriden depends on the experiment. I.e, the allen-cahn
test uses four separate hdf5 files. One for training / val, and three more with data 
at different resolutions. While bubbleml uses separate train, val, and test hdf5 files.