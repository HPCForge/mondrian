# Domain Decomposed Neural Operator

## Running Code:

### Environment Setup

```console
conda env create -n mondrian -f conda/pytorch-2.0.1-cuda-11.7.yaml
conda activate mondrian
```

Next, you need to install the utility library, `mondrian_lib`:

```console
python -m pip install --editable .
```

### Training

After the environment has been setup, we can run training on the shear
layer Navier-Stokes problem from [CNO](https://arxiv.org/abs/2302.01178).
Download and unzip `data.zip` from their [Zenodo](https://zenodo.org/records/10406879).

There are several config files for different experiments in `config/experiment/shear_layer/`.
To run the training for the Factorized Fourier Neural Operator (FFNO), just run

```console
python src/train.py \
	experiment=shear_layer/ffno \
	experiment.data_path=</path/to/unzipped/data/>
```
