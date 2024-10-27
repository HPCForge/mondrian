# Domain Decomposed Neural Operator

### Environment Setup

```console
conda env create -n mondrian -f conda/pytorch-2.0.1-cuda-11.7.yaml
conda activate mondrian
```

Next, you need to install the utility library, `mondrian_lib`:

```console
python -m pip install --editable .
```

To run the `tests/`, execute

```console
python -m pytest
```