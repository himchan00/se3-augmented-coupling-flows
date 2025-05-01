
# VGS Baseline Experiment (Added)

## Training

To train the model, run the following commands:

```shell
python examples/dw4_fab.py
python examples/lj13_fab.py
python examples/lj55_fab.py
```

The model checkpoint with the lowest validation TVD will be saved in `{logger.wandb.name}/model_checkpoints`.


## Sample Generation

To generate 10k samples from the trained model, first set:

```yaml
training.resume = True  # in config/training/default.yaml
```
Although this flag was originally intended for resuming training, here we repurpose it to enable sample generation from a saved checkpoint.

Then run the same scripts:
```shell
python examples/dw4_fab.py
python examples/lj13_fab.py
python examples/lj55_fab.py
```
The generated samples will be saved as `gen_sample.pt` in the current directory.


# SE(3) Equivariant Augmented Coupling Flows
Code for the paper https://arxiv.org/abs/2308.10364. 
Results can be obtained by running the commands in the [Experiments](#experiments) section.

# Install
JAX needs to be installed independently following the instruction on the [JAX homepage](https://github.com/google/jax#installation).
At time of publishing we used JAX 0.4.13 with python 3.10.
This repo also has dependency on pytorch (NB: use CPU version so it doesn't clash with JAX) which may be installed with:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
For the alanine dipeptide problem we need to install openmmtools with conda:
```
conda install -c conda-forge openmm openmmtools
```
Finally then,
```
pip install -e .
```

## Experiments
Experiments may be run with the following commands. 
We use hydra to configure all experiments. 
The flow type may be set as shown in the first line.
For Alanine Dipeptide the data must first be downloaded from [Zenodo](https://zenodo.org/record/6993124/) which may be 
done with the script `eacf/targets/aldp.sh`.
For running an experiment make sure to configure the WANDB logger in the config file to match your WANDB account (e.g. for dw4 see [experiments/config/dw4.yaml](https://github.com/lollcat/se3-augmented-coupling-flows/blob/5f719ec6b1cb5b0d00c5b42e6d755fc76ca9b0b4/examples/config/dw4.yaml#L34) ). 

```shell
python examples/dw4.py flow.type=spherical # Flow types: spherical,proj,along_vector,non_equivariant
python examples/lj13.py
python examples/qm9.py
python examples/aldp.py
python examples/dw4_fab.py
python examples/lj13_fab.py
```

The code for the equivariant CNF baseline can be found in the [ecnf-baseline-neurips-2023](https://github.com/lollcat/ecnf-baseline-neurips-2023) repo. 

## Upcoming additions
- Quickstart notebook with inference using model weights

## Citation

If you use this code in your research, please cite it as:

```
@inproceedings{
midgley2023eacf,
title={{SE}(3) Equivariant Augmented Coupling Flows},
author={Laurence Illing Midgley and Vincent Stimper and Javier Antoran and Emile Mathieu and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=KKxO6wwx8p}
}
```
