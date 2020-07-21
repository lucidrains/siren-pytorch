## SIREN in Pytorch

[![PyPI version](https://badge.fury.io/py/siren-pytorch.svg)](https://badge.fury.io/py/siren-pytorch)

Pytorch implementation of SIREN -  <a href="https://arxiv.org/abs/2006.09661">Implicit Neural Representations with Periodic Activation Function</a>

## Install

```bash
$ pip install siren-pytorch
```

## Usage

A SIREN based multi-layered neural network

```python
import torch
from torch import nn
from siren_pytorch import SirenNet

net = SirenNet(
    dim_in = 2,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 3,                       # output dimension, ex. rgb value
    num_layers = 5,                    # number of layers
    final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

coor = torch.randn(1, 2)
net(coor) # (1, 3) <- rgb value
```

One SIREN layer

```python
import torch
from siren_pytorch import Siren

neuron = Siren(
    dim_in = 3,
    dim_out = 256
)

coor = torch.randn(1, 3)
neuron(coor) # (1, 256)
```

Sine activation (just a wrapper around `torch.sin`)

```python
import torch
from siren_pytorch import Sine

act = Sine(1.)
coor = torch.randn(1, 2)
act(coor)
```

## Citations

```bibtex
@misc{sitzmann2020implicit,
    title={Implicit Neural Representations with Periodic Activation Functions},
    author={Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
    year={2020},
    eprint={2006.09661},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
