## SIREN in Pytorch

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
    final_activation = nn.Sigmoid()    # activation of final layer
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

spatial_coor = torch.randn(1, 3)
neuron(spatial_coor) # (1, 256)
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
