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

Wrapper to train on a specific image of specified height and width from a given `SirenNet`, and then to subsequently generate.


```python
import torch
from torch import nn
from siren_pytorch import SirenNet, SirenWrapper

net = SirenNet(
    dim_in = 2,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 3,                       # output dimension, ex. rgb value
    num_layers = 5,                    # number of layers
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

wrapper = SirenWrapper(
    net,
    image_width = 256,
    image_height = 256
)

img = torch.randn(1, 3, 256, 256)
loss = wrapper(img)
loss.backward()

# after much training ...
# simply invoke the wrapper without passing in anything

pred_img = wrapper() # (1, 3, 256, 256)
```

## Modulation with Latent Code

A <a href="https://arxiv.org/abs/2104.03960">new paper</a> proposes that the best way to condition a Siren with a latent code is to pass the latent vector through a modulator feedforward network, where each layer's hidden state is elementwise multiplied with the corresponding layer of the Siren.

You can use this simply by setting an extra keyword `latent_dim`, on the `SirenWrapper`

```python
import torch
from torch import nn
from siren_pytorch import SirenNet, SirenWrapper

net = SirenNet(
    dim_in = 2,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 3,                       # output dimension, ex. rgb value
    num_layers = 5,                    # number of layers
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

wrapper = SirenWrapper(
    net,
    latent_dim = 512,
    image_width = 256,
    image_height = 256
)

latent = nn.Parameter(torch.zeros(512).normal_(0, 1e-2))
img = torch.randn(1, 3, 256, 256)

loss = wrapper(img, latent = latent)
loss.backward()

# after much training ...
# simply invoke the wrapper without passing in anything

pred_img = wrapper(latent = latent) # (1, 3, 256, 256)
```

## Citations

```bibtex
@misc{sitzmann2020implicit,
    title   = {Implicit Neural Representations with Periodic Activation Functions},
    author  = {Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
    year    = {2020},
    eprint  = {2006.09661},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{mehta2021modulated,
    title   = {Modulated Periodic Activations for Generalizable Local Functional Representations}, 
    author  = {Ishit Mehta and Michaël Gharbi and Connelly Barnes and Eli Shechtman and Ravi Ramamoorthi and Manmohan Chandraker},
    year    = {2021},
    eprint  = {2104.03960},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```