# saa_admm_paper
This repository contains the code to reproduce the experimental results
from the paper *"Simultaneous activity and attenuation estimation in TOF-PET with TV-constrained nonconvex optimization"*.

## Prerequisuites
- python
- numpy
- numba
- pylab
- scipy

## Usage
- `noiseless-setting.ipynb` contains the code to reproduce 
all the numerical results in the noisess setting, i.e., Figure
1 and Figure 3-8.
- `grid-search.ipynb` contains the code for grid searching the 
best optimization parameters; it reproduces Figure 2.
- `noisy-setting.ipynb` contains the code to reproduce the 
results in the noisy-setting, i.e., Figure 9-11.
