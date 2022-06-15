SATRAM
======
[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Actions Build Status](https://github.com/noegroup/satram/workflows/CI/badge.svg)](https://github.com/noegroup/satram/actions?query=workflow%3ACI)

A python implementation of MBAR and TRAM and their respective stochastic aproximators SAMBAR and SATRAM.

Installation
------------
1. Clone the repository from github
```bash
git@github.com:noegroup/SATRAM.git
``` 
2. Navigate to the cloned repository and run the installation script
```bash
cd satram     
python setup.py install
```
3. Validate your installation by running all tests in the repository with the command
```bash 
pytest
```

Minimum working example
-----------------------
Use the 4-state test dataset to generate input for TRAM.
```python
from examples.datasets import toy_problem
ttrajs, dtrajs, bias_matrices = toy_problem.get_tram_input()
```

Use the `ThermodynamicEstimator` class to estimate the free energies from the 
dataset. The default `solver_type` is `"SATRAM"`.
```python
from satram import ThermodynamicEstimator

estimator = ThermodynamicEstimator()
estimator.fit((ttrajs, dtrajs, bias), solver_type="SATRAM")
```

The estimated free energies can be accessed as
```python
estimator.free_energies
```

More extensive examples can be found in the jupyter notebooks in the examples 
folder.


Dependencies
------------
* pytorch
* scipy
* deeptime (v0.4.1)


### Copyright

Copyright (c) 2022, NoeGroup


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
