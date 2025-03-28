Overlap analysis module
===============================

Library install
---------------

After downloading the repository, it can be installed as a Python library with:

```
  pip install -r requirements.txt
  pip install -e .
```
The dependency on QuantumMAMBO needs some extra care since the juliacall package will by default install the latest version of Julia in your environment, which will not be compatible with QuantumMAMBO. A temporary resolution is as follows (assuming the virtual environment is located at /.venv):

(1) Locate the .venv/julia_env/pyjuliapkg folder

(2) Copy the juliapkg.json file to that folder

(3) Run
```
python setup-julia.py
```

Usage
-----
Calculating wavefunction overlaps with Hamiltonian eigenstates from Hamiltonian power expectation values.
arXiv: https://arxiv.org/abs/2503.12224
