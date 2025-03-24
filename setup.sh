#!/bin/bash

which python
python -m venv my_env
source my_env/bin/activate
pip install numpy scipy matplotlib numba
