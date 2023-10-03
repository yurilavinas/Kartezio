#!/bin/bash

# LOAD MODULES
module purge
module avail python
module load python/3.8.5

conda activate cgp_AL

# Update pip manager for the venv
python3 -m pip install --upgrade pip --user


cd sources

python3 -m pip install .

pip install -r requirements.txt
