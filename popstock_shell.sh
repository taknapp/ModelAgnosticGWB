#!/bin/bash -
#SBATCH -J popstock_rjmcmc                  # Job Name
#SBATCH -o popstock_rjmcmc.stdout           # Output file name
#SBATCH -e popstock_rjmcmc.stderr           # Error file name
#SBATCH -t 72:0:00                          # Run time
#SBATCH -A sxs                              # Account name

python popstock_tsf_script.py