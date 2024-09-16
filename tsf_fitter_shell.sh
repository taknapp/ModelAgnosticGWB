#!/bin/bash -
#SBATCH -J tsf_fitter                       # Job Name
#SBATCH -o tsf_fitter.stdout                # Output file name
#SBATCH -e tsf_fitter.stderr                # Error file name
#SBATCH -t 72:0:00                          # Run time
#SBATCH -A sxs                              # Account name

python tsf_fitter.py