#!/bin/bash -
#SBATCH -J bayes_factor                     # Job Name
#SBATCH -o bayes_factor.stdout              # Output file name
#SBATCH -e bayes_factor.stderr              # Error file name
#SBATCH -t 72:0:00                          # Run time
#SBATCH -A sxs                              # Account name

python bayes_factor_generator.py