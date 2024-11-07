#!/bin/env python

# -*- coding: utf-8 -*-
# Copyright (C) Arianna I. Renzini 2024
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import sys

sys.path.append("/home/arianna.renzini/PROJECTS/popstock")

import argparse
import json
import os
from pathlib import Path

import bilby
import copy
import numpy as np
import tqdm
from bilby.core.prior import Interped
from bilby.core.utils import infer_args_from_function_except_n_args
from gwpopulation.models.mass import BrokenPowerLawPeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from gwpopulation.utils import xp

from popstock.PopulationOmegaGW import PopulationOmegaGW

"""
***
"""

parser = argparse.ArgumentParser()
parser.add_argument('-ns', '--number_samples',help="number of samples.",action="store", type=int, default=None)
parser.add_argument('-wf', '--waveform_approximant',help="Wavefrom approximant. Default is IMRPhenomD.",action="store", type=str, default='IMRPhenomD')
parser.add_argument('-rd', '--run_directory',help="Run directory.",action="store", type=str, default='./')
parser.add_argument('-sm', '--samples',help="Samples to use.",action="store", type=str, default=None)
parser.add_argument('-fr', '--frequencies',help="txt file with frequencies to use for the spectrum calculation.",action="store", type=str, default=None)
args = parser.parse_args()

N_proposal_samples=args.number_samples
wave_approx=args.waveform_approximant 
tag=f'BPL_{N_proposal_samples}_samples'
rundir=Path(args.run_directory)

"""
***
"""

mass_obj = BrokenPowerLawPeakSmoothedMassDistribution()
redshift_obj = MadauDickinsonRedshift(z_max=10)

models = {
        'mass_model' : mass_obj,
        'redshift_model' : redshift_obj,
        }
if args.frequencies is None:
    freqs = np.arange(10, 2000, 2.5)
else:
    try:
        freqs = np.loadtxt(args.frequencies)
    except ValueError:
        raise ValueError(f"{args.frequencies} is not a txt file.")

newpop = PopulationOmegaGW(models=models, frequency_array=freqs)

if args.samples is not None:
    with open(args.samples) as samples_file:
        samples_dict = json.load(samples_file)
    Lambda_0 = samples_dict['Lambda_0']
    samples_dict.pop('Lambda_0')
    if args.number_samples is None:
        args.number_samples = len(samples_dict['redshift'])
    else:
        if args.number_samples is None:
            args.number_samples = 50000
        for key in samples_dict.keys():
            samples_dict[key] = samples_dict[key][:args.number_samples]
    newpop.set_proposal_samples(proposal_samples = samples_dict)
    print(f'Using {args.number_samples} samples...')

else:
    Lambda_0 =  {'alpha_1': 2, 'alpha_2': 1.4, 'beta':1, 'break_fraction': 0.4, 'mmax': 100, 'mmin': 4, 'lam': 0, 'mpp': 33, 'sigpp':5, 'delta_m':4.5, 'gamma': 2.7, 'kappa': 5.6, 'z_peak': 1.9, 'rate': 16}
    newpop.draw_and_set_proposal_samples(Lambda_0, N_proposal_samples=N_proposal_samples)

newpop.calculate_omega_gw(waveform_approximant=wave_approx, Lambda=Lambda_0)

if args.samples is not None:
    np.savez(f"{os.path.join(rundir, f'omegagw_0_{tag}.npz')}", omega_gw=newpop.omega_gw, freqs=newpop.frequency_array, Lambda_0=Lambda_0)

else:
    np.savez(f"{os.path.join(rundir, f'omegagw_0_{tag}.npz')}", omega_gw=newpop.omega_gw, freqs=newpop.frequency_array, fiducial_samples=newpop.proposal_samples, Lambda_0=Lambda_0, draw_dict=newpop.pdraws)

print('Done!')

exit()
