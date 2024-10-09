#!/usr/bin/env python3

import sys
sys.path.append("../modules")
sys.path.append("../modules/TransdimensionalSplineFitting")
from loguru import logger
import numpy as np
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from spline_redshift import createSplineRedshift
import pickle
import cloudpickle

import json


from popstock_tsf_helper import (create_injected_OmegaGW,
                                 get_sigma_from_noise_curves,
                                 create_popOmegaGW
import transdimensional_spline_fitting as tsf
from tsf_models import RedshiftSampler
import argparse


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.fontsize'] = 20

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def main(args):
    # variables for generating injected Omega_GW
    freqs = np.arange(10, 200, 0.25)
    Tobs = 86400 * 365.25 * args.Tobs
    Lambda_0 =  {'alpha': 2.5, 'beta': 1, 'delta_m': 3, 'lam': 0.04,
                'mmax': 100, 'mmin': 4, 'mpp': 33, 'sigpp':5, 'gamma': 2.7,
                'kappa': 3, 'z_peak': 1.9, 'rate': 15}
#     num_waveforms = int(1.e4)
#     num_knots_for_redshift_model = 40
#
#     # variables for sampling the redshift distribution that calculates Omega_GW
#     num_mcmc_samples = 10_000

    mass_obj = SinglePeakSmoothedMassDistribution()
    redshift_obj = MadauDickinsonRedshift(z_max=10)

    sigma = get_sigma_from_noise_curves(['CE'], freqs, Tobs)

    # plotting everything
    # plt.loglog(injected_pop.frequency_array, injected_pop.omega_gw, color='blue', linestyle='--', label=r'new $\Omega_{\rm GW}$: $\alpha=4.5$')
    # plt.loglog(spline_pop.frequency_array, spline_pop.omega_gw, color='red', linestyle='-', label='Spline $R(z)$')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel(r'$\Omega_{\rm GW}$')
    # plt.legend()
    # plt.show()

    # plt.loglog(freqs, np.abs(fake_data))
    # plt.loglog(freqs, sigma)
    # plt.show()


    xvals, spline_pop, injected_pop, amplitudes, configuration, Lambda_start = \
        create_injected_OmegaGW(freqs, Lambda_0, args.num_waveforms,
                                mass_obj, redshift_obj)


    # plt.plot(xvals, amplitudes)
    # plt.show()
    fake_data = np.random.randn(freqs.size) * sigma + injected_pop.omega_gw
    data_object = tsf.SmoothCurveDataObj(freqs, fake_data, sigma)

    plt.plot(freqs, np.abs(fake_data))
    plt.plot(freqs, sigma)
    plt.plot(freqs, injected_pop.omega_gw)
    plt.yscale("log")
    plt.show()
    # get sigma for making fake noise
    # make fake_data_with_injection
    start_zvals = np.linspace(0.01, 10, num=args.max_num_knots)
    start_amps = np.interp(start_zvals, xvals, amplitudes)
    params_start = {**{f'amplitudes{ii}': start_amps[ii] for ii in range(start_amps.size)},
                    **{f'xvals{ii}': start_zvals[ii] for ii in range(start_zvals.size)},
                    **{f'configuration{ii}': True for ii in range(start_zvals.size)}}
    Lambda_start = {**params_start, **Lambda_0}
    logger.info(f"Using {args.max_num_knots} knots")
    splredshift_lowknots = createSplineRedshift(args.max_num_knots)(z_max=10)
    pop_sample = create_popOmegaGW(freqs, mass_obj, splredshift_lowknots)

    pop_sample.draw_and_set_proposal_samples(Lambda_start, N_proposal_samples=args.num_waveforms)

    # hardcoded for now...
    # range of redshifts
    zval_range = (0.01, 10)
    # range of p(z) values
    p_z_range = (1e-5, 40)
    rs_sampler = RedshiftSampler(data_object, args.max_num_knots,
                                 zval_range, p_z_range,
                                 min_knots=0, birth_gauss_scalefac=0.1)
    rs_sampler.set_base_population_information(pop_sample, Lambda_start)

    results = rs_sampler.sample(args.num_mcmc_samples,
                                start_knots=start_zvals,
                                start_heights=start_amps,
                                start_config=np.ones(args.max_num_knots).astype(bool),
                                proposal_weights=[1, 1, 0, 1, 1])

    results.max_num_knots = args.max_num_knots
    results.num_waveforms = args.num_waveforms
    results.data_object = data_object
    results.injected_amplitudes = amplitudes
    results.injected_xvals = xvals
    results.injected_omega = injected_pop.omega_gw
    results.frequencies = freqs
    results.lambda_start = Lambda_start
    results.splredshift = splredshift_lowknots
    results.mass_obj = mass_obj

    # save pickled results
    with open("test_results.pkl", "wb") as myf:
        cloudpickle.dump(results, myf)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--Tobs', type=float,
                    help='observation time in years', default=1)
    parser.add_argument('--num-waveforms',
                        help='number of waveforms for resampling', type=int,
                        default=int(1e5))
    parser.add_argument("--num-mcmc-samples", type=int, default=10000,
                        help='number of mcmc samples for rj fitting')
    parser.add_argument("--max-num-knots", type=int, default=40,
                        help='max number of knots for interpolation fitting')


    args = parser.parse_args()
    main(args)
