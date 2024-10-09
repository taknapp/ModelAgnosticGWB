#!/usr/bin/env python3

import cloudpickle
import argparse
import sys
sys.path.append("../modules")
sys.path.append("../modules/TransdimensionalSplineFitting")
from loguru import logger
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from spline_redshift import createSplineRedshift
from popstock_tsf_helper import (create_injected_OmegaGW,
                                 get_sigma_from_noise_curves,
                                 create_popOmegaGW)

from tsf_models import RedshiftSampler



def main(args):
    intermediate_results = cloudpickle.load(open(args.results_pickle, 'rb'))
    mass_obj = SinglePeakSmoothedMassDistribution()
    splredshift_lowknots = createSplineRedshift(intermediate_results.max_num_knots)(z_max=10)

    pop_sample = create_popOmegaGW(intermediate_results.frequencies,
                                   mass_obj,
                                   splredshift_lowknots)

    pop_sample.draw_and_set_proposal_samples(intermediate_results.lambda_start,
                                             N_proposal_samples=intermediate_results.num_waveforms)

    # hardcoded for now...
    # range of redshifts
    zval_range = (0.01, 10)
    # range of p(z) values
    p_z_range = (1e-5, 40)

    print(intermediate_results.max_num_knots)
    rs_sampler = RedshiftSampler(intermediate_results.data_object,
                                 intermediate_results.max_num_knots,
                                 zval_range, p_z_range,
                                 min_knots=0, birth_gauss_scalefac=0.1)
    rs_sampler.set_base_population_information(pop_sample,
                                               intermediate_results.lambda_start)

    results = rs_sampler.sample(args.num_mcmc_samples,
                                start_knots=intermediate_results.knots[-1],
                                start_heights=intermediate_results.heights[-1],
                                start_config=intermediate_results.configurations[-1].astype(bool),
                                proposal_weights=[1, 1, 0, 1, 1])




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-pickle", help="results pickle that we can restart from")
    parser.add_argument("--num-mcmc-samples", help="number of mcmc samples to restart", default=100)
    args = parser.parse_args()
    main(args)
