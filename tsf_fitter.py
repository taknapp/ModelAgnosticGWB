import sys
import transdimensional_spline_fitting as tsf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
from scipy.optimize import minimize
from functools import partial
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d

mpl.rcParams.update(mpl.rcParamsDefault)

import pygwb
import bilby
import astropy.cosmology
from copy import deepcopy
from pygwb.baseline import Baseline
import seaborn as sns

from popstock_tsf_helper import *

# Signficantly speeds things up
import lal
lal.swig_redirect_standard_output_error(False)

R0 = 31.4
H0 = astropy.cosmology.Planck18.H0.to(astropy.units.s**-1).value

class SmoothCurveDataObj(object):
    """
    A data class that can be used with our spline model
    """
    def __init__(self, data_xvals, data_yvals, data_errors):
        self.data_xvals = data_xvals
        self.data_yvals = data_yvals
        self.data_errors = data_errors

class FitRedshift(tsf.BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """
    def ln_likelihood(self, config, heights):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!
        redshift_model = 10**self.evaluate_interp_model(np.log10(bbh_pickle.ref_zs), heights, config, log_xvals=True)
        
        model = bbh_pickle.eval(R0, redshift_model, self.data.data_xvals)
        
        return np.sum(norm.logpdf(model - self.data.data_yvals, scale=self.data.data_errors))

class FitOmega(tsf.BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """

    
    def ln_likelihood(self, config, heights, knots):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!
        omega_model = 10**self.evaluate_interp_model(np.log10(self.data.data_xvals), heights, config, np.log10(knots))
        # print(omega_model)
        # print(self.data.data_yvals)
        # print(self.data.data_errors)

        return np.sum(norm.logpdf(omega_model - self.data.data_yvals, scale=self.data.data_errors))

# VARIABLES   
freqs = np.arange(20, 100, 0.03125)

T = 365.25 * 24 * 60 * 60 * 10

N_samples = 100_000
N_offset = int(0.5 * N_samples)

# generate data given a keyword ['BPL', 'squiggly', 'Sachdev'] - currently using a txt file for the Sachdev curve
sig_type = 'BPL'

# GENERATE DATA
# get the sigma curves based on detector pairs
HL_sigma = get_sigma_from_noise_curves(['H1', 'L1'], freqs, T)
CE_sigma = get_sigma_from_noise_curves(['CE'], freqs, T)

# defaulting to using HL A+ noise curve, can also use CE just as easily
signal, data, data_obj = generate_data(sig_type, freqs, HL_sigma)

# SAMPLE RJMCMC
fit_omega, fit_results_omega = sample_Omega(freqs, N_samples, data_obj)

# PLOT RECOVERED OMEGA_GW
plot_posterior_fits(fit_omega, fit_results_omega, freqs, N_samples, offset=N_offset, num_posteriors=1000)

plt.plot(freqs, data, alpha=0.3, label='Data', zorder=1000)
plt.plot(freqs, signal, c='r', ls='--', label='True signal', zorder=1001)
plt.yscale("log")
plt.xscale("log")
plt.ylim(1e-12, 1e-5)
plt.xlabel('freqs [Hz]')
plt.ylabel('$\Omega_{GW}$')
plt.show()
plt.savefig('tsf_posterior_fits.pdf')

plt.clf()

# BAYES FACTOR CALCULATIONS
configs, num_knots = return_knot_info(fit_results_omega, offset=0)

xs = return_knot_frequencies(fit_results_omega, toggle=False)
ys = return_knot_heights(fit_results_omega, toggle=False)

two_slopes = []
knees = []
three_slopes1 = []
three_slopes2 = []

for ii, num_kn in enumerate(num_knots):
    if num_kn == 2: 
        bool_config = [bool(x) for x in configs[ii]]

        xs_iith = np.array(xs)[:, ii]
        xs_masked = xs_iith[bool_config]

        ys_iith = np.array(ys[ii])
        ys_masked = ys_iith[bool_config]

        # need to make loglog slope
        temp_slope = (np.log10(ys_masked[1]) - np.log10(ys_masked[0])) / (np.log10(xs_masked[1]) - np.log10(xs_masked[0]))
        two_slopes.append(temp_slope)
        #plt.scatter(xs_masked, ys_masked, c = 'red')

# for 3 knots, calculate the index and knee point
    elif num_kn == 3: 

        bool_config = [bool(x) for x in configs[ii]]

        xs_iith = np.array(xs)[:, ii]
        xs_masked = xs_iith[bool_config]

        ys_iith = np.array(ys[ii])
        ys_masked = ys_iith[bool_config]

        temp_knee = xs_masked[1]

        # need to make loglog slope
        temp_slope1 = (np.log10(ys_masked[1]) - np.log10(ys_masked[0])) / (np.log10(xs_masked[1]) - np.log10(xs_masked[0]))
        temp_slope2 = (np.log10(ys_masked[2]) - np.log10(ys_masked[1])) / (np.log10(xs_masked[2]) - np.log10(xs_masked[1]))
        knees.append(temp_knee)
        three_slopes1.append(temp_slope1)
        three_slopes2.append(temp_slope2)

        #plt.scatter(xs_masked, ys_masked, c = 'blue')

print('Power Law:')
print('--------------')
print('mean | median | std')
print(f'{np.average(two_slopes)} | {np.median(two_slopes)} | {np.std(two_slopes)} \n')

print('Broken Power Law:')
print('--------------')
print('mean | median | std')
print(f'{np.average(knees)} | {np.median(knees)} | {np.std(knees)}')
print(f'{np.average(three_slopes1)} | {np.median(three_slopes1)} | {np.std(three_slopes1)}')
print(f'{np.average(three_slopes2)} | {np.median(three_slopes2)} | {np.std(three_slopes2)} \n')

print('Bayes Factors \n --------------')
configs, num_knots = return_knot_info(fit_results_omega, offset=0)
try:
    print(f'Bayes Factor SNR: {len([i for i in num_knots if i > 0]) / len([i for i in num_knots if i == 0])}')
except: print('DIVIDE BY ZERO ERROR')
try:
    print(f'Bayes Factor PL: {len([i for i in num_knots if i > 2]) / len([i for i in num_knots if i == 1])}')
except: print('DIVIDE BY ZERO ERROR')
try:
    print(f'Bayes Factor BPL: {len([i for i in num_knots if i > 2]) / len([i for i in num_knots if i == 2])}')
except: print('DIVIDE BY ZERO ERROR')

# plot histogram of number of knots turned on
plt.hist(num_knots, bins=np.linspace(-0.5, 30.5, num=32), edgecolor='w')
plt.axvline(np.average(num_knots), label = 'avg = '+str(np.average(num_knots)), linestyle='--', c='black')
plt.xticks(np.arange(0, 30))
plt.yscale("log")
plt.xlim(-1, 10)
plt.xlabel('num knots')
plt.legend()
plt.show()
plt.savefig('tsf_knot_hist.pdf')

plt.clf()

