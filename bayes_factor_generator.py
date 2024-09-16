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
import pygwb
import bilby
import astropy.cosmology
from copy import deepcopy
from pygwb.baseline import Baseline
import seaborn as sns
from popstock_tsf_helper import *
import lal
lal.swig_redirect_standard_output_error(False)
mpl.rcParams.update(mpl.rcParamsDefault)

def calc_Bayes(num_knots, num, den):
    try: return len([i for i in num_knots if i > num]) / len([i for i in num_knots if i == den])
    except: return np.inf

freqs = np.arange(20, 100, 0.03125)
N_samples = 1_000_000
N_offset = int(0.5 * N_samples)
sig_type = 'BPL'

Tvals = [60*60,     1* 24 * 60 * 60,    7* 24 * 60 * 60,        30* 24 * 60 * 60, 
         168* 24 * 60 * 60,             365.25* 24 * 60 * 60,   365.25*2* 24 * 60 * 60, 
         365.25*5* 24 * 60 * 60,        365.25*10* 24* 60 * 60, 365.25*20* 24 * 60 * 60, 
         365.25*50* 24 * 60 * 60,       365.25*100* 24 * 60 * 60 ] 

Bayes_HL = []
Bayes_CE = []

sigmas_CE = []
sigmas_HL = []

for T in Tvals:    
    # get the sigma curves based on detector pairs
    HL_sigma = get_sigma_from_noise_curves(['H1', 'L1'], freqs, T)
    sigmas_HL.append(HL_sigma)
    signal, data, data_obj = generate_data(sig_type, freqs, HL_sigma)
    fit_omega, fit_results_omega = sample_Omega(freqs, N_samples, data_obj)
    configs, num_knots = return_knot_info(fit_results_omega, offset=0)

    Bayes_HL.append(calc_Bayes(num_knots, 0, 0))
    Bayes_HL.append(calc_Bayes(num_knots, 2, 1))
    Bayes_HL.append(calc_Bayes(num_knots, 2, 2))
    
    CE_sigma = get_sigma_from_noise_curves(['CE'], freqs, T)
    sigmas_CE.append(CE_sigma)
    signal, data, data_obj = generate_data(sig_type, freqs, CE_sigma)
    fit_omega, fit_results_omega = sample_Omega(freqs, N_samples, data_obj)
    configs, num_knots = return_knot_info(fit_results_omega, offset=0)

    Bayes_CE.append(calc_Bayes(num_knots, 0, 0))
    Bayes_CE.append(calc_Bayes(num_knots, 2, 1))
    Bayes_CE.append(calc_Bayes(num_knots, 2, 2))
    
plt.loglog(np.array(Tvals)/(365.25*60*60*24), Bayes_HL[::3], marker='.', label='B_SNR')
plt.loglog(np.array(Tvals)/(365.25*60*60*24), Bayes_HL[1::3], marker='.', label='B_PL')
plt.loglog(np.array(Tvals)/(365.25*60*60*24), Bayes_HL[2::3], marker='.', label='B_BPL')

plt.loglog(np.array(Tvals)/(365.25*60*60*24), Bayes_CE[::3], marker='.', label='B_SNR')
plt.loglog(np.array(Tvals)/(365.25*60*60*24), Bayes_CE[1::3], marker='.', label='B_PL')
plt.loglog(np.array(Tvals)/(365.25*60*60*24), Bayes_CE[2::3], marker='.', label='B_BPL')

plt.xlabel('Observing Time [yrs]')
plt.ylabel('Bayes Factor')
print(Bayes_HL)
plt.xlim(0,15)
plt.legend()
plt.savefig('Bayes_Factors.pdf')
