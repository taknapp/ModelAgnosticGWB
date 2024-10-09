#!/usr/bin/env python3
#
import gwpopulation
import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator



class SplineRedshift(gwpopulation.models.redshift._Redshift):
    """
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)
    See https://arxiv.org/abs/2003.12152 (2) for the normalisation

    The parameterisation differs a little from there, we use

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= \frac{(1 + z)^\gamma}{1 + (\frac{1 + z}{1 + z_p})^\kappa}

    Parameters
    ----------
    gamma: float
        Slope of the distribution at low redshift
    kappa: float
        Slope of the distribution at high redshift
    z_peak: float
        Redshift at which the distribution peaks.
    z_max: float, optional
        The maximum redshift allowed.
    """
    base_variable_names = ["gamma", "kappa", "z_peak"]

    def psi_of_z(self, redshift, **parameters):
        amplitudes = np.array([parameters[key] for key in parameters if 'amplitude' in key])
        configuration = np.array([parameters[key] for key in parameters if 'configuration' in key])
        xvals = np.array([parameters[key] for key in parameters if 'xval' in key])
        return interp1d(xvals[configuration], amplitudes[configuration], fill_value="extrapolate")(redshift)

def createSplineRedshift(max_knots=10):
    class SplineRedshift(gwpopulation.models.redshift._Redshift):
        r"""
        Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)
        See https://arxiv.org/abs/2003.12152 (2) for the normalisation

        The parameterisation differs a little from there, we use

        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) = interpolation with at most max_knots in z-space and a list of amplitudes

        Parameters
        ----------
        amplitudes: float
            Slope of the distribution at low redshift
        configuration: float
            Slope of the distribution at high redshift
        xvals: float
            Redshift at which the distribution peaks.
        """
        variable_names = [f"amplitudes{ii}" for ii in range(max_knots)] + \
            [f"configuration{ii}" for ii in range(max_knots)] + \
            [f"xvals{ii}" for ii in range(max_knots)]

        def psi_of_z(self, redshift, **parameters):
            amplitudes = np.array([parameters[key] for key in parameters if 'amplitude' in key])
            configuration = np.array([parameters[key] for key in parameters if 'configuration' in key])
            xvals = np.array([parameters[key] for key in parameters if 'xval' in key])
            if np.sum(configuration)==0:
                tmp = 0 * np.zeros(redshift.size)
            elif np.sum(configuration)==1:
                tmp = np.ones(redshift.size) * amplitudes[configuration]
            else:
                tmp = interp1d(xvals[configuration],
                               amplitudes[configuration],
                               fill_value="extrapolate")(redshift)
                # tmp = Akima1DInterpolator(xvals[configuration],
                #                           amplitudes[configuration])(redshift,
                #                                                 extrapolate=True)
            return tmp
    return SplineRedshift
