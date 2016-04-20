""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict as odict

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm

import gary.coordinates as gc
from gary.dynamics import orbitfit
import gary.integrate as gi
from gary.units import galactic
from gary.util import atleast_2d

# Project
from .galcen_frame import FRAME

def _unpack(p, potential_param_names, mcmc_to_potential_param, freeze=None):
    """ Unpack a parameter vector """

    if freeze is None:
        freeze = dict()

    # these are for the initial conditions
    phi2,d,mul,mub,vr = p[:5]
    count_ix = 5

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        phi2_sigma = p[count_ix]
        count_ix += 1
    else:
        phi2_sigma = freeze['phi2_sigma']

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        d_sigma = p[count_ix]
        count_ix += 1
    else:
        d_sigma = freeze['d_sigma']

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        vr_sigma = p[count_ix]
        count_ix += 1
    else:
        vr_sigma = freeze['vr_sigma']

    # potential parameters are always the last
    potential_params = odict()
    for name in potential_param_names:
        _name = "potential_{}".format(name)
        if _name not in freeze:
            potential_params[name] = p[count_ix]
            count_ix += 1
        else:
            potential_params[name] = freeze[_name]

    potential_params = mcmc_to_potential_param(potential_params)

    return (phi2,d,mul,mub,vr,phi2_sigma,d_sigma,vr_sigma,potential_params)

def ln_orbitfit_prior(p, data, err, R, Potential, potential_param_names, ln_potential_prior,
                      mcmc_to_potential_param, dt, n_steps, freeze=None):
    """
    Evaluate the prior over stream orbit fit parameters.
    See docstring for `ln_likelihood()` for information on args and kwargs.
    """

    # log prior value
    lp = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,phi2_sigma,d_sigma,vr_sigma,potential_params = _unpack(p, potential_param_names,
                                                                             mcmc_to_potential_param, freeze)

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        if phi2_sigma <= 0.:
            return -np.inf
        lp += -np.log(phi2_sigma)

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        if d_sigma <= 0.:
            return -np.inf
        lp += -np.log(d_sigma)

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        if vr_sigma <= 0.:
            return -np.inf
        lp += -np.log(vr_sigma)

    # strong prior on phi2
    if phi2 < -np.pi/2. or phi2 > np.pi/2:
        return -np.inf
    lp += norm.logpdf(phi2, loc=0., scale=phi2_sigma)

    # compute prior on potential params
    lp += ln_potential_prior(potential_params, freeze)

    return lp

def _mcmc_sample_to_coord(p, R):
    p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
    rep = coord.SphericalRepresentation(lon=p[0]*0.*u.radian,
                                        lat=p[0]*u.radian, # this index looks weird but is right
                                        distance=p[1]*u.kpc)
    return coord.Galactic(orbitfit.rotate_sph_coordinate(rep, R.T))

def _mcmc_sample_to_w0(p, R):
    p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
    c = _mcmc_sample_to_coord(p, R)
    x0 = c.transform_to(FRAME['galactocentric_frame']).cartesian.xyz.decompose(galactic).value
    v0 = gc.vhel_to_gal(c, pm=(p[2]*u.rad/u.Myr,p[3]*u.rad/u.Myr), rv=p[4]*u.kpc/u.Myr,
                        **FRAME).decompose(galactic).value
    w0 = np.concatenate((x0, v0))
    return w0

def ln_orbitfit_likelihood(p, data, err, R, Potential, potential_param_names, ln_potential_prior,
                           mcmc_to_potential_param, dt, n_steps, freeze=None):
    """ Evaluate the stream orbit fit likelihood. """
    chi2 = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,phi2_sigma,d_sigma,vr_sigma,potential_params = _unpack(p, potential_param_names,
                                                                             mcmc_to_potential_param, freeze)

    w0 = _mcmc_sample_to_w0([phi2,d,mul,mub,vr], R)[:,0]

    # HACK: a prior on velocities
    # vmag2 = np.sum(w0[3:]**2)
    # chi2 += -vmag2 / (0.15**2)

    # integrate the orbit
    potential = Potential(units=galactic, **potential_params)
    orbit = potential.integrate_orbit(w0, dt=dt, nsteps=n_steps,
                                      Integrator=gi.DOPRI853Integrator)

    # rotate the model points to stream coordinates
    model_c,model_v = orbit.to_frame(coord.Galactic, **FRAME)
    model_oph = orbitfit.rotate_sph_coordinate(model_c.spherical, R)

    # model stream points in ophiuchus coordinates
    model_phi1 = model_oph.lon
    model_phi2 = model_oph.lat.radian
    model_d = model_oph.distance.decompose(galactic).value
    model_mul,model_mub,model_vr = [x.decompose(galactic).value for x in model_v]

    # for independent variable, use cos(phi)
    data_x = np.cos(data['phi1'])
    model_x = np.cos(model_phi1)
    ix = np.argsort(model_x)

    # shortening for readability -- the data
    phi2 = data['phi2'].radian
    dist = data['distance'].decompose(galactic).value
    mul = data['mul'].decompose(galactic).value
    mub = data['mub'].decompose(galactic).value
    vr = data['vr'].decompose(galactic).value

    # define interpolating functions
    order = 3
    bbox = [-1, 1]
    phi2_interp = InterpolatedUnivariateSpline(model_x[ix], model_phi2[ix], k=order, bbox=bbox) # change bbox to units of model_x
    d_interp = InterpolatedUnivariateSpline(model_x[ix], model_d[ix], k=order, bbox=bbox)
    mul_interp = InterpolatedUnivariateSpline(model_x[ix], model_mul[ix], k=order, bbox=bbox)
    mub_interp = InterpolatedUnivariateSpline(model_x[ix], model_mub[ix], k=order, bbox=bbox)
    vr_interp = InterpolatedUnivariateSpline(model_x[ix], model_vr[ix], k=order, bbox=bbox)

    chi2 += -(phi2_interp(data_x) - phi2)**2 / phi2_sigma**2 - 2*np.log(phi2_sigma)

    _err = err['distance'].decompose(galactic).value
    chi2 += -(d_interp(data_x) - dist)**2 / (_err**2 + d_sigma**2) - np.log(_err**2 + d_sigma**2)

    _err = err['mul'].decompose(galactic).value
    chi2 += -(mul_interp(data_x) - mul)**2 / (_err**2) - 2*np.log(_err)

    _err = err['mub'].decompose(galactic).value
    chi2 += -(mub_interp(data_x) - mub)**2 / (_err**2) - 2*np.log(_err)

    _err = err['vr'].decompose(galactic).value
    chi2 += -(vr_interp(data_x) - vr)**2 / (_err**2 + vr_sigma**2) - np.log(_err**2 + vr_sigma**2)

    return 0.5*chi2
