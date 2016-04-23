from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import abc
from collections import OrderedDict as odict

# Third-party
from astropy.constants import G
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
import six

import gary.potential as gp
import gary.coordinates as gc
from gary.dynamics import orbitfit
import gary.integrate as gi
from gary.units import galactic

import biff

# Project
from .galcen_frame import FRAME

_G = G.decompose(galactic).value

__all__ = ['OrbitfitModel', 'SCFOrbitfitModel', 'PlummerOrbitfitModel']

@six.add_metaclass(abc.ABCMeta)
class OrbitfitModel(object):

    def __init__(self, data, err, R, Potential, potential_param_names,
                 dt, n_steps, freeze=None):

        if freeze is None:
            freeze = dict()
        self.freeze = freeze

        self.data = data
        self.err = err
        self.R = R

        self.Potential = Potential
        self.potential_param_names = potential_param_names

        self.dt = dt
        self.n_steps = n_steps

    def pack(self, **kwargs):
        raise NotImplementedError()

    def unpack(self, p):
        count = 0
        count, orbit_pars = self._unpack_orbit(count, p)
        count, width_pars = self._unpack_width(count, p)
        count, potential_pars = self._unpack_potential(count, p)

        return orbit_pars, width_pars, potential_pars

    def _unpack_orbit(self, count, p):
        # the orbital initial conditions (these will always be in p)
        names = ['phi2','d','mul','mub','vr']

        pars = odict()
        for i,name in zip(range(count, count+len(names)), names):
            pars[name] = p[i]
        return count+5, pars

    def _unpack_width(self, count, p):
        # the nuisance width parameters
        names = ['phi2_sigma','d_sigma','mu_sigma','vr_sigma']

        pars = odict()
        for name in names:
            if name not in self.freeze:
                pars[name] = p[count]
                count += 1
            else:
                pars[name] = self.freeze[name]

        return count, pars

    def _unpack_potential(self, count, p):
        pars = odict()
        for name in self.potential_param_names:
            _name = "potential_{}".format(name)
            if _name not in self.freeze:
                pars[name] = p[count]
                count += 1
            else:
                pars[name] = self.freeze[_name]
        return count, pars

    # ------------------------------------------------------------------------
    # Priors
    #
    def ln_prior(self, p):
        orbit_pars, width_pars, potential_pars = self.unpack(p)

        lp = 0.
        lp += self._ln_orbit_prior(orbit_pars)
        lp += self._ln_width_prior(width_pars)
        lp += self._ln_potential_prior(potential_pars)

        return lp

    def _ln_orbit_prior(self, pars):
        lp = 0.

        # strong prior on phi2
        if pars['phi2'] < -np.pi/2. or pars['phi2'] > np.pi/2:
            return -np.inf
        # lp += norm.logpdf(pars['phi2'], loc=0., scale=phi2_sigma)

        return lp

    def _ln_width_prior(self, pars):
        lp = 0.

        # prior on instrinsic widths of stream
        for name in pars.keys():
            if name not in self.freeze:
                if pars[name] <= 0.:
                    return -np.inf
                lp += -np.log(pars[name])

        return lp

    @abc.abstractmethod
    def _ln_potential_prior(self, pars):
        return 0.

    # ------------------------------------------------------------------------
    # Helper functions for likelihood
    #
    def _mcmc_sample_to_coord(self, p):
        _,orbit_pars = self._unpack_orbit(0, p)
        rep = coord.SphericalRepresentation(lon=[0.]*u.radian,
                                            lat=[orbit_pars['phi2']]*u.radian,
                                            distance=[orbit_pars['d']]*u.kpc)
        return coord.Galactic(orbitfit.rotate_sph_coordinate(rep, self.R.T))

    def _mcmc_sample_to_w0(self, p):
        _,orbit_pars = self._unpack_orbit(0, p)
        c = self._mcmc_sample_to_coord(p)
        x0 = c.transform_to(FRAME['galactocentric_frame']).cartesian.xyz.decompose(galactic).value
        v0 = gc.vhel_to_gal(c, pm=(orbit_pars['mul']*u.rad/u.Myr,
                                   orbit_pars['mub']*u.rad/u.Myr),
                            rv=orbit_pars['vr']*u.kpc/u.Myr, **FRAME).decompose(galactic).value
        w0 = np.concatenate((x0, v0))
        return w0

    def ln_likelihood(self, p):
        """ Evaluate the stream orbit fit likelihood. """
        chi2 = 0.

        # unpack the parameters and the frozen parameters
        orbit_pars, width_pars, potential_pars = self.unpack(p)

        w0 = self._mcmc_sample_to_w0(p)

        # HACK: a prior on velocities
        vmag2 = np.sum(w0[3:]**2)
        chi2 += -vmag2 / (0.15**2)

        # integrate the orbit
        potential = self.Potential(units=galactic, **potential_pars)

        try:
            orbit = potential.integrate_orbit(w0, dt=self.dt, nsteps=self.n_steps,
                                              Integrator=gi.DOPRI853Integrator)
        except RuntimeError:
            return -np.inf

        # rotate the model points to stream coordinates
        model_c,model_v = orbit.to_frame(coord.Galactic, **FRAME)
        model_oph = orbitfit.rotate_sph_coordinate(model_c.spherical, self.R)

        # model stream points in ophiuchus coordinates
        model_phi1 = model_oph.lon
        model_phi2 = model_oph.lat.radian
        model_d = model_oph.distance.decompose(galactic).value
        model_mul,model_mub,model_vr = [x.decompose(galactic).value for x in model_v]

        # data, errors
        data = self.data
        err = self.err
        wi = width_pars

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

        var = wi['phi2_sigma']**2
        chi2 += -(phi2_interp(data_x) - phi2)**2 / var - np.log(var)

        _err = err['distance'].decompose(galactic).value
        var = _err**2 + wi['d_sigma']**2
        chi2 += -(d_interp(data_x) - dist)**2 / var - np.log(var)

        _err = err['mul'].decompose(galactic).value
        var = _err**2 + wi['mu_sigma']**2
        chi2 += -(mul_interp(data_x) - mul)**2 / var - np.log(var)

        _err = err['mub'].decompose(galactic).value
        var = _err**2 + wi['mu_sigma']**2
        chi2 += -(mub_interp(data_x) - mub)**2 / var - np.log(var)

        _err = err['vr'].decompose(galactic).value
        var = _err**2 + wi['vr_sigma']**2
        chi2 += -(vr_interp(data_x) - vr)**2 / var - np.log(var)

        return 0.5*chi2

    def ln_posterior(self, p):
        lp = self.ln_prior(p)
        if not np.isfinite(lp):
            return -np.inf

        ll = self.ln_likelihood(p)
        if not np.all(np.isfinite(ll)):
            return -np.inf

        return lp + ll.sum()

    def __call__(self, p):
        return self.ln_posterior(p)

class SCFOrbitfitModel(OrbitfitModel):
    """
    For spherical potentials.
    """

    def __init__(self, nmax, data, err, R, dt, n_steps, freeze=None):
        super(SCFOrbitfitModel, self).__init__(data, err, R, biff.SCFPotential, [],
                                               dt, n_steps, freeze)
        self.nmax = nmax
        self._xyz = np.zeros((1024, 3))
        self._xyz[:,0] = np.logspace(-1,2.5,self._xyz.shape[0])

    def _ln_potential_prior(self, pars):
        lp = 0.
        lp += -(pars['Snlm']**2).sum()

        grad = biff.gradient(self._xyz, Snlm=pars['Snlm'], Tnlm=pars['Tnlm'],
                             nmax=self.nmax, lmax=0, G=_G, M=pars['m'], r_s=pars['r_s'])
        if np.any(grad < 0.):
            return -np.inf

        dens = biff.density(self._xyz, Snlm=pars['Snlm'], Tnlm=pars['Tnlm'],
                            nmax=self.nmax, lmax=0, M=pars['m'], r_s=pars['r_s'])
        if np.any(dens < 0.):
            return -np.inf

        return lp

    def _unpack_potential(self, count, p):
        pars = odict()

        for name in ['m', 'r_s']:
            _name = "potential_{}".format(name)
            if _name not in self.freeze:
                pars[name] = p[count]
                count += 1
            else:
                pars[name] = self.freeze[_name]

        if "potential_Snlm" not in self.freeze:
            pars['Snlm'] = np.array(p[count:count+self.nmax+1])
            count += self.nmax+1
        else:
            pars['Snlm'] = np.array(self.freeze['potential_Snlm'])

        pars['Snlm'] = pars['Snlm'].reshape((self.nmax+1,1,1))
        pars['Tnlm'] = np.zeros((self.nmax+1,1,1))

        return count, pars

class PlummerOrbitfitModel(OrbitfitModel):
    def __init__(self, data, err, R, dt, n_steps, freeze=None):
        super(PlummerOrbitfitModel, self).__init__(data, err, R, gp.PlummerPotential,
                                                   ['m', 'b'], dt, n_steps, freeze)

    def _ln_potential_prior(self, pars):
        lp = 0.

        if 'potential_m' not in self.freeze:
            if pars['m'] < 5E10 or pars['m'] > 5E12:
                return -np.inf

        if 'potential_b' not in self.freeze:
            if pars['b'] < 0.1 or pars['b'] > 100.:
                return -np.inf

        return lp
