from __future__ import division, print_function

# Standard library
import pickle
import os
import sys
from math import log

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
import emcee
from gary.util import get_pool
from gary.units import galactic

import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.dynamics import mockstream, orbitfit

# Project
from streambfe import FRAME
from streambfe.orbitfit import ln_orbitfit_prior, ln_orbitfit_likelihood, _mcmc_sample_to_w0

# ----------------------------------------------------------------------------
# NEED TO CHANGE THESE WHEN CHANGING FIT POTENTIAL
#

def ln_potential_prior(potential_params, freeze=None):
    lp = 0.

    logm = np.log10(potential_params['m'])
    if logm < 11 or logm > 13:
        return -np.inf

    logb = np.log10(potential_params['b'])
    if logb < 0 or logb > 2:
        return -np.inf

    return lp
#
# ----------------------------------------------------------------------------

def ln_posterior(p, *args, **kwargs):
    lp = ln_orbitfit_prior(p, *args, **kwargs)
    if not np.isfinite(lp):
        return -np.inf

    ll = ln_orbitfit_likelihood(p, *args, **kwargs)
    if not np.all(np.isfinite(ll)):
        return -np.inf

    return lp + ll.sum()

def observe_data(c, v):
    frac_distance_err = 2.

    n_data = len(c)
    data = dict()
    err = dict()

    err['distance'] = frac_distance_err/100. * c.distance
    err['mul'] = 0.1*u.mas/u.yr
    err['mub'] = 0.1*u.mas/u.yr
    err['vr'] = 1.*u.km/u.s

    data['phi1'] = c.lon
    data['phi2'] = c.lat
    data['distance'] = c.distance + np.random.normal(0., err['distance'].value, size=n_data)*c.distance.unit
    data['mul'] = v[0] + np.random.normal(0., err['mul'].value, size=n_data)*err['mul'].unit
    data['mub'] = v[1] + np.random.normal(0., err['mub'].value, size=n_data)*err['mub'].unit
    data['vr'] = (v[2] + np.random.normal(0., err['vr'].value, size=n_data)*err['vr'].unit).to(u.km/u.s)

    return data, err

def plot_stream_obs(stream_c, stream_v):
    style = dict(ls='none', marker='.', alpha=0.4)
    fig,axes = pl.subplots(2,3,figsize=(12,8), sharex=True)

    axes[0,0].plot(stream_c.l.degree, stream_c.b.degree, **style)
    axes[1,0].plot(stream_c.l.degree, stream_c.distance.value, **style)

    axes[0,1].plot(stream_c.l.degree, galactic.decompose(stream_v[0]).value, **style)
    axes[1,1].plot(stream_c.l.degree, galactic.decompose(stream_v[1]).value, **style)

    axes[0,2].plot(stream_c.l.degree, galactic.decompose(stream_v[2]).value, **style)

    axes[1,1].set_xlabel('$l$ [deg]')

    fig.tight_layout()

    axes[1,2].set_visible(False)

    return fig,axes

def plot_data(data, err, R, fig=None):
    if fig is None:
        fig,axes = pl.subplots(2,3,figsize=(12,8), sharex=True)
    else:
        axes = np.array(fig.axes).reshape(2,3)

    rep = coord.SphericalRepresentation(lon=data['phi1'], lat=data['phi2'],
                                        distance=data['distance'])
    g = coord.Galactic(orbitfit.rotate_sph_coordinate(rep, R.T))

    style = dict(ls='none', marker='.', ecolor='#aaaaaa')

    axes[0,0].errorbar(g.l.degree, g.b.degree, 1E-8*g.b.degree, **style)
    axes[1,0].errorbar(g.l.degree, g.distance.value, err['distance'].value, **style)

    axes[0,1].errorbar(g.l.degree, galactic.decompose(data['mul']).value,
                       galactic.decompose(err['mul']).value, **style)
    axes[1,1].errorbar(g.l.degree, galactic.decompose(data['mub']).value,
                       galactic.decompose(err['mub']).value, **style)

    axes[0,2].errorbar(g.l.degree, galactic.decompose(data['vr']).value,
                       galactic.decompose(err['vr']).value, **style)

    axes[1,1].set_xlabel('$l$ [deg]')

    try:
        fig.tight_layout()
    except AttributeError: # already called
        pass

    axes[1,2].set_visible(False)

    return fig,axes

def plot_orbit(orbit, fig=None):
    orbit_c, orbit_v = orbit.to_frame(coord.Galactic, **FRAME)

    if fig is None:
        fig,axes = pl.subplots(2,3,figsize=(12,8), sharex=True)
    else:
        axes = np.array(fig.axes).reshape(2,3)

    style = dict(ls='-', marker=None, alpha=0.75, color='lightblue')

    axes[0,0].plot(orbit_c.l.degree, orbit_c.b.degree, **style)
    axes[1,0].plot(orbit_c.l.degree, orbit_c.distance.value, **style)

    axes[0,1].plot(orbit_c.l.degree, galactic.decompose(orbit_v[0]).value, **style)
    axes[1,1].plot(orbit_c.l.degree, galactic.decompose(orbit_v[1]).value, **style)

    axes[0,2].plot(orbit_c.l.degree, galactic.decompose(orbit_v[2]).value, **style)

    axes[1,1].set_xlabel('$l$ [deg]')

    try:
        fig.tight_layout()
    except AttributeError: # already called
        pass

    axes[1,2].set_visible(False)

    return fig,axes

def main(mpi=False, n_walkers=None, n_iterations=None, overwrite=False):
    np.random.seed(42)

    pool = get_pool(mpi=mpi)

    # potential to generate stream in
    true_potential = gp.PlummerPotential(m=1E12, b=30, units=galactic)
    fit_potential = gp.PlummerPotential(m=1E12, b=30, units=galactic)

    # integrate some orbit and generate mock stream
    w0 = gd.CartesianPhaseSpacePosition(pos=[0.,22,10]*u.kpc,
                                        vel=[-190,-25,-10]*u.km/u.s)
    prog_orbit = true_potential.integrate_orbit(w0, dt=1., nsteps=5800)
    logger.debug("peri,apo: {}, {}".format(prog_orbit.pericenter(), prog_orbit.apocenter()))

    stream = mockstream.fardal_stream(true_potential, prog_orbit=prog_orbit,
                                      prog_mass=1E5*u.Msun, release_every=1,
                                      Integrator=gi.DOPRI853Integrator)

    # only take leading tail for orbit fitting
    leading = stream[1::2]

    # downsample
    idx = np.random.permutation(leading.pos.shape[1])[:128]
    leading = leading[idx]

    prog_c,prog_v = prog_orbit.to_frame(coord.Galactic, **FRAME)
    stream_c,stream_v = leading.to_frame(coord.Galactic, **FRAME)

    # fig = leading.plot()
    # prog_orbit[-1000:].plot(axes=fig.axes, alpha=1., color='lightblue')
    # pl.show()

    R = orbitfit.compute_stream_rotation_matrix(stream_c, align_lon='max')

    # rotate all data to stream coordinates
    rot_rep = orbitfit.rotate_sph_coordinate(stream_c, R)

    # _ = plot_stream_obs(stream_c, stream_v)
    # pl.show()

    # "observe" the data
    data,err = observe_data(rot_rep, stream_v)
    # _ = plot_data(data, err, R)
    # pl.show()

    # for now, freeze potential parameters and just sample over orbit
    freeze = dict()

    # these estimated from the plots
    freeze['phi2_sigma'] = np.radians(0.25)
    freeze['d_sigma'] = 0.35
    freeze['vr_sigma'] = (1.5*u.km/u.s).decompose(galactic).value

    potential_param_names = list(fit_potential.parameters.keys())
    # for k in potential_param_names:
    #     logger.debug("freezing potential:{}".format(k))
    #     freeze['potential_{}'.format(k)] = fit_potential.parameters[k].value

    pot_guess = []
    for k in potential_param_names:
        _name = "potential_{}".format(k)
        if _name in freeze: continue
        logger.debug("varying potential:{}".format(k))
        pot_guess += [np.log10(true_potential.parameters[k].value)] # HACK

    idx = data['phi1'].argmin()
    p0_guess = [data['phi2'].radian[idx],
                data['distance'].decompose(galactic).value[idx],
                data['mul'].decompose(galactic).value[idx],
                data['mub'].decompose(galactic).value[idx],
                data['vr'].decompose(galactic).value[idx]]
    p0_guess = p0_guess + pot_guess
    logger.debug("Initial guess: {}".format(p0_guess))

    # integration stuff -- using leading tail, starting near prog, so need to integrate forward?
    dt = 1.
    n_steps = 120

    # orbit = potential.integrate_orbit(_mcmc_sample_to_w0(p0_guess, R), dt=dt, nsteps=n_steps)
    # fig,axes = plot_data(data, err, R)
    # _ = plot_orbit(orbit, fig=fig)
    # pl.show()

    # first, optimize to get a good guess to initialize MCMC
    args = (data, err, R, fit_potential.__class__, potential_param_names, ln_potential_prior,
            dt, n_steps, freeze)

    logger.info("optimizing ln_posterior...")
    res = so.minimize(lambda *args,**kwargs: -ln_posterior(*args, **kwargs),
                      x0=p0_guess, method='powell', args=args)
    logger.info("finished optimizing")
    logger.debug("optimization returned: {}".format(res))
    if not res.success:
        pool.close()
        raise ValueError("Failed to optimize!")

    # orbit = true_potential.integrate_orbit(_mcmc_sample_to_w0(res.x, R), dt=dt, nsteps=n_steps)
    # fig,axes = plot_data(data, err, R)
    # _ = plot_orbit(orbit, fig=fig)
    # pl.show()
    # return

    # now, create initial conditions for MCMC walkers in a small ball around the
    #   optimized parameter vector
    if n_walkers is None:
        n_walkers = 8*len(p0_guess)
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=len(p0_guess), lnpostfn=ln_posterior,
                                    args=args, pool=pool)
    mcmc_p0 = emcee.utils.sample_ball(res.x, 1E-3*np.array(p0_guess), size=n_walkers)

    if n_iterations is None:
        n_iterations = 1024

    logger.info("running mcmc sampler with {} walkers for {} steps".format(n_walkers, n_iterations))
    _ = sampler.run_mcmc(mcmc_p0, N=n_iterations)
    logger.info("finished sampling")

    pool.close()

    # same sampler to pickle file
    sampler_path = "plummer_sampler.pickle"

    if os.path.exists(sampler_path) and overwrite:
        os.remove(sampler_path)

    sampler.lnprobfn = None
    sampler.pool = None
    sampler.args = None
    logger.debug("saving emcee sampler to: {}".format(sampler_path))
    with open(sampler_path, 'wb') as f:
        pickle.dump(sampler, f)

    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. FILES.")

    # emcee
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--mcmc-walkers", dest="mcmc_walkers", type=int, default=None,
                        help="Number of walkers.")
    parser.add_argument("--mcmc-steps", dest="mcmc_steps", type=int,
                        help="Number of steps to take MCMC.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(mpi=args.mpi, n_walkers=args.mcmc_walkers, n_iterations=args.mcmc_steps,
         overwrite=args.overwrite)
