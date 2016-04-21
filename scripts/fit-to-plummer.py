from __future__ import division, print_function

# Standard library
import pickle
import os
import sys

# Third-party
from astropy.constants import G
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
from scipy.stats import norm
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
from streambfe.plot import *

# ----------------------------------------------------------------------------
# NEED TO CHANGE THESE WHEN CHANGING FIT POTENTIAL
#

def ln_potential_prior(potential_params, freeze=dict()):
    lp = 0.

    if 'potential_m' not in freeze:
        lp += norm.logpdf(np.log10(potential_params['m']), loc=12., scale=0.25)
        # lp += norm.logpdf(potential_params['b'], loc=30., scale=10.)

    if 'potential_b' not in freeze:
        b = potential_params['b']
        if b < 1 or b > 100:
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
    n_data = len(c)
    data = dict()
    err = dict()

    err['distance'] = 1E-3/100. * c.distance
    err['mul'] = 1E-5*u.mas/u.yr
    err['mub'] = 1E-5*u.mas/u.yr
    err['vr'] = 1E-3*u.km/u.s

    data['phi1'] = c.lon
    data['phi2'] = c.lat
    data['distance'] = c.distance + np.random.normal(0., err['distance'].value, size=n_data)*c.distance.unit
    data['mul'] = v[0] + np.random.normal(0., err['mul'].value, size=n_data)*err['mul'].unit
    data['mub'] = v[1] + np.random.normal(0., err['mub'].value, size=n_data)*err['mub'].unit
    data['vr'] = (v[2] + np.random.normal(0., err['vr'].value, size=n_data)*err['vr'].unit).to(u.km/u.s)

    return data, err

def main(mpi=False, n_walkers=None, n_burn=256, n_iterations=None, overwrite=False):
    np.random.seed(42)

    pool = get_pool(mpi=mpi)

    # save sampler to pickle file
    sampler_path = "plummer_sampler.pickle"
    data_path = "plummer_data.pickle"

    if os.path.exists(sampler_path) and overwrite:
        os.remove(sampler_path)

    if os.path.exists(data_path) and overwrite:
        os.remove(data_path)

    # potential to generate stream in
    true_potential = gp.PlummerPotential(m=1E12, b=30, units=galactic)
    fit_potential = gp.PlummerPotential(m=1E12, b=30, units=galactic)

    # integrate some orbit and generate mock stream
    w0 = gd.CartesianPhaseSpacePosition(pos=[0.,22,10]*u.kpc,
                                        vel=[-190,-25,-10]*u.km/u.s)

    m = 1E5*u.Msun
    rtide = (m/true_potential.mass_enclosed(w0.pos))**(1/3.) * np.sqrt(np.sum(w0.pos**2))
    vdisp = np.sqrt(G*m/(2*rtide)).to(u.km/u.s)
    logger.debug("rtide, vdisp: {}, {}".format(rtide, vdisp))

    # ------------------------------------------------------------------------
    #   Option 1
    # To generate streams with Fardal's method:
    # prog_orbit = true_potential.integrate_orbit(w0, dt=1., nsteps=5800)
    # logger.debug("peri,apo: {}, {}".format(prog_orbit.pericenter(), prog_orbit.apocenter()))

    # stream = mockstream.fardal_stream(true_potential, prog_orbit=prog_orbit,
    #                                   prog_mass=m, release_every=1,
    #                                   Integrator=gi.DOPRI853Integrator)

    # # only take leading tail for orbit fitting
    # leading = stream[1::2]

    # # downsample
    # idx = np.random.permutation(leading.pos.shape[1])[:128]
    # leading = leading[idx]
    # stream_c,stream_v = leading.to_frame(coord.Galactic, **FRAME)

    # fig = leading.plot()
    # prog_orbit[-1000:].plot(axes=fig.axes, alpha=1., color='lightblue')
    # pl.show()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    #   Option 2
    # To generate test particle balls of orbits:
    mean_w0 = gd.CartesianPhaseSpacePosition(pos=[0.,22,10]*u.kpc,
                                             vel=[-190,-25,-10]*u.km/u.s)

    std = gd.CartesianPhaseSpacePosition(pos=[rtide.value]*3*u.kpc,
                                         vel=[vdisp.value]*3*u.km/u.s)

    ball_w = mean_w0.w(galactic)[:,0]
    ball_std = std.w(galactic)[:,0]
    ball_w0 = emcee.utils.sample_ball(ball_w, ball_std, size=128)
    w0 = gd.CartesianPhaseSpacePosition.from_w(ball_w0.T, units=galactic)

    mean_orbit = true_potential.integrate_orbit(mean_w0, dt=1., nsteps=6000)
    stream_orbits = true_potential.integrate_orbit(w0, dt=1., nsteps=5800)
    stream = stream_orbits[-1]
    stream_c,stream_v = stream.to_frame(coord.Galactic, **FRAME)
    # ------------------------------------------------------------------------

    R = orbitfit.compute_stream_rotation_matrix(stream_c, align_lon='min')

    # rotate all data to stream coordinates
    rot_rep = orbitfit.rotate_sph_coordinate(stream_c, R)

    # _ = plot_stream_obs(stream_c, stream_v)
    # pl.show()

    # "observe" the data
    data,err = observe_data(rot_rep, stream_v)
    # _ = plot_data(data, err, R)
    # pl.show()
    # return

    # write data out
    with open(data_path, 'wb') as f:
        pickle.dump((data,err,R), f)

    # for now, freeze potential parameters and just sample over orbit
    freeze = dict()

    # these estimated from the plots
    mean_d = stream_c.distance.mean()
    freeze['phi2_sigma'] = (rtide / mean_d).decompose().value[0]
    freeze['d_sigma'] = rtide.value[0]
    freeze['vr_sigma'] = vdisp.decompose(galactic).value[0]
    term1 = (np.sqrt(stream_v[0]**2 + stream_v[1]**2).mean()/mean_d*rtide).to(u.mas/u.yr, equivalencies=u.dimensionless_angles())
    term2 = (vdisp/mean_d).to(u.mas/u.yr, equivalencies=u.dimensionless_angles())
    freeze['mu_sigma'] = (term1+term2).decompose(galactic).value[0]

    potential_param_names = list(fit_potential.parameters.keys())
    # for k in potential_param_names:
    #     logger.debug("freezing potential:{}".format(k))
    #     freeze['potential_{}'.format(k)] = fit_potential.parameters[k].value

    pot_guess = []
    for k in potential_param_names:
        _name = "potential_{}".format(k)
        if _name in freeze: continue
        logger.debug("varying potential:{}".format(k))
        pot_guess += [true_potential.parameters[k].value] # HACK

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

    # orbit = true_potential.integrate_orbit(_mcmc_sample_to_w0(p0_guess, R), dt=dt, nsteps=n_steps)
    # fig,axes = plot_data(data, err, R)
    # _ = plot_orbit(orbit, fig=fig)
    # _ = plot_orbit(mean_orbit[-300:], fig=fig)
    # pl.show()
    # return

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
    _ = sampler.run_mcmc(mcmc_p0, N=n_burn)
    logger.debug("finished burn-in")

    restart_p = np.median(sampler.chain[:,-1], axis=0)
    mcmc_p0 = emcee.utils.sample_ball(restart_p, 1E-3*restart_p, size=n_walkers)
    _ = sampler.run_mcmc(mcmc_p0, N=n_iterations)
    logger.info("finished sampling")

    pool.close()

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
