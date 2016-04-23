__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import pickle

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
import emcee
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
import h5py
import gary.dynamics as gd
from gary.util import get_pool
from gary.units import galactic
from gary.dynamics.orbitfit import rotate_sph_coordinate

# Project
from streambfe import potentials, FRAME
from streambfe.data import observe_data
from streambfe.plot import plot_data, plot_orbit
from streambfe import orbitfit

def main(potential_name, index, n_walkers=None, n_burn=0, n_iterations=1024,
         seed=42, mpi=False, overwrite=False, dont_optimize=False):
    np.random.seed(seed)
    pool = get_pool(mpi=mpi)

    true_potential_name = 'plummer'
    true_potential = potentials[true_potential_name]

    _path,_ = os.path.split(os.path.abspath(__file__))
    top_path = os.path.abspath(os.path.join(_path, ".."))
    simulation_path = os.path.join(top_path, "output", "simulations", true_potential_name)
    output_path = os.path.join(top_path, "output", "orbitfit", true_potential_name, potential_name)
    plot_path = os.path.join(output_path, "plots")
    sampler_file = os.path.join(output_path, "emcee.h5")
    model_file = os.path.join(output_path, "model-{}.pickle".format(index))

    if os.path.exists(sampler_file):
        with h5py.File(sampler_file, 'r') as f:
            if str(index) in f and not overwrite:
                logger.info("Orbit index {} already complete.".format(index))
                return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    logger.info("Potential: {}".format(potential_name))
    if potential_name == 'plummer':
        Model = orbitfit.PlummerOrbitfitModel
        kw = dict()

        freeze = dict()
        freeze['potential_b'] = 10.

        potential_guess = [6E11]
        mcmc_potential_std = [1E8]

    elif potential_name == 'scf':
        Model = orbitfit.SCFOrbitfitModel
        kw = dict(nmax=8)

        freeze = dict()
        freeze['potential_r_s'] = 10.

        potential_guess = [6E11] + [1.3,0,0,0,0,0,0,0,0] # HACK: try this first
        mcmc_potential_std = [1E8] + [1E-3]*9

    else:
        raise ValueError("Invalid potential name '{}'".format(potential_name))

    with h5py.File(os.path.join(simulation_path, "mock_stream_data.h5"), "r") as f:
        g = f[str(index)]
        pos = g['pos'][:] * u.Unit(f[str(index)]['pos'].attrs['unit'])
        vel = g['vel'][:] * u.Unit(f[str(index)]['vel'].attrs['unit'])
        R = g['R'][:]
        dt = g.attrs['dt']
        n_steps = g.attrs['n_steps']

    stream = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)
    stream_c,stream_v = stream.to_frame(coord.Galactic, **FRAME)
    stream_rot = rotate_sph_coordinate(stream_c, R)

    data,err = observe_data(stream_rot, stream_v)
    # fig = plot_data(data, err, R, gal=False)
    # pl.show()

    # freeze all intrinsic widths (all are smaller than errors)
    freeze['phi2_sigma'] = 1E-7
    freeze['d_sigma'] = 1E-3
    freeze['vr_sigma'] = 1E-5
    freeze['mu_sigma'] = 1E-7

    model = Model(data=data, err=err, R=R, dt=dt, n_steps=int(1.5*n_steps),
                  freeze=freeze, **kw)

    # pickle the model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # starting position for optimization
    p_guess = ([stream_rot[0].lat.radian, stream_rot[0].distance.value] +
               [v[0].decompose(galactic).value for v in stream_v] +
               potential_guess)

    logger.debug("ln_posterior at initialization: {}".format(model(p_guess)))

    if n_walkers is None:
        n_walkers = 8*len(p_guess)

    if not dont_optimize:
        logger.debug("optimizing ln_posterior first...")
        res = so.minimize(lambda p: -model(p), x0=p_guess, method='powell')
        p_best = res.x

        logger.debug("...done. optimization returned: {}".format(p_best))
        if not res.success:
            pool.close()
            raise ValueError("Failed to optimize!")
        logger.debug("ln_posterior at optimized p: {}".format(model(p_best)))

        # plot the orbit of the optimized parameters
        orbit = true_potential.integrate_orbit(model._mcmc_sample_to_w0(res.x),
                                               dt=dt, nsteps=n_steps)
        fig,_ = plot_data(data, err, R, gal=False)
        fig,_ = plot_orbit(orbit, fig=fig, R=R, gal=False)
        fig.savefig(os.path.join(plot_path, "optimized-{}.png".format(index)))

        mcmc_p0 = emcee.utils.sample_ball(res.x, 1E-3*np.array(p_best), size=n_walkers)
    else:

        mcmc_std = ([freeze['phi2_sigma'], freeze['d_sigma'], freeze['mu_sigma']] +
                    [freeze['mu_sigma'], freeze['vr_sigma']] + mcmc_potential_std)
        mcmc_p0 = emcee.utils.sample_ball(p_guess, mcmc_std, size=n_walkers)

    # now, create initial conditions for MCMC walkers in a small ball around the
    #   optimized parameter vector
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=len(p_best),
                                    lnpostfn=model, pool=pool)

    logger.info("running mcmc sampler with {} walkers for {} steps".format(n_walkers, n_iterations))
    if n_burn > 0:
        pos,_,_ = sampler.run_mcmc(mcmc_p0, N=n_burn)
        logger.debug("finished burn-in")
        sampler.reset()
    else:
        pos = mcmc_p0

    # restart_p = np.median(sampler.chain[:,-1], axis=0)
    # mcmc_p0 = emcee.utils.sample_ball(restart_p, 1E-3*restart_p, size=n_walkers)
    # sampler.reset()

    _ = sampler.run_mcmc(mcmc_p0, N=n_iterations)
    logger.info("finished sampling")

    pool.close()

    logger.debug("saving sampler data")

    if not os.path.exists(sampler_file):
        mode = 'w'
    else:
        mode = 'r+'

    with h5py.File(sampler_file, mode) as f:
        if str(index) in f:
            g = f[str(index)]
            del g['chain']
            del g['acceptance_fraction']
            del g['lnprobability']
        else:
            g = f.create_group(str(index))
        g['chain'] = sampler.chain
        g['acceptance_fraction'] = sampler.acceptance_fraction
        g['lnprobability'] = sampler.lnprobability

    if n_iterations > 256:
        logger.debug("plotting...")
        flatchain = np.vstack(sampler.chain[:,-256::4])

        fig,_ = plot_data(data, err, R, gal=False)
        for link in flatchain:
            orbit = true_potential.integrate_orbit(model._mcmc_sample_to_w0(link),
                                                   dt=dt, nsteps=n_steps)
            fig,_ = plot_orbit(orbit, fig=fig, R=R, gal=False, alpha=0.25)
        fig.savefig(os.path.join(plot_path, "mcmc-{}.png".format(index)))

    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. FILES.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("--seed", dest="seed", default=42,
                        type=int, help="Random number seed")
    parser.add_argument("-i", "--index", dest="index", required=True,
                        type=int, help="Index of stream to fit.")
    parser.add_argument("-p", "--potential", dest="potential_name", required=True,
                        type=str, help="Name of the fitting potential can be: "
                                       "plummer, scf")

    parser.add_argument("--dont-optimize", action="store_true", dest="dont_optimize",
                        default=False, help="Don't optimize, just sample from prior.")

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

    main(potential_name=args.potential_name, index=args.index, seed=args.seed,
         mpi=args.mpi, n_walkers=args.mcmc_walkers, n_iterations=args.mcmc_steps,
         overwrite=args.overwrite, dont_optimize=args.dont_optimize)
