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

def main(true_potential_name, fit_potential_name, index, pool,
         frac_distance_err=1, n_stars=32,
         n_walkers=None, n_burn=0, n_iterations=1024,
         overwrite=False, dont_optimize=False, name=None):

    true_potential = potentials[true_potential_name]

    _path,_ = os.path.split(os.path.abspath(__file__))
    top_path = os.path.abspath(os.path.join(_path, ".."))
    simulation_path = os.path.join(top_path, "output", "simulations", true_potential_name)
    output_path = os.path.join(top_path, "output", "orbitfit",
                               true_potential_name, fit_potential_name,
                               "d_{:.1f}percent".format(frac_distance_err))
    plot_path = os.path.join(output_path, "plots")
    sampler_file = os.path.join(output_path, "{}-emcee-{}.h5".format(name, index))
    model_file = os.path.join(output_path, "{}-model-{}.pickle".format(name, index))

    if os.path.exists(sampler_file) and not overwrite:
        logger.info("Orbit index {} already complete.".format(index))
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    logger.info("Potential: {}".format(fit_potential_name))
    if fit_potential_name == 'plummer':
        Model = orbitfit.PlummerOrbitfitModel
        kw = dict()

        freeze = dict()
        freeze['potential_b'] = 10.

        potential_guess = [6E11]
        mcmc_potential_std = [1E8]

    elif fit_potential_name == 'scf':
        Model = orbitfit.SCFOrbitfitModel
        kw = dict(nmax=8)

        freeze = dict()
        freeze['potential_r_s'] = 10.

        potential_guess = [6E11] + [1.3,0,0,0,0,0,0,0,0] # HACK: try this first
        mcmc_potential_std = [1E8] + [1E-3]*9

    elif fit_potential_name == 'triaxialnfw':
        Model = orbitfit.TriaxialNFWOrbitfitModel
        kw = dict()

        freeze = dict()
        freeze['potential_r_s'] = 20.
        # freeze['potential_a'] = 1.

        potential_guess = [(200*u.km/u.s).decompose(galactic).value, 1., 0.8, 0.6]
        mcmc_potential_std = [1E-5, 1E-4, 1E-4, 1E-4]

    else:
        raise ValueError("Invalid potential name '{}'".format(fit_potential_name))

    with h5py.File(os.path.join(simulation_path, "mock_stream_data.h5"), "r") as f:
        g = f[str(index)]
        pos = g['pos'][:] * u.Unit(f[str(index)]['pos'].attrs['unit'])
        vel = g['vel'][:] * u.Unit(f[str(index)]['vel'].attrs['unit'])
        R = g['R'][:]
        dt = g.attrs['dt']
        n_steps = g.attrs['n_steps']

    stream = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)
    idx = np.concatenate(([0], np.random.permutation(pos.shape[1])[:n_stars-1]))
    stream_c,stream_v = stream[idx].to_frame(coord.Galactic, **FRAME)
    stream_rot = rotate_sph_coordinate(stream_c, R)

    data,err = observe_data(stream_rot, stream_v,
                            frac_distance_err=frac_distance_err,
                            vr_err=10*u.km/u.s)
    # fig = plot_data(data, err, R, gal=False)
    # pl.show()

    # freeze all intrinsic widths (all are smaller than errors)
    freeze['phi2_sigma'] = 1E-7
    freeze['d_sigma'] = 1E-3
    freeze['vr_sigma'] = 5E-4
    freeze['mu_sigma'] = 1000.

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
        fig.savefig(os.path.join(plot_path, "{}-optimized-{}.png".format(name, index)))

        mcmc_p0 = emcee.utils.sample_ball(res.x, 1E-3*np.array(p_best), size=n_walkers)
    else:

        # mcmc_std = ([freeze['phi2_sigma'], freeze['d_sigma'], freeze['mu_sigma']] +
        #             [freeze['mu_sigma'], freeze['vr_sigma']] + mcmc_potential_std)

        # HACK:
        mcmc_std = ([freeze['phi2_sigma'], freeze['d_sigma'], freeze['mu_sigma']/1E9] +
                    [freeze['mu_sigma']/1E9, freeze['vr_sigma']] + mcmc_potential_std)
        mcmc_p0 = emcee.utils.sample_ball(p_guess, mcmc_std, size=n_walkers)

    # now, create initial conditions for MCMC walkers in a small ball around the
    #   optimized parameter vector
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=len(p_guess),
                                    lnpostfn=model, pool=pool)

    if n_burn > 0:
        logger.info("burning in sampler for {} steps".format(n_burn))
        pos,_,_ = sampler.run_mcmc(mcmc_p0, N=n_burn)
        logger.debug("finished burn-in")
        sampler.reset()
    else:
        pos = mcmc_p0

    logger.info("running mcmc sampler with {} walkers for {} steps".format(n_walkers, n_iterations))

    # restart_p = np.median(sampler.chain[:,-1], axis=0)
    # mcmc_p0 = emcee.utils.sample_ball(restart_p, 1E-3*restart_p, size=n_walkers)
    # sampler.reset()

    _ = sampler.run_mcmc(pos, N=n_iterations)
    logger.info("finished sampling")

    pool.close()

    logger.debug("saving sampler data")

    with h5py.File(sampler_file, 'w') as g:
        g['chain'] = sampler.chain
        g['acceptance_fraction'] = sampler.acceptance_fraction
        g['lnprobability'] = sampler.lnprobability
        g.attrs['n_stars'] = n_stars
        g.attrs['frac_distance_err'] = frac_distance_err

    if n_iterations > 256:
        logger.debug("plotting...")
        flatchain = np.vstack(sampler.chain[:,-256::4])

        fig,_ = plot_data(data, err, R, gal=False)
        for i,link in enumerate(flatchain):
            orbit = true_potential.integrate_orbit(model._mcmc_sample_to_w0(link),
                                                   dt=dt, nsteps=n_steps)
            fig,_ = plot_orbit(orbit, fig=fig, R=R, gal=False, alpha=0.25)
            if i == 32: break
        fig.savefig(os.path.join(plot_path, "{}-mcmc-{}.png".format(name, index)))

    sys.exit(0)

def continue_sampling(true_potential_name, fit_potential_name, index, pool, n_iterations,
                      frac_distance_err=1, name=None):

    true_potential = potentials[true_potential_name]

    _path,_ = os.path.split(os.path.abspath(__file__))
    top_path = os.path.abspath(os.path.join(_path, ".."))
    output_path = os.path.join(top_path, "output", "orbitfit",
                               true_potential_name, fit_potential_name,
                               "d_{:.1f}percent".format(frac_distance_err))
    plot_path = os.path.join(output_path, "plots")
    sampler_file = os.path.join(output_path, "{}-emcee-{}.h5".format(name, index))
    model_file = os.path.join(output_path, "{}-model-{}.pickle".format(name, index))

    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except UnicodeDecodeError:
        with open(model_file, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

    with h5py.File(sampler_file, 'r') as g:
        n_walkers,n_prev_steps,n_dim = g['chain'][:].shape
        mcmc_pos = g['chain'][:,-1,:][:]

    # now, create initial conditions for MCMC walkers in a small ball around the
    #   optimized parameter vector
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=n_dim,
                                    lnpostfn=model, pool=pool)

    logger.info("continuing mcmc sampler with {} walkers for {} steps".format(n_walkers, n_iterations))

    _ = sampler.run_mcmc(mcmc_pos, N=n_iterations)
    logger.info("finished sampling")

    pool.close()

    logger.debug("saving sampler data")

    with h5py.File(sampler_file, 'r+') as g:
        prev_chain = g['chain'][:]
        prev_lnprobability = g['lnprobability'][:]
        del g['chain']
        del g['acceptance_fraction']
        del g['lnprobability']

        g['chain'] = np.hstack((prev_chain, sampler.chain))
        g['acceptance_fraction'] = sampler.acceptance_fraction
        g['lnprobability'] = np.hstack((prev_lnprobability, sampler.lnprobability))

    if n_iterations > 256:
        logger.debug("plotting...")
        flatchain = np.vstack(sampler.chain[:,-256::4])

        fig,_ = plot_data(model.data, model.err, model.R, gal=False)
        for i,link in enumerate(flatchain):
            orbit = true_potential.integrate_orbit(model._mcmc_sample_to_w0(link),
                                                   dt=model.dt, nsteps=model.n_steps)
            fig,_ = plot_orbit(orbit, fig=fig, R=model.R, gal=False, alpha=0.25)
            if i == 32: break
        fig.savefig(os.path.join(plot_path, "{}-mcmc-{}.png".format(name, index)))

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
    parser.add_argument("-fp", "--fit-potential", dest="fit_potential_name", required=True,
                        type=str, help="Name of the fitting potential can be: "
                                       "plummer, scf, triaxialnfw")
    parser.add_argument("-tp", "--true-potential", dest="true_potential_name", required=True,
                        type=str, help="Name of the true potential can be: "
                                       "plummer, scf, triaxialnfw")
    parser.add_argument("--name", dest="name", required=True,
                        type=str, help="Name of the run.")

    parser.add_argument("--frac-d-err", dest="frac_distance_err", default=1,
                        type=float, help="Fractional distance errors.")
    parser.add_argument("-n", "--n-stars", dest="n_stars", default=32,
                        type=int, help="Number of 'stars'.")

    parser.add_argument("--dont-optimize", action="store_true", dest="dont_optimize",
                        default=False, help="Don't optimize, just sample from prior.")

    # emcee
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--mcmc-walkers", dest="mcmc_walkers", type=int, default=None,
                        help="Number of walkers.")
    parser.add_argument("--mcmc-steps", dest="mcmc_steps", type=int,
                        help="Number of steps to take MCMC.")
    parser.add_argument("--mcmc-burn", dest="mcmc_burn", type=int,
                        help="Number of burn-in steps to take MCMC.")

    parser.add_argument("--continue", action="store_true", dest="_continue",
                        default=False, help="Continue the mcmc")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)
    pool = get_pool(mpi=args.mpi)

    if args._continue:
        continue_sampling(true_potential_name=args.true_potential_name,
                          fit_potential_name=args.fit_potential_name, index=args.index,
                          pool=pool, n_iterations=args.mcmc_steps,
                          frac_distance_err=args.frac_distance_err, name=args.name)
        sys.exit(0)

    main(true_potential_name=args.true_potential_name, fit_potential_name=args.fit_potential_name,
         n_stars=args.n_stars, n_burn=args.mcmc_burn, pool=pool, n_walkers=args.mcmc_walkers,
         n_iterations=args.mcmc_steps, overwrite=args.overwrite, index=args.index,
         dont_optimize=args.dont_optimize, frac_distance_err=args.frac_distance_err,
         name=args.name)
