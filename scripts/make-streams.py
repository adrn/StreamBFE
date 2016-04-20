__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
import h5py
import gary.dynamics as gd
from gary.dynamics import mockstream
from gary.coordinates.quaternion import Quaternion
import gary.integrate as gi
from gary.units import galactic

# Project
from streambfe import potentials

# this sets the number of simulations to run
per_apo = [(15.,25)]*8 + [(25.,60)]*8 + [(85.,125)]*8

def peri_apo_to_random_w0(pericenter, apocenter, potential):
    def _func(E, L, r):
        return 2*(E - potential.value([r,0,0.]).value) - L**2/r**2

    def f(p):
        E,L = p
        return np.array([_func(E,L,apocenter), _func(E,L,pericenter)])

    r_start = np.random.uniform() * (apocenter - pericenter) + pericenter
    E0 = 0.5*0.2**2 + potential.value([(apocenter+pericenter)/2.,0,0]).value[0]
    L0 = 0.2 * r_start
    E,L = so.broyden1(f, [E0, L0])
    _rdot = np.sqrt(2*(E-potential.value([r_start,0,0.]).value[0]) - L**2/r_start**2)

    w0 = gd.CartesianPhaseSpacePosition(pos=[r_start,0.,0]*u.kpc,
                                        vel=[_rdot, L/r_start, 0.]*u.kpc/u.Myr)

    # sample a random rotation matrix
    q = Quaternion.random()
    random_R = q.rotation_matrix

    # now rotate by random rotation matrix
    new_pos = random_R.dot(w0.pos)
    new_vel = random_R.dot(w0.vel)
    w0 = gd.CartesianPhaseSpacePosition(pos=new_pos, vel=new_vel)

    orbit = potential.integrate_orbit(w0, dt=1., nsteps=10000)
    logger.debug("Desired (peri,apo): ({:.1f},{:.1f}), estimated (peri,apo): ({:.1f},{:.1f})"
                 .format(pericenter, apocenter, orbit.pericenter(), orbit.apocenter()))

    return w0

def main(seed=42):
    np.random.seed(seed)

    _path,_ = os.path.split(os.path.abspath(__file__))
    top_path = os.path.abspath(os.path.join(_path, ".."))
    output_path = os.path.join(top_path, "output", "simulations")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for potential_name,potential in potentials.items():
        logger.info("Potential: {}".format(potential_name))

        this_output_path = os.path.join(output_path, potential_name)
        if not os.path.exists(this_output_path):
            os.mkdir(this_output_path)

        with h5py.File(os.path.join(this_output_path, "streams.h5"), "w") as f:
            for i,(per,apo) in enumerate(per_apo):
                g = f.create_group(str(i))
                g.attrs['apocenter'] = apo
                g.attrs['pericenter'] = per

                # randomly draw integration time
                n_steps = np.random.randint(2000, 8000)
                g.attrs['n_steps'] = n_steps

                # get random initial conditions for given pericenter, apocenter
                w0 = peri_apo_to_random_w0(per, apo, potential)
                # g.attrs['w0'] = w0

                # integrate orbit
                logger.debug("Integrating for {} steps".format(n_steps))
                prog_orbit = potential.integrate_orbit(w0, dt=-1., nsteps=n_steps, t1=float(n_steps))
                prog_orbit = prog_orbit[::-1]

                # generate stream
                stream = mockstream.fardal_stream(potential, prog_orbit=prog_orbit,
                                                  prog_mass=1E5*u.Msun, release_every=1,
                                                  Integrator=gi.DOPRI853Integrator)

                # save simulated stream data
                g.attrs['mass'] = 1E5
                g.create_dataset('pos', shape=stream.pos.shape, dtype=np.float64,
                                 data=stream.pos.decompose(galactic).value)
                g.create_dataset('vel', shape=stream.vel.shape, dtype=np.float64,
                                 data=stream.vel.decompose(galactic).value)

                fig = prog_orbit.plot(color='lightblue', alpha=0.5)
                fig = stream.plot(axes=fig.axes, marker='.', alpha=0.5)
                for ax in fig.axes:
                    ax.set_xlim(-apo-10, apo+10)
                    ax.set_ylim(-apo-10, apo+10)
                fig.savefig(os.path.join(this_output_path, "orbit-{}.png".format(i)))

                pl.close('all')

                break # HACK TODO: remove this

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("--seed", dest="seed", default=42,
                        type=int, help="Random number seed")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(seed=args.seed)
