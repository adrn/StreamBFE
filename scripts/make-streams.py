__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy.constants import G
from astropy import log as logger
from astropy.coordinates.angles import rotation_matrix
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
import h5py
import gary.coordinates as gc
import gary.dynamics as gd
from gary.units import galactic
from gary.dynamics.orbitfit import rotate_sph_coordinate

# Project
from streambfe import potentials, FRAME
from streambfe.coordinates import compute_stream_rotation_matrix
from streambfe.plot import plot_orbit, plot_data, plot_stream_obs

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
    q = gc.Quaternion.random()
    random_R = q.rotation_matrix

    # now rotate by random rotation matrix
    new_pos = random_R.dot(w0.pos)
    new_vel = random_R.dot(w0.vel)
    w0 = gd.CartesianPhaseSpacePosition(pos=new_pos, vel=new_vel)

    orbit = potential.integrate_orbit(w0, dt=1., nsteps=10000)
    logger.debug("Desired (peri,apo): ({:.1f},{:.1f}), estimated (peri,apo): ({:.1f},{:.1f})"
                 .format(pericenter, apocenter, orbit.pericenter(), orbit.apocenter()))

    return w0

def main(progenitor_mass, n_stars=1024, seed=42):
    np.random.seed(seed)

    _path,_ = os.path.split(os.path.abspath(__file__))
    top_path = os.path.abspath(os.path.join(_path, ".."))
    output_path = os.path.join(top_path, "output", "simulations")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for potential_name,potential in potentials.items():
        logger.info("Potential: {}".format(potential_name))

        this_output_path = os.path.join(output_path, potential_name)
        this_plot_path = os.path.join(this_output_path, 'plots')
        if not os.path.exists(this_output_path):
            os.mkdir(this_output_path)
        if not os.path.exists(this_plot_path):
            os.mkdir(this_plot_path)

        with h5py.File(os.path.join(this_output_path, "mock_stream_data.h5"), "w") as f:
            for i,(per,apo) in enumerate(per_apo):
                g = f.create_group(str(i))
                g.attrs['apocenter'] = apo
                g.attrs['pericenter'] = per

                # randomly draw integration time
                n_steps = np.random.randint(2000, 4000)
                g.attrs['n_steps'] = n_steps

                # get random initial conditions for given pericenter, apocenter
                w0 = peri_apo_to_random_w0(per, apo, potential)
                # g.attrs['w0'] = w0

                # integrate orbit
                logger.debug("Integrating for {} steps".format(n_steps))
                prog_orbit = potential.integrate_orbit(w0, dt=-1., nsteps=n_steps, t1=float(n_steps))
                prog_orbit = prog_orbit[::-1]

                m = progenitor_mass*u.Msun
                rtide = (m/potential.mass_enclosed(w0.pos))**(1/3.) * np.sqrt(np.sum(w0.pos**2))
                vdisp = np.sqrt(G*m/(2*rtide)).to(u.km/u.s)
                logger.debug("rtide, vdisp: {}, {}".format(rtide, vdisp))

                # Option 1: generate mock stream
                # stream = mockstream.fardal_stream(potential, prog_orbit=prog_orbit,
                #                                   prog_mass=m, release_every=1,
                #                                   Integrator=gi.DOPRI853Integrator)

                # Option 2: integrate a ball of test particle orbits
                std = gd.CartesianPhaseSpacePosition(pos=[rtide.value]*3*u.kpc,
                                                     vel=[vdisp.value]*3*u.km/u.s)

                ball_w = w0.w(galactic)[:,0]
                ball_std = std.w(galactic)[:,0]
                ball_w0 = np.random.normal(ball_w, ball_std, size=(n_stars,6))
                ball_w0 = gd.CartesianPhaseSpacePosition.from_w(ball_w0.T, units=galactic)

                stream_orbits = potential.integrate_orbit(ball_w0, dt=1., nsteps=n_steps)
                stream = stream_orbits[-1]

                # save simulated stream data
                g.attrs['mass'] = progenitor_mass

                g.create_dataset('pos', shape=stream.pos.shape, dtype=np.float64,
                                 data=stream.pos.decompose(galactic).value)
                g['pos'].attrs['unit'] = 'kpc'

                g.create_dataset('vel', shape=stream.vel.shape, dtype=np.float64,
                                 data=stream.vel.decompose(galactic).value)
                g['vel'].attrs['unit'] = 'kpc/Myr'

                # plot the orbit in cartesian coords
                fig = prog_orbit.plot(color='lightblue', alpha=0.5)
                fig = stream.plot(axes=fig.axes, marker='.', alpha=0.5)
                for ax in fig.axes:
                    ax.set_xlim(-apo-10, apo+10)
                    ax.set_ylim(-apo-10, apo+10)
                fig.savefig(os.path.join(this_plot_path, "orbit-{}.png".format(i)))

                # convert to sky coordinates and compute the stream rotation matrix
                stream_c,stream_v = stream.to_frame(coord.Galactic, **FRAME)
                R1 = compute_stream_rotation_matrix(stream_c, zero_pt="median")
                stream_rot = rotate_sph_coordinate(stream_c, R1)

                # pl.figure()
                # pl.scatter(stream_rot.lon.wrap_at(180*u.degree).degree, stream_rot.lat.degree)
                # pl.show()
                # return

                # leading tail should be +lon -- figure out if we need to flip the poles
                E_plus_lon = stream[stream_rot.lon.argmax()].energy(potential)
                E_minus_lon = stream[stream_rot.lon.wrap_at(180*u.degree).argmin()].energy(potential)
                E_diff = E_plus_lon - E_minus_lon

                # E_diff should be < 0 if the leading tail is at positive lon
                if E_diff > 0:
                    flip_R = rotation_matrix(180*u.degree, 'x')
                    lon_sign = -1.
                else:
                    flip_R = 1.
                    lon_sign = 1.

                lon_R = rotation_matrix(lon_sign*stream_rot.lon.wrap_at(180*u.degree).min(), 'z')
                R = lon_R*flip_R*R1

                stream_rot = rotate_sph_coordinate(stream_c, R)

                # pl.figure()
                # pl.scatter(stream_rot.lon.degree, stream_rot.lat.degree)
                # pl.show()

                # plot the orbit on the sky in galactic and in stream coordinates
                fig_gal,_ = plot_stream_obs(stream_c, stream_v)
                fig_rot,_ = plot_stream_obs(stream_rot, stream_v)
                fig_gal.savefig(os.path.join(this_plot_path, "stream-{}-gal.png".format(i)))
                fig_rot.savefig(os.path.join(this_plot_path, "stream-{}-rot.png".format(i)))

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
    parser.add_argument("--prog-mass", dest="prog_mass", default=1E4,
                        type=float, help="Progenitor mass")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(progenitor_mass=args.prog_mass, seed=args.seed)
