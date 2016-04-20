from __future__ import division, print_function

# Third-party
import astropy.coordinates as coord
import matplotlib.pyplot as pl
import numpy as np
from gary.units import galactic
from gary.dynamics import orbitfit

# Project
from . import FRAME

__all__ = ["plot_stream_obs", "plot_data", "plot_orbit"]

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
