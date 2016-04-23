from __future__ import division, print_function

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as pl
import numpy as np
from gary.units import galactic
from gary.dynamics import orbitfit

# Project
from . import FRAME

__all__ = ["plot_stream_obs", "plot_data", "plot_orbit"]

def plot_stream_obs(stream_c, stream_v, wrap_angle=180*u.degree, **style_kw):
    style = dict(linestyle='none', marker='.', alpha=0.4)
    for k,v in style_kw.items():
        style[k] = v

    fig,axes = pl.subplots(2,3,figsize=(12,8), sharex=True)

    if hasattr(stream_c, 'name') and stream_c.name == 'galactic':
        lon = stream_c.l
        lat = stream_c.b
        axes[1,1].set_xlabel('$l$ [deg]')

    elif isinstance(stream_c, coord.BaseRepresentation):
        lon = stream_c.lon
        lat = stream_c.lat
        axes[1,1].set_xlabel('$\phi_1$ [deg]')

    lon = lon.wrap_at(wrap_angle)
    axes[0,0].plot(lon.degree, lat.degree, **style)
    axes[1,0].plot(lon.degree, stream_c.distance.value, **style)

    axes[0,1].plot(lon.degree, galactic.decompose(stream_v[0]).value, **style)
    axes[1,1].plot(lon.degree, galactic.decompose(stream_v[1]).value, **style)

    axes[0,2].plot(lon.degree, galactic.decompose(stream_v[2]).value, **style)

    fig.tight_layout()

    axes[1,2].set_visible(False)

    return fig,axes

def plot_data(data, err, R, wrap_angle=180*u.degree, fig=None, gal=True, **style_kw):
    if fig is None:
        fig,axes = pl.subplots(2,3,figsize=(12,8), sharex=True)
    else:
        axes = np.array(fig.axes).reshape(2,3)

    if gal:
        rep = coord.SphericalRepresentation(lon=data['phi1'], lat=data['phi2'],
                                            distance=data['distance'])
        g = coord.Galactic(orbitfit.rotate_sph_coordinate(rep, R.T))
        lon = g.l
        lat = g.b

        axes[1,1].set_xlabel('$l$ [deg]')

    else:
        lon = coord.Angle(data['phi1'])
        lat = coord.Angle(data['phi2'])
        axes[1,1].set_xlabel('$\phi_1$ [deg]')

    style = dict(linestyle='none', marker='.', ecolor='#aaaaaa')
    for k,v in style_kw.items():
        style[k] = v

    lon = lon.wrap_at(wrap_angle)
    axes[0,0].errorbar(lon.degree, lat.degree, 1E-8*lat.degree, **style)
    axes[1,0].errorbar(lon.degree, data['distance'].value, err['distance'].value, **style)

    axes[0,1].errorbar(lon.degree, galactic.decompose(data['mul']).value,
                       galactic.decompose(err['mul']).value, **style)
    axes[1,1].errorbar(lon.degree, galactic.decompose(data['mub']).value,
                       galactic.decompose(err['mub']).value, **style)

    axes[0,2].errorbar(lon.degree, galactic.decompose(data['vr']).value,
                       galactic.decompose(err['vr']).value, **style)

    try:
        fig.tight_layout()
    except AttributeError: # already called
        pass

    axes[1,2].set_visible(False)

    return fig,axes

def plot_orbit(orbit, wrap_angle=180*u.degree, fig=None, gal=True, R=None, **style_kw):
    orbit_c, orbit_v = orbit.to_frame(coord.Galactic, **FRAME)

    if fig is None:
        fig,axes = pl.subplots(2,3,figsize=(12,8), sharex=True)
    else:
        axes = np.array(fig.axes).reshape(2,3)

    style = dict(linestyle='-', marker=None, alpha=0.75, color='lightblue')
    for k,v in style_kw.items():
        style[k] = v

    if gal:
        lon = orbit_c.l
        lat = orbit_c.b
        axes[1,1].set_xlabel('$l$ [deg]')

    else:
        rep = orbitfit.rotate_sph_coordinate(orbit_c, R)
        lon = rep.lon
        lat = rep.lat
        axes[1,1].set_xlabel('$\phi_1$ [deg]')

    lon = lon.wrap_at(wrap_angle)
    axes[0,0].plot(lon.degree, lat.degree, **style)
    axes[1,0].plot(lon.degree, orbit_c.distance.value, **style)

    axes[0,1].plot(lon.degree, galactic.decompose(orbit_v[0]).value, **style)
    axes[1,1].plot(lon.degree, galactic.decompose(orbit_v[1]).value, **style)

    axes[0,2].plot(lon.degree, galactic.decompose(orbit_v[2]).value, **style)

    try:
        fig.tight_layout()
    except AttributeError: # already called
        pass

    axes[1,2].set_visible(False)

    return fig,axes
