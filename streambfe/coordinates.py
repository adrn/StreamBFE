from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates.angles import rotation_matrix
import numpy as np
from scipy.optimize import minimize

def _rotation_opt_func(th, xyz):
    if th < -1. or th > 1.:
        return np.inf
    R = rotation_matrix(np.arccos(th)*u.radian, 'x')
    xyz2 = coord.CartesianRepresentation(R.dot(xyz)*u.one).xyz
    return np.sum(xyz2[2]**2).value

def compute_stream_rotation_matrix(coords, zero_pt='mean'):
    """
    Compute the rotation matrix to go from the frame of the input
    coordinate to closely align the equator with the stream.

    .. note::

        if using zero_pt='mean' or 'median, the coordinates in the new system
        might not hit (0,0) because the mean / median is taken across each
        dimension separately.

    Parameters
    ----------
    coords : :class:`astropy.coordinate.SkyCoord`, :class:`astropy.coordinate.BaseCoordinateFrame`
        The coordinates of the stream stars.
    zero_pt : str,:class:`astropy.coordinate.SkyCoord`, :class:`astropy.coordinate.BaseCoordinateFrame`
        The origin of the new coordinates in the old coordinates.

    Returns
    -------
    R : :class:`~numpy.ndarray`
        A 3 by 3 rotation matrix (has shape ``(3,3)``) to convert heliocentric,
        Cartesian coordinates in the input coordinate frame to stream coordinates.

    """
    if hasattr(coords, 'spherical'):
        sph = coords.spherical
    else:
        sph = coords

    if zero_pt == 'mean' or zero_pt == 'median':
        lon = sph.lon.wrap_at(360*u.degree)
        lat = sph.lat.wrap_at(90*u.degree)

        if np.any(lon < 10*u.degree) and np.any(lon > 350*u.degree): # it's wrapping
            lon = lon.wrap_at(180*u.degree)

        if np.any(lat < -80*u.degree) and np.any(lat > 80*u.degree): # it's wrapping
            lat = lat.wrap_at(180*u.degree)

        zero_pt = coord.UnitSphericalRepresentation(lon=getattr(np, zero_pt)(lon),
                                                    lat=getattr(np, zero_pt)(lat))

    elif hasattr(zero_pt, 'spherical'):
        zero_pt = zero_pt.spherical

    # first determine rotation matrix to put zero_pt at (0,0)
    R1 = rotation_matrix(zero_pt.lon, 'z')
    R2 = rotation_matrix(-zero_pt.lat, 'y')

    xyz2 = (R2*R1).dot(sph.represent_as(coord.CartesianRepresentation).xyz)

    # determine initial guess for angle with some math trickery
    _r = np.sqrt(xyz2[1]**2 + xyz2[2]**2)
    ix = _r.argmin()
    if ix == 0:
        ix += 1
    elif ix == (xyz2[1].size-1):
        ix -= 1

    guess = 180*u.degree - np.arctan2(xyz2[2][ix+1]-xyz2[2][ix], xyz2[1][ix+1]-xyz2[1][ix]).to(u.degree)
    res = minimize(_rotation_opt_func, x0=np.cos(guess), args=(xyz2,), method="powell")

    if not res.success:
        raise ValueError("Failed to compute final alignment angle.")

    if np.allclose(np.abs(res.x), 1., atol=1E-5):
        guess = 180*u.degree - guess
        res = minimize(_rotation_opt_func, x0=np.cos(guess), args=(xyz2,), method="powell")

    R3 = rotation_matrix(np.arccos(res.x)*u.radian, 'x')
    R = R3*R2*R1

    return R
