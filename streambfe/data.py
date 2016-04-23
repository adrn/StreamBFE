from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np

def observe_data(c, v,
                 frac_distance_err=1.,
                 pm_err=0.2*u.mas/u.yr,
                 vr_err=5*u.km/u.s,
                 seed=42):
    np.random.seed(seed)

    n_data = len(c)
    data = dict()
    err = dict()

    err['distance'] = frac_distance_err/100. * c.distance
    err['mul'] = pm_err
    err['mub'] = pm_err
    err['vr'] = vr_err

    data['phi1'] = c.lon
    data['phi2'] = c.lat
    data['distance'] = c.distance + np.random.normal(0., err['distance'].value, size=n_data)*c.distance.unit
    data['mul'] = v[0] + np.random.normal(0., err['mul'].value, size=n_data)*err['mul'].unit
    data['mub'] = v[1] + np.random.normal(0., err['mub'].value, size=n_data)*err['mub'].unit
    data['vr'] = (v[2] + np.random.normal(0., err['vr'].value, size=n_data)*err['vr'].unit).to(u.km/u.s)

    return data, err
