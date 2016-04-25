__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict as odict

# Third-party
import astropy.units as u
import gary.potential as gp
from gary.units import galactic

potentials = odict()
potentials['plummer'] = gp.PlummerPotential(m=6E11, b=10, units=galactic)
potentials['triaxialnfw'] = gp.LeeSutoTriaxialNFWPotential(v_c=200*u.km/u.s, r_s=20*u.kpc,
                                                           a=1., b=0.8, c=0.6, units=galactic)

# halo_disk = gp.CCompositePotential()
# halo_disk['disk'] = gp.MiyamotoNagaiPotential(m=6.E10, a=3, b=0.28, units=galactic)
# halo_disk['halo'] = gp.PlummerPotential(m=1E12, b=10, units=galactic)
# potentials['halo+disk'] = halo_disk
