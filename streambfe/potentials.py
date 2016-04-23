__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict as odict

# Third-party
import gary.potential as gp
from gary.units import galactic

potentials = odict()
potentials['plummer'] = gp.PlummerPotential(m=6E11, b=10, units=galactic)

# halo_disk = gp.CCompositePotential()
# halo_disk['disk'] = gp.MiyamotoNagaiPotential(m=6.E10, a=3, b=0.28, units=galactic)
# halo_disk['halo'] = gp.PlummerPotential(m=1E12, b=10, units=galactic)
# potentials['halo+disk'] = halo_disk
