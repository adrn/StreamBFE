import astropy.coordinates as coord
import astropy.units as u

# Galactocentric reference frame to use for this project
galactocentric_frame = coord.Galactocentric(z_sun=0.*u.pc,
                                            galcen_distance=8.3*u.kpc)
vcirc = 238.*u.km/u.s
vlsr = [-11.1, 12.24, 7.25]*u.km/u.s

FRAME = {
    'galactocentric_frame': galactocentric_frame,
    'vcirc': vcirc,
    'vlsr': vlsr
}
