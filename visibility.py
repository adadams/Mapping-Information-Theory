import numpy as np
from astropy import units as u


def latlon_to_xyz(lat, lon):
    ### Convert lat-lon points to Cartesian points.
    if lat.ndim == 0:
        lat = np.atleast_1d(lat)
    if lon.ndim == 0:
        lon = np.atleast_1d(lon)
    R1 = RAxisAngle([1.0, 0.0, 0.0], -lat)
    R2 = RAxisAngle([0.0, 1.0, 0.0], lon)
    R = np.einsum("...ij,...jk->...ik", R2, R1)
    xyz = np.einsum("...ik,k->...i", R, [0.0, 0.0, 1.0])
    return xyz


def RAxisAngle(axis, theta):
    cost = np.cos(theta)
    sint = np.sin(theta)
    return np.einsum("ab...->...ab", np.reshape(
        [
            cost + axis[0] * axis[0] * (1 - cost),
            axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
            axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
            cost + axis[1] * axis[1] * (1 - cost),
            axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
            axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
            cost + axis[2] * axis[2] * (1 - cost),
        ],
        [3, 3, *np.shape(cost)],
    ))


def RotationMatrix(inc, obl, theta):
    R = [
        RAxisAngle((np.cos(obl), np.sin(obl), 0), 0.5 * np.pi - inc),
        RAxisAngle((0, 0, 1), -obl),
        RAxisAngle((1, 0, 0), 0.5 * np.pi),
        RAxisAngle((0, 0, 1), -theta),
        RAxisAngle((1, 0, 0), -0.5 * np.pi),
    ]
    R = R[0] @ R[1] @ R[2] @ R[3] @ R[4]
    return R


def visibility_kernel(system, times, lat, lon,
                      rotation_rate=None, theta0=None,
                      obl_LOS=None, obl_insky=None):
    planet = system.secondaries[0]
    if rotation_rate is None:
        rotation_rate = 2*np.pi*u.rad / (planet.porb*u.day)
    if theta0 is None:
        theta0 = planet.theta0 * u.deg
    if obl_LOS is not None:
        inc = np.deg2rad(90 - obl_LOS)
    else:
        inc = np.deg2rad(planet.map.inc)
    if obl_insky is not None:
        obl = np.deg2rad(obl_insky)
    else:
        obl = np.deg2rad(planet.map.obl)
    
    ### The Cartesian coordinates of the planet center in the system frame.
    xp, yp, zp = np.asarray(system.position(times))[:, 1, :]
    bp = np.sqrt(xp**2 + yp**2)
    Rs = system.primary.r
    Rp = system.secondaries[0].r
    
    planet_xyz = latlon_to_xyz(lat, lon) * Rp
    thetas = (theta0.to(u.rad) - (rotation_rate*(times*u.day))).to(u.deg)

    rotation_matrix = np.asarray([RotationMatrix(inc, obl, theta) for theta in thetas])
    sky_xyz = np.einsum("tij,...j->...ti", rotation_matrix, planet_xyz)
    
    ### Calculate the visibility before occultation.
    ### Negative values are on the hemisphere facing away from the observer.
    visibility = np.einsum("...i,i->...", sky_xyz, [0, 0, 1])
    visibility = np.where(visibility > 0, visibility, 0)
    
    ### Use system position to convert to system frame,
    ### then calculate impact parameter to check for occultation (b < 1).
    system_xyz = sky_xyz + np.asarray([xp, yp, zp]).T
    system_b = np.sqrt(system_xyz[..., 0]**2 + system_xyz[..., 1]**2)
    visibility = np.where(system_b > Rs, visibility, 0)
    
    return np.einsum("...t->t...", visibility)
