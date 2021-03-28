from astropy import units as u
import numpy as np
from orbital_functions import eclipse_intermediate_times, minimum_transit_cosi


### Orbital parameters
def vary_eccentricity(system, eccentricity):
    planet = system.secondaries[0]
    planet.ecc = eccentricity
    return r"$e={:.2g}$".format(eccentricity)


def vary_impact_parameter(system, impact_parameter):
    planet = system.secondaries[0]
    planet.inc = np.rad2deg(np.arccos(minimum_transit_cosi(system) * impact_parameter))
    return r"$b={:.2g}$".format(impact_parameter)


# def vary_inclination(system, inclination):
#     planet = system.secondaries[0]
#     planet.inc = inclination.to(u.deg).value
#     return r"$i={}^\circ$".format(inclination)


### Rotation parameters
def vary_obliquity_LOS(system, obliquity_LOS):
    planet_map = system.secondaries[0].map
    planet_map.inc = 90 - obliquity_LOS.to(u.deg).value
    return r"$\psi_\mathrm{{LOS}}={:.2g}^\circ$".format(obliquity_LOS.to(u.deg).value)


def vary_obliquity_skyplane(system, obliquity_skyplane):
    planet_map = system.secondaries[0].map
    planet_map.obl = obliquity_skyplane.to(u.deg).value
    return r"$\psi_\mathrm{{sky}}={:.2g}^\circ$".format(obliquity_skyplane.to(u.deg).value)


def vary_rotation_rate_ratio(system, rotation_rate_ratio):
    planet = system.secondaries[0]
    planet.prot = planet.porb / rotation_rate_ratio

    t_midingress = eclipse_intermediate_times(system)["mid-ingress"]
    t_midtransit = planet.t0 * planet.time_unit
    planet.theta0 = (-360/(planet.prot*planet.time_unit) * (t_midingress-t_midtransit)).decompose().value
    return r"$\omega_\mathrm{{rot}}/\omega_\mathrm{{orb}}={:.2g}$".format(rotation_rate_ratio)
