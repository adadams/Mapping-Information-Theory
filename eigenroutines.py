from astropy import constants as c
from astropy import units as u
import numpy as np
from orbital_functions import *
import re
from sklearn.decomposition import PCA
import starry


def cut_to_eclipse(system, time_resolution, extent="T1-T4"):
    intervals = re.findall(r"T[1-4]-T[1-4]", extent)

    if intervals:
        porb = system.secondaries[0].porb
        full_orbit_times = np.linspace(-0.5, 0.5, num=time_resolution) * porb

        xp, yp, zp = np.asarray(system.position(full_orbit_times))[:, 1, :]
        bp = np.sqrt(xp**2 + yp**2)
        Rs = system.primary.r
        Rp = system.secondaries[0].r

        if "T1-T4" in intervals:
            eclipse_condition = (bp < Rs+Rp) & (zp < 0)
            return full_orbit_times[eclipse_condition]
        elif "T2-T3" in intervals:
            eclipse_condition = (bp < Rs-Rp) & (zp < 0)
            return full_orbit_times[eclipse_condition]
        else:
            partial_eclipse = (bp >= Rs) & (bp < Rp+Rs) & (zp < 0)
            if "T1-T2" not in intervals:
                eclipse_condition = partial_eclipse & (xp < 0)
            if "T3-T4" not in intervals:
                eclipse_condition = partial_eclipse & (xp > 0)
            if all(~partial_eclipse):
                return None
            else:
                return full_orbit_times[eclipse_condition]
    else:
        return None


# Currently not set up. Work in progress!
'''
def generate_grid_of_eigenmodes(system,
                                times,
                                function_to_vary_parameter,
                                parameter_values,
                                max_degree=20,
                                num_eigencurves=50):
    planet = system.secondaries[0]
    component_sets = []
    harmonic_sets = []
    score_sets = []
    
    possible_fixed_labels = [r"$b={:.2g}$".format(np.cos(np.deg2rad(planet.inc))/minimum_transit_cosi(system)),
                             r"$e={:.2g}$".format(planet.ecc),
                             r"$\omega_\mathrm{{rot}}/\omega_\mathrm{{orb}}={:.2g}$".format(planet.porb/planet.prot),
                             r"$\psi_\mathrm{{LOS}}={:.2g}^\circ$".format(90-planet.map.inc),
                             r"$\psi_\mathrm{{sky}}={:.2g}^\circ$".format(planet.map.obl)]
    variable_labels = []

    for parameter_value in parameter_values:
        parameter_label = function_to_vary_parameter(system, parameter_value)
        variable_labels.append(parameter_label)
        components, harmonics, scores = run_PCA(system,
                                                times,
                                                max_degree,
                                                num_eigencurves)
        component_sets.append(components)
        harmonic_sets.append(harmonics)
        score_sets.append(scores)

    parameter_regex = re.compile(".*=")
    fixed_labels = []
    for possible_fixed_label, label_check in zip(possible_fixed_labels,
                                                 [parameter_regex.match(fixed_label).group() for fixed_label in possible_fixed_labels]):
        if parameter_regex.match(parameter_label).group() != label_check:
            fixed_labels.append(possible_fixed_label.replace("\\\\", "\\"))
        else:
            fixed_labels.append(r"${}\textendash{}".format(variable_labels[0].split("$")[1], variable_labels[-1].split("=")[1]))
            # fixed_labels.append(parameter_label.split("=")[0]+r"$ varies")

    return {"component sets": np.asarray(component_sets),
            "harmonic sets": np.asarray(harmonic_sets),
            "score sets": np.asarray(score_sets),
            "fixed labels": fixed_labels,
            "variable labels": variable_labels}
'''


def generate_hotspot_data(system, times,
                          resolution=1*u.deg,
                          latitude=0*u.deg,
                          longitude_offset=0*u.deg,
                          angular_size=5*u.deg,
                          subobserver_reference_point="mid-ingress"):
    planet = system.secondaries[0]

    reference_time = eclipse_intermediate_times(system)[subobserver_reference_point]
    t_midtransit = planet.t0 * planet.time_unit
    subobserver_reference_longitude = planet.theta0*planet.angle_unit + (360*u.deg/(planet.prot*planet.time_unit) * (reference_time-t_midtransit))

    num_theta = int(((180*u.deg)/resolution).decompose().value + 1)
    num_phi = int(((360*u.deg)/resolution).decompose().value)
    theta_arr = np.deg2rad(np.linspace(0, 180, num_theta))
    phi_arr = np.deg2rad(np.linspace(-180, 180-resolution.to(u.deg).value, num_phi))
    thetas, phis = np.meshgrid(theta_arr, phi_arr)

    hotspot_theta = (90*u.deg-latitude).to(u.rad).value
    hotspot_phi = (subobserver_reference_longitude+longitude_offset).to(u.rad).value
    hotspot_std = angular_size.to(u.rad).value
    hotspot_concentration = 1/np.sqrt(hotspot_std)
    hotspot_map = np.exp(hotspot_concentration * (
        np.cos(thetas)*np.cos(hotspot_theta) + np.sin(thetas)*np.sin(hotspot_theta)*np.cos(phis-hotspot_phi)
        )).T
    planet.map.load(hotspot_map)
    
    return {"map": planet.map,
            "light curve": system.flux(times)}


def generate_set_of_eigenmodes(system,
                               times,
                               function_to_vary_parameter,
                               parameter_values,
                               max_degree=2,
                               num_eigencurves=4):
    planet = system.secondaries[0]
    component_sets = []
    constant_curves = []
    harmonic_sets = []
    score_sets = []
    
    possible_fixed_labels = [r"$b={:.2g}$".format(np.cos(np.deg2rad(planet.inc))/minimum_transit_cosi(system)),
                             r"$e={:.2g}$".format(planet.ecc),
                             r"$\omega_\mathrm{{rot}}/\omega_\mathrm{{orb}}={:.2g}$".format(planet.porb/planet.prot),
                             r"$\psi_\mathrm{{LOS}}={:.2g}^\circ$".format(90-planet.map.inc),
                             r"$\psi_\mathrm{{sky}}={:.2g}^\circ$".format(planet.map.obl)]
    variable_labels = []

    for parameter_value in parameter_values:
        parameter_label = function_to_vary_parameter(system, parameter_value)
        variable_labels.append(parameter_label)
        components, constant_curve, harmonics, scores = run_PCA(system,
                                                                times,
                                                                max_degree,
                                                                num_eigencurves)
        component_sets.append(components)
        constant_curves.append(constant_curve)
        harmonic_sets.append(harmonics)
        score_sets.append(scores)

    parameter_regex = re.compile(".*=")
    fixed_labels = []
    for possible_fixed_label, label_check in zip(possible_fixed_labels,
                                                 [parameter_regex.match(fixed_label).group() for fixed_label in possible_fixed_labels]):
        if parameter_regex.match(parameter_label).group() != label_check:
            fixed_labels.append(possible_fixed_label.replace("\\\\", "\\"))
        else:
            fixed_labels.append(r"${}\textendash{}".format(variable_labels[0].split("$")[1], variable_labels[-1].split("=")[1]))
            # fixed_labels.append(parameter_label.split("=")[0]+r"$ varies")

    return {"component sets": np.asarray(component_sets),
            "constant curves": np.asarray(constant_curves),
            "harmonic sets": np.asarray(harmonic_sets),
            "score sets": np.asarray(score_sets),
            "fixed labels": fixed_labels,
            "variable labels": variable_labels}


def generate_set_of_test_planets(system,
                                 times,
                                 function_to_vary_parameter,
                                 parameter_values,
                                 latitude=0*u.deg,
                                 longitude_offset=0*u.deg,
                                 angular_size=5*u.deg):
    planet = system.secondaries[0]
    hotspot_maps = []
    hotspot_curves = []
    
    possible_fixed_labels = [r"$b={:.2g}$".format(np.cos(np.deg2rad(planet.inc))/minimum_transit_cosi(system)),
                             r"$e={:.2g}$".format(planet.ecc),
                             r"$\omega_\mathrm{{rot}}/\omega_\mathrm{{orb}}={:.2g}$".format(planet.porb/planet.prot),
                             r"$\psi_\mathrm{{LOS}}={:.2g}^\circ$".format(90-planet.map.inc),
                             r"$\psi_\mathrm{{sky}}={:.2g}^\circ$".format(planet.map.obl)]
    variable_labels = []

    for parameter_value in parameter_values:
        parameter_label = function_to_vary_parameter(system, parameter_value)
        variable_labels.append(parameter_label)
        hotspot_data = generate_hotspot_data(system,
                                             times,
                                             latitude=latitude,
                                             longitude_offset=longitude_offset,
                                             angular_size=angular_size)
        hotspot_maps.append(hotspot_data["map"])
        hotspot_curves.append(hotspot_data["light curve"])

    parameter_regex = re.compile(".*=")
    fixed_labels = []
    for possible_fixed_label, label_check in zip(possible_fixed_labels,
                                                 [parameter_regex.match(fixed_label).group() for fixed_label in possible_fixed_labels]):
        if parameter_regex.match(parameter_label).group() != label_check:
            fixed_labels.append(possible_fixed_label.replace("\\\\", "\\"))
        else:
            fixed_labels.append(r"${}\textendash{}".format(variable_labels[0].split("$")[1], variable_labels[-1].split("=")[1]))
            # fixed_labels.append(parameter_label.split("=")[0]+r"$ varies")

    return {"maps": hotspot_maps,
            "light curves": hotspot_curves,
            "fixed labels": fixed_labels,
            "variable labels": variable_labels}


def generate_system(system_properties, max_degree=20):
    star = starry.Primary(starry.Map(udeg=2), **system_properties["primary"])
    star.map[1:] = [0.5, 0.25]
    planet = starry.Secondary(starry.Map(ydeg=max_degree,
                                         **system_properties["secondary map"]),
                              **system_properties["secondary"])
    system = starry.System(star, planet)

    reference_times = eclipse_intermediate_times(system)
    t_mideclipse = reference_times["mid-eclipse"]
    t_midingress = reference_times["mid-ingress"]
    t_midtransit = planet.t0 * planet.time_unit
    planet.t0 = planet.t0 - t_mideclipse.to(planet.time_unit).value
    planet.theta0 = (-360/(planet.prot*u.day) * (t_midingress-t_midtransit)).decompose().value

    return system


def iterative_fit(basis_curves, data_curve, basis_cutoff=50, tolerance=5e-3):
    return None


def rotation_parameter(system, timescale="ingress"):
    planet = system.secondaries[0]

    porb = planet.porb * planet.time_unit
    duration = eclipse_durations(system)[timescale]

    return (porb/(duration)).decompose().value


def run_PCA(system,
            times,
            max_degree=2,
            num_eigencurves=4):
    planet = system.secondaries[0]
    map_inc = planet.map.inc
    map_obl = planet.map.obl

    ### Re-initialize a constant map with the appropriate orientation.
    for l in range(1, max_degree+1):
        for m in range(-l, l+1):
            planet.map[l, m] = 0
    offset_flux = system.flux(times)
    harmonic_curves = []

    l_last = 1
    m_last = np.nan
    for l in range(1, max_degree+1):
        for m in range(-l, l+1):
            if m > -l:
                l_last = l
            if not np.isnan(m_last):
                planet.map[l_last, m_last] = 0
            planet.map[l, m] = 1
            flux = system.flux(times) - offset_flux
            harmonic_curves.append(flux)
            m_last = m
        

    harmonic_curves = np.asarray(harmonic_curves).T
    pca = PCA(n_components=num_eigencurves)
    pca.fit(harmonic_curves)

    return pca.components_, offset_flux, harmonic_curves, pca.explained_variance_ratio_
