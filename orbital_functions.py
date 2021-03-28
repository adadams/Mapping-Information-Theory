from astropy import constants as c
from astropy import units as u
import numpy as np
from scipy.optimize import fsolve


def anomaly_from_projected_distance(system, projected_distance, estimate):
    planet = system.secondaries[0]

    porb = planet.porb * planet.time_unit
    a = semimajor_axis(system)
    e = planet.ecc
    sin_i = np.sin(planet.inc*planet.angle_unit)
    arg_peri = planet.omega*planet.angle_unit

    projection_anomaly = lambda anomaly: (1-(sin_i*np.sin(anomaly*u.deg+arg_peri))**2)/(1+e*np.cos(anomaly*u.deg))**2 - (projected_distance/(a*(1-e**2)))**2

    return ((fsolve(projection_anomaly, estimate))[0] % 360) * u.deg


def duration_from_anomalies(system, anomaly_A, anomaly_B):
    planet = system.secondaries[0]

    porb = planet.porb * planet.time_unit
    e = planet.ecc
    
    # The durations are chosen to always move forward in time.
    anomaly_A = anomaly_A % (360*u.deg)
    anomaly_B = anomaly_B % (360*u.deg)
    if anomaly_A > anomaly_B:
        anomaly_B = anomaly_B + 360*u.deg

    radical = np.sqrt(1+e**2)
    prefactor = porb/(2*np.pi) * (1-e**2)**(3/2)
    anomaly_integral = lambda nu_A, nu_B: prefactor * ((np.arctan(-radical*np.tan(nu_A-(np.pi/2)*u.rad)) - np.arctan(-radical*np.tan(nu_B-(np.pi/2)*u.rad))) / radical).to(u.rad).value
    
    # The anomaly integral is fast but does a sign change at 0 and 180 degrees, so if our anomaly interval contains any, we need to split it up.
    integral_breakpoints = np.array([0, 180]) * u.deg
    breaks = (anomaly_A<integral_breakpoints) & (integral_breakpoints<anomaly_B)
    
    if not np.any(breaks):
        return anomaly_integral(anomaly_A, anomaly_B)
    else:
        boundaries = np.sort(np.r_[u.Quantity([anomaly_A, anomaly_B], u.deg), integral_breakpoints[breaks]])
        return np.sum(np.abs(u.Quantity([anomaly_integral(nu_start, nu_end) for nu_start, nu_end in zip(boundaries[:-1], boundaries[1:])], planet.time_unit)))


def eclipse_durations(system):
    star = system.primary
    planet = system.secondaries[0]

    r = planet.r * planet.length_unit
    R = star.r * star.length_unit
    porb = planet.porb * planet.time_unit
    a = semimajor_axis(system)
    e = planet.ecc
    sin_i = np.sin(planet.inc*planet.angle_unit)
    arg_peri = planet.omega*planet.angle_unit
    
    b_ecl = impact_parameter(system, "eclipse")

    if e == 0:
        circular_integral = lambda radius_ratio: (porb/np.pi * np.arcsin(R/a * (np.sqrt((1+radius_ratio)**2 - b_ecl**2)/sin_i)).to(u.rad).value).to(planet.time_unit)

        total_eclipse_duration = circular_integral(r/R)
        full_eclipse_duration = circular_integral(-r/R)
        ingress_duration = (total_eclipse_duration-full_eclipse_duration)/2
        egress_duration = ingress_duration

        return {"total": total_eclipse_duration,
                "full": full_eclipse_duration,
                "ingress": ingress_duration,
                "egress": egress_duration}

    else:
        projection_anomaly = lambda anomaly, projected_distance: (1-(sin_i*np.sin(anomaly*u.deg+arg_peri))**2)/(1+e*np.cos(anomaly*u.deg))**2 - (projected_distance/(a*(1-e**2)))**2

        if e <= 0.9:
            mideclipse_estimate = ((-np.pi/2)*u.rad - arg_peri).to(u.deg).value
        else:
            mideclipse_estimate = mideclipse_anomaly(system).to(u.deg).value
        estimate_nudge = 5

        anomaly_T1 = anomaly_from_projected_distance(system, R+r, mideclipse_estimate-estimate_nudge)
        anomaly_T2 = anomaly_from_projected_distance(system, R-r, mideclipse_estimate-estimate_nudge)
        anomaly_T3 = anomaly_from_projected_distance(system, R-r, mideclipse_estimate+estimate_nudge)
        anomaly_T4 = anomaly_from_projected_distance(system, R+r, mideclipse_estimate+estimate_nudge)

        total_eclipse_duration = duration_from_anomalies(system, anomaly_T1, anomaly_T4)
        full_eclipse_duration = duration_from_anomalies(system, anomaly_T2, anomaly_T3)
        ingress_duration = duration_from_anomalies(system, anomaly_T1, anomaly_T2)
        egress_duration = duration_from_anomalies(system, anomaly_T3, anomaly_T4)

        return {"total": total_eclipse_duration,
                "full": full_eclipse_duration,
                "ingress": ingress_duration,
                "egress": egress_duration}


def eclipse_intermediate_times(system):
    star = system.primary
    planet = system.secondaries[0]
    
    R = star.r * star.length_unit
    porb = planet.porb * planet.time_unit
    e = planet.ecc
    t_midtransit = planet.t0 * planet.time_unit

    mideclipse_nu = mideclipse_anomaly(system).to(u.deg)
    
    if e == 0:
        t_mideclipse = t_midtransit + porb/2
    else:
        midtransit_nu = mideclipse_nu - 180*u.deg
        t_mideclipse = t_midtransit + duration_from_anomalies(system, midtransit_nu, mideclipse_nu)
    
    estimate_nudge = 5
    midingress_nu = anomaly_from_projected_distance(system, R, mideclipse_nu.value-estimate_nudge)
    midegress_nu = anomaly_from_projected_distance(system, R, mideclipse_nu.value+estimate_nudge)
    t_midingress = t_mideclipse - duration_from_anomalies(system, midingress_nu, mideclipse_nu)
    t_midegress = t_mideclipse + duration_from_anomalies(system, mideclipse_nu, midegress_nu)

    return {"mid-eclipse": t_mideclipse,
            "mid-ingress": t_midingress,
            "mid-egress": t_midegress}


def impact_parameter(system, occultation="eclipse"):
    star = system.primary
    planet = system.secondaries[0]

    R = star.r * star.length_unit
    a = semimajor_axis(system)
    e = planet.ecc
    cos_i = np.cos(planet.inc*planet.angle_unit)
    sin_arg_peri = np.sin(planet.omega*planet.angle_unit)

    if occultation == "eclipse":
        return (a*cos_i / R) * ((1-e**2)/(1-e*sin_arg_peri))
    elif occultation == "transit":
        return (a*cos_i / R) * ((1-e**2)/(1+e*sin_arg_peri))
    else:
        raise ValueError("Occultation must be eclipse or transit.")


def mideclipse_anomaly(system):
    planet = system.secondaries[0]

    e = planet.ecc
    arg_peri = planet.omega*planet.angle_unit
    
    if e == 0:
        return -90*u.deg - arg_peri
    else:
        sin2_i = np.sin(planet.inc*planet.angle_unit)**2
        cos2_i = 1 - sin2_i
        sin_arg_peri = np.sin(arg_peri)
    
        projected_distance = lambda anomaly: cos2_i*((1+e*np.cos(anomaly*u.deg))/(1-e*sin_arg_peri))**2 + sin2_i*np.sin(anomaly*u.deg+arg_peri)**2 - 1

        return fsolve(projected_distance, (-90-arg_peri.value)%360)[0] * u.deg


def minimum_transit_cosi(system):
    star = system.primary

    R = star.r * star.length_unit
    a = semimajor_axis(system)

    return (R/a).decompose().value


def semimajor_axis(system):
    star = system.primary
    planet = system.secondaries[0]

    m = planet.m * planet.mass_unit
    M = star.m * star.mass_unit
    porb = planet.porb * planet.time_unit

    return ((c.G*(m+M) * porb**2) / (4 * np.pi**2))**(1/3)
