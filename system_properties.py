from astropy import units as u

HD189733 = {"name": "HD189733b",
            "primary": {"m": (0.79*u.M_sun).to(u.M_sun).value,
                        "r": (0.75*u.R_sun).to(u.R_sun).value},
            "secondary map": {"inc": (90*u.deg).to(u.deg).value,
                              "obl": (0.0*u.deg).to(u.deg).value,
                              "L": 0.00391},
            "secondary": {"m": (1.13*u.M_jup).to(u.M_sun).value,
                          "r": (1.13*u.R_jup).to(u.R_sun).value,
                          "porb": (2.21857567*u.day).to(u.day).value,
                          #"a": (0.0310*u.au).to(u.au).value,
                          "ecc": 0,
                          "t0": (0*u.day).to(u.day).value,
                          "theta0": (0*u.deg).to(u.deg).value,
                          "prot": (2.21857567*u.day).to(u.day).value,
                          "inc": (85.71*u.deg).to(u.deg).value,
                          "Omega": (0*u.deg).to(u.deg).value,
                          "omega": (0*u.deg).to(u.deg).value}
            }

warm_Jupiter = {"name": "warm_Jupiter",
                "primary": {"m": (1*u.M_sun).to(u.M_sun).value,
                            "r": (1*u.R_sun).to(u.R_sun).value},
                "secondary map": {"inc": (90*u.deg).to(u.deg).value,
                                  "obl": (0.0*u.deg).to(u.deg).value,
                                  "L": 1e-3},
                "secondary": {"m": (1*u.M_jup).to(u.M_sun).value,
                              "r": (1*u.R_jup).to(u.R_sun).value,
                              "porb": (10*u.day).to(u.day).value,
                              "ecc": 0,
                              "t0": (0*u.day).to(u.day).value,
                              "theta0": (0*u.deg).to(u.deg).value,
                              "prot": (10*u.day).to(u.day).value,
                              "inc": (88*u.deg).to(u.deg).value,
                              "Omega": (0*u.deg).to(u.deg).value,
                              "omega": (0*u.deg).to(u.deg).value}
               }
