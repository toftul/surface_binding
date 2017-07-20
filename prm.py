#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 01:23:49 2017

@author: ivan
"""

import numpy as np

# ## Characteristic quantities
# (P ~ 100 mW, r_focuse ~ 1e-6)   # ???? #
E0_charastic = 1e+4  # [V/m] 
# Characteristic time
time_charastic = 1e-4  # [s]
# Particle radius
a_charastic_nm = 15.  # [nm]
# Particle density (for gold)
#density_of_particle = 19320.  # [kg / m^3]
# Particle density (for melanine)
density_of_particle = 1570.  # [kg / m^3]

# Prm of incident wave
wave_length = 350.  # [nm]

# ## Add viscosity for stability
# from the Stokes' law
# F = - gamma v
eta_air = 1.84e-5  # [kg/(m s)]
gamma = 6 * np.pi * eta_air * a_charastic_nm * 1e-9  # [kg / s]

# for polarizability coef prm
# concentration_SI = 1e28  # [1/m^3]
# omega_p_SI = 13.8e15  # [1/s]  | For gold particle
# GAMMA_SI = 1.075e14  # [1/s]   | info is taken from Novotny

# ## Particle material
particle_material = 'silver'

# ## Particle position 
hight = 25  # [nm]
distance_between_particles = 250  # [nm]

hight *= 1e-9  # [nm] -> [m]
distance_between_particles *= 1e-9  # [nm] -> [m]

# ## Interface
interface_material = 'silver'

# ## host medium
epsilon_m = 1.  # air

# ## Modelation prm
# time
tmax = 600.
dt = 1.
# space
dr = 1e-2

# ## Incident wave prm
# \vec{k} \in (y,z)
theta_wave = 0. * np.pi / 180.  # angle between \vec{E} and incedent plane (\vec{k} ^ \vec{n})
phi_wave = 0. * np.pi / 180.  # incedent angle (with normal line)


# KMAX in Gref integration
n_times_k0 = 60  