#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 01:25:58 2017

@author: ivan
"""

# import standart libs
import numpy as np
import cmath
from scipy.integrate import ode, quad
from scipy import interpolate
import matplotlib.pyplot as plt
import time
import progressbar
#import random
import scipy.special as sp
# import my libs
import const
import prm
import extra




# ## Creating variables
# ## TO CHANGE VARIABLES GO TO prm.py file nearby!
E0_charastic = 1.  # [V/m]
time_charastic = 1.  # [s]
a_charastic = 1.  # [m]
a_charastic_nm = 1.  # [nm]
density_of_particle = 1.  # [kg / m^3]
wave_length = 1.  # [nm]
gamma = 1.  # phenomenological parameter
epsilon_interface_default = 1. + 0j
epsilon_particle = 1. + 0j
# epsilon_m = 1.
tmax = 1.
dt = 1.
dr = 1.
# angle between \vec{E} and incedent plane (\vec{k} ^ \vec{n})
theta_wave = 0. * np.pi / 180.
# incedent angle (angle with normal line)
phi_wave = 0. * np.pi / 180.

distance_between_particles = 1
hight = 1. 

interface_material = 'tmp'

n_times_k0 = 1


def setup_blobal_prm():
    global E0_charastic, time_charastic, a_charastic, a_charastic_nm, density_of_particle, \
            wave_length, gamma, tmax, dt, dr, interface_material, \
            theta_wave, phi_wave, epsilon_particle, distance_between_particles, hight, n_times_k0

    E0_charastic = prm.E0_charastic
    time_charastic = prm.time_charastic
   
    a_charastic_nm = prm.a_charastic_nm
    a_charastic = a_charastic_nm * 1e-9
    
    distance_between_particles = prm.distance_between_particles  # [m]
    hight = prm.hight  # [m]
    distance_between_particles /= a_charastic  # [m] -> [1]
    hight /= a_charastic # [m] -> [1]
    
    density_of_particle = prm.density_of_particle
    wave_length = prm.wave_length  # [nm]
    gamma = prm.gamma
    print('gamma = %.1e' % prm.gamma)
    # gamma_dimless = gamma * T_char / m_char
    gamma *= time_charastic / (4 / 3 * np.pi * a_charastic ** 3 * density_of_particle)
    print('gamma_dimless = %.1e' % gamma)
    interface_material = prm.interface_material
    # f_real, f_imag = const.get_eps_function(prm.particle_material)
    # epsilon_particle = f_real(wave_length) + 1j * f_imag(wave_length)
    epsilon_particle = 3.0
    
    tmax = prm.tmax
    dt = prm.dt
    print("dt =", dt)
    dr = prm.dr
    theta_wave = prm.theta_wave
    phi_wave = prm.phi_wave
    
    n_times_k0 = prm.n_times_k0


setup_blobal_prm()


# setting up the epsilon function
if interface_material != 'tmp':
    eps_RE, eps_IM = const.get_eps_function(prm.interface_material)
else:
    def eps_RE(wl):
        return(-15.)
    
    def eps_IM(wl):
        return(.4)

epsilon_interface_default = eps_RE(wave_length) + 1j * eps_IM(wave_length)
# epsilon_interface_default = 3.

# [wl] = [nm]
def epsilon_interface(wl):
    if wl == wave_length:
        return(epsilon_interface_default)
    else:
        return(eps_RE(wl) + 1j * eps_IM(wl))


print('epsilon particle = ', epsilon_particle)
print('epsilon interface = ', epsilon_interface_default)

# Particle mass
m_charastic = 4 / 3 * np.pi * a_charastic ** 3 * density_of_particle  # [kg]
print('m = %.1e' % m_charastic)
# Prm of incident wave
omega0_SI = 2 * np.pi * const.c / (wave_length * 1e-9)  # [1/s]

# [k0] = [1]
def k0(wl):
    return(2 * np.pi * a_charastic_nm / wl)

k0_default = k0(wave_length)


mu0_const_dimless = const.mu0 * 4 * np.pi * const.epsilon0 * a_charastic ** 2 / time_charastic ** 2

print("k0 = ", k0_default, " (dimless)")
print("lambda = ", 2 * np.pi / k0_default, " (dimless)")


# ## Main consts (see notes)
C1_WAVE = time_charastic**2 * 2 * np.pi * const.epsilon0 * \
    a_charastic * E0_charastic**2 / m_charastic

C2_WAVE_default = 4 * np.pi * k0_default ** 2

def C2_WAVE(wl):
    if wl == wave_length:
        return(C2_WAVE_default)
    else:
        return(4 * np.pi * k0(wl) ** 2)


CONST_FOR_GREF_default = (omega0_SI * time_charastic) ** 2 * mu0_const_dimless

def CONST_FOR_GREF(wl):
    if wl == wave_length:
        return(CONST_FOR_GREF_default)
    else:
        return((2 * np.pi * const.c / (wl * 1e-9) * time_charastic) ** 2 * mu0_const_dimless)



print("C1 = ", C1_WAVE)
print("C2 = ", C2_WAVE_default)


# Polarizability coef
# psi
def ricc1_psi(x):
    return(x * sp.spherical_jn(1, x, 0))


# psi'
def ricc1_psi_ch(x):
    return(sp.spherical_jn(1, x, 0) + x * sp.spherical_jn(1, x, 1))


# xi
def ricc1_xi(x):
    return(x * (sp.spherical_jn(1, x, 0) + 1j * sp.spherical_yn(1, x, 0)))

# xi'
def ricc1_xi_ch(x):
    return((sp.spherical_jn(1, x, 0) + 1j * sp.spherical_yn(1, x, 0)) + x * (sp.spherical_jn(1, x, 1) + 1j * sp.spherical_yn(1, x, 1)))


# ## Mie theory
# m = sqrt(epsilon_p) / sqrt(epsilon_m)
# x = k a
def a1(m, x):
    up = m * ricc1_psi(m * x) * ricc1_psi_ch(x) - ricc1_psi(x) * ricc1_psi_ch(m * x)
    bot = m * ricc1_psi(m * x) * ricc1_xi_ch(x) - ricc1_xi(x) * ricc1_psi_ch(m * x)
    return (up / bot)


polarizability_default = 1.5j * a1(cmath.sqrt(epsilon_particle), k0_default) / k0_default ** 3

def polarizability(wl):
    if wl == wave_length:
        return(polarizability_default)
    else:
        return(1.5j * a1(cmath.sqrt(epsilon_particle), k0(wl)) / k0(wl) ** 3)

print("alpha_mie = ", polarizability_default)


# dipoles num
dip_num = 2

dr_vec = np.array([[dr, 0, 0], [0, dr, 0], [0, 0, dr]])
step_num = int(tmax / dt)
time_space = np.linspace(0, tmax, step_num + 1)

# ## Plane wave
# Wave vector
# \vec{k} \in (y,z)
# theta_wave = 0. * np.pi / \
#    180.  # angle between \vec{E} and incedent plane (\vec{k} ^ \vec{n})
phi_wave = prm.phi_wave  # incedent angle

K_wave_default = np.array([0., k0_default * np.sin(phi_wave), - k0_default *
                   np.cos(phi_wave)])
# kz -> -kz
K_wave_ref_default = np.array([0., k0_default * np.sin(phi_wave), k0_default *
                   np.cos(phi_wave)])

def K_wave(wl):
    if wl == wave_length:
        return(K_wave_default)
    else:
        return(np.array([0., k0(wl) * np.sin(phi_wave), - k0(wl) *
                   np.cos(phi_wave)]))

def K_wave_ref(wl):
    if wl == wave_length:
        return(K_wave_ref_default)
    else:
        return(np.array([0., k0(wl) * np.sin(phi_wave), k0(wl) * np.cos(phi_wave)]))
    
    
# perpendicular part ( = \sqrt(kx^2 + ky^2) )
k0_perpendicular_default = np.sqrt(K_wave_default[0]**2 + K_wave_default[1]**2)

def k0_perpendicular(wl):
    if wl == wave_length:
        return(k0_perpendicular_default)
    else:
        return(np.sqrt(K_wave(wl)[0]**2 + K_wave(wl)[1]**2))


# Electric field
# only p-wave
E0 = np.array([0., np.cos(phi_wave), np.sin(phi_wave)], dtype=complex)


# array of dipoles
dip_r = np.zeros([dip_num, 3])
dip_r[0] = np.array([0., 0., hight])
dip_r[1] = np.array([0., distance_between_particles, hight])
print('r1 = {%.1f, %.1f, %.1f}'%(dip_r[0, 0], dip_r[0, 1], dip_r[0, 2]))
print('r2 = {%.1f, %.1f, %.1f}'%(dip_r[1, 0], dip_r[1, 1], dip_r[1, 2]))
dip_v = np.zeros([dip_num, 3])
dip_mu = np.zeros([dip_num, 3], dtype=complex)
dip_mass = np.ones(dip_num) * m_charastic


# extra consts
# sys_vector_len = 0


# Fresnel reflection coefficents
def kkzz11(k_II, wl):
    if wl == wave_length:
        return(cmath.sqrt(k0_default ** 2 - k_II ** 2))
    else:
        return(cmath.sqrt(k0(wl) ** 2 - k_II ** 2))

k0_tr_default = k0_default * cmath.sqrt(epsilon_interface_default)

def k0_tr(wl):
    return(k0(wl) * cmath.sqrt(epsilon_interface(wl)))


def kkzz22(k_II, wl):
    if wl == wave_length:
        return(cmath.sqrt(k0_tr_default ** 2 - k_II ** 2))
    else:
        return(cmath.sqrt(k0_tr(wl) ** 2 - k_II ** 2))


def rs(k_II, wl):
    return((kkzz11(k_II, wl) - kkzz22(k_II, wl)) /
           (kkzz11(k_II, wl) + kkzz22(k_II, wl)))


# epsilon of media = 1
def rp(k_II, wl):
    return((epsilon_interface(wl) * kkzz11(k_II, wl) - kkzz22(k_II, wl)) /
           (epsilon_interface(wl) * kkzz11(k_II, wl) + kkzz22(k_II, wl)))


# Incedent wave
def inc_efield(r, t, wl):
    return E0 * np.exp(1j * np.dot(K_wave(wl), r))

# E0_ref = Ep0 * rp(k0_perpendicular) + Es0 * rs(k0_perpendicular)
# Reflected wave
def E0_ref(wl):
    E0_ref_tmp = E0 * rp(K_wave(wl)[1], wl)
    E0_ref_tmp[1] *= - 1
    return(E0_ref_tmp)



def ref_efield(r, t, wl):
    return(E0_ref(wl) * np.exp(1j * np.dot(K_wave_ref(wl), r)))


# External field
def ex_efield(r, t, wl):
    return(inc_efield(r, t, wl) + ref_efield(r, t, wl))


# Dimentionless Green func
def green_func(r1, r2, wl):
    if np.linalg.norm(r1 - r2) != 0:
        if wl == wave_length:
            R = r1 - r2
            Rmod = np.linalg.norm(R)
            kR = k0_default * Rmod
            EXPikR4piR = np.exp(1j * kR) / (4. * np.pi * Rmod)
            CONST1 = (1 + (1j * kR - 1) / kR**2)
            CONST2 = (3 - 3j * kR - kR**2) / (kR**2 * Rmod**2)
            # return Green function
            result = EXPikR4piR * (CONST1 * np.identity(3) +
                                   CONST2 * np.outer(R, R))
        else:
            R = r1 - r2
            Rmod = np.linalg.norm(R)
            kR = k0(wl) * Rmod
            EXPikR4piR = np.exp(1j * kR) / (4. * np.pi * Rmod)
            CONST1 = (1 + (1j * kR - 1) / kR**2)
            CONST2 = (3 - 3j * kR - kR**2) / (kR**2 * Rmod**2)
            # return Green function
            result = EXPikR4piR * (CONST1 * np.identity(3) +
                                   CONST2 * np.outer(R, R))
    else:
        print("WHAT THE HELL??? WHY I'M GETTING SAME RADIUS VECTOR?!?!")
        result = 0.

    return result


# Gref for r = r0
# xx = yy
def GFref_xx_rr0(z, wl):
    Gxx = 0.125j / np.pi
    if wl == wave_length:
        def f_real(krho):
            kz1 = kkzz11(krho, wl)
            ff = cmath.exp(1j * kz1 * z) * krho / kz1 * (rs(krho, wl) - rp(krho, wl) * (kz1 / k0_default)**2)
            return(ff.real)
    
        def f_imag(krho):
            kz1 = kkzz11(krho, wl)
            ff = cmath.exp(1j * kz1 * z) * krho / kz1 * (rs(krho, wl) - rp(krho, wl) * (kz1 / k0_default)**2)
            return(ff.imag)
    
        KMAX = max(1.1 * k0_default, n_times_k0 * k0_default / z)
        Gxx *= quad(f_real, 0, KMAX, points=[k0_default])[0] + 1j * quad(f_imag, 0, KMAX, points=[k0_default])[0]
    else:
        def f_real(krho):
            kz1 = kkzz11(krho, wl)
            ff = cmath.exp(1j * kz1 * z) * krho / kz1 * (rs(krho, wl) - rp(krho, wl) * (kz1 / k0(wl))**2)
            return(ff.real)
    
        def f_imag(krho):
            kz1 = kkzz11(krho, wl)
            ff = cmath.exp(1j * kz1 * z) * krho / kz1 * (rs(krho, wl) - rp(krho, wl) * (kz1 / k0(wl))**2)
            return(ff.imag)
    
        KMAX = max(1.1 * k0(wl), n_times_k0 * k0(wl) / z)
        Gxx *= quad(f_real, 0, KMAX, points=[k0(wl)])[0] + 1j * quad(f_imag, 0, KMAX, points=[k0(wl)])[0]
    return Gxx


# zz
def GFref_zz_rr0(z, wl):        
    k0_local = k0(wl)
    Gzz = 1j / (4. * np.pi * k0_local ** 2)

    def f_real(krho):
        kz1 = kkzz11(krho, wl)
        ff = cmath.exp(1j * kz1 * z) * krho ** 3 * rp(krho, wl) / kz1
        return(ff.real)

    def f_imag(krho):
        kz1 = kkzz11(krho, wl)
        ff = cmath.exp(1j * kz1 * z) * krho ** 3 * rp(krho, wl) / kz1
        return(ff.imag)

    KMAX = max(1.1 * k0_local, n_times_k0 * k0_local / z)

    Gzz *= quad(f_real, 0, KMAX, points=[k0_local])[0] + 1j * quad(f_imag, 0, KMAX, points=[k0_local])[0]
    
    return Gzz


# ## Gref function for any r, r0
def g_func_ss(x, phi):
    return(2 * np.pi * (sp.j0(x) * np.sin(phi)**2 + sp.j1(x) / x * np.cos(2 * phi)))

def g_func_cc(x, phi):
    return(2 * np.pi * (sp.j0(x) * np.cos(phi)**2 - sp.j1(x) / x * np.cos(2 * phi)))

def g_func_sc(x, phi):
    return(np.pi * np.sin(2 * phi) * (2 * sp.j1(x) / x - sp.j0(x)))

def g_func_s(x, phi):
    return(2j * np.pi * np.sin(phi) * sp.j1(x))

def g_func_c(x, phi):
    return(2j * np.pi * np.cos(phi) * sp.j1(x))


def m_element_xx(k, rho, phi, wl, k0_local):
    kz1 = kkzz11(k, wl)
    return(k / kz1 * (rs(k, wl) * g_func_ss(k * rho, phi) - rp(k, wl) * g_func_cc(k * rho, phi) * (kz1 / k0_local)**2))

def m_element_yy(k, rho, phi, wl, k0_local):
    kz1 = kkzz11(k, wl)
    return(k / kz1 * (rs(k, wl) * g_func_cc(k * rho, phi) - rp(k, wl) * g_func_ss(k * rho, phi) * (kz1 / k0_local)**2))

def m_element_zz(k, rho, phi, wl, k0_local):
    return(2 * np.pi * rp(k, wl) * k**3 / (k0_local**2 * kkzz11(k, wl)) * sp.j0(k * rho))

def m_element_xy(k, rho, phi, wl, k0_local):
    kz1 = kkzz11(k, wl)
    return(g_func_sc(k * rho, phi) * k / kz1 * (rs(k, wl) + rp(k, wl) * (kz1 / k0_local)**2))

def m_element_xz(k, rho, phi, wl, k0_local):
    return(- rp(k, wl) * g_func_c(k * rho, phi) * (k / k0_local)**2)

def m_element_yz(k, rho, phi, wl, k0_local):
    return(- rp(k, wl) * g_func_s(k * rho, phi) * (k / k0_local)**2)


# ## Gref function for any r, r0
# r0 -- dipole location
# r  -- view point
def GFref(r, r0, wl):
    k0_local = k0(wl)
    zz = r[2] + r0[2]
    rho, phi0 = cmath.polar((r[0] - r0[0]) + 1j * (r[1] - r0[1]))

    G = np.zeros([3, 3], dtype=complex)
        
    if rho == 0:
        G[0, 0] = GFref_xx_rr0(zz, wl)
        G[1, 1] = G[0, 0] 
        G[2, 2] = GFref_zz_rr0(zz, wl)
    else:
        def calculate_int_for(element_function):
            def f_real(k):
                integrand = cmath.exp(1j * kkzz11(k, wl) * zz) * element_function(k, rho, phi0, wl, k0_local)
                return(integrand.real)

            def f_imag(k):
                integrand = cmath.exp(1j * kkzz11(k, wl) * zz) * element_function(k, rho, phi0, wl, k0_local)
                return(integrand.imag)

            KMAX = max(1.1 * k0_local, n_times_k0 * k0_local / zz)
            KMIN = k0_local * 0.01
            return(quad(f_real, KMIN, KMAX, points=[k0_local])[0] + 1j * quad(f_imag, KMIN, KMAX, points=[k0_local])[0])

        G[0, 0] = calculate_int_for(m_element_xx)
        G[0, 1] = calculate_int_for(m_element_xy)
        G[0, 2] = calculate_int_for(m_element_xz)
        G[1, 1] = calculate_int_for(m_element_yy)
        G[1, 2] = calculate_int_for(m_element_yz)
        G[2, 2] = calculate_int_for(m_element_zz)
        G[1, 0] = G[0, 1]
        G[2, 0] = - G[0, 2]
        G[2, 1] = - G[1, 2]

        G *= 0.125j / np.pi**2

    return G



# reflected electric field from interface
def dipole_efield_ref(r, t, n, wl):
    G = GFref(r, dip_r[n], wl)
    E = C2_WAVE(wl) * np.dot(G, dip_mu[n])
    return E

# scattered field from dipole
def dipole_efield(r, t, n, wl):
    G = green_func(r, dip_r[n], wl)
    E = C2_WAVE(wl) * np.dot(G, dip_mu[n])
    return E

# sum of e-field for n-th dipole
# NOTE: this done for grad E
def field_sum(r, t, n, wl):
    # for nabla E
    E_sum_plus, E_sum_minus = np.zeros(
        [3, 3], dtype=complex), np.zeros([3, 3], dtype=complex)

    E_sum_plus[0] = ex_efield(r + dr_vec[0], t, wl)
    E_sum_plus[1] = ex_efield(r + dr_vec[1], t, wl)
    E_sum_plus[2] = ex_efield(r + dr_vec[2], t, wl)
    E_sum_minus[0] = ex_efield(r - dr_vec[0], t, wl)
    E_sum_minus[1] = ex_efield(r - dr_vec[1], t, wl)
    E_sum_minus[2] = ex_efield(r - dr_vec[2], t, wl)

    # sum of reflected fields
    for i in range(dip_num):     
        E_sum_plus[0] += dipole_efield_ref(r + dr_vec[0], t, i, wl)
        E_sum_plus[1] += dipole_efield_ref(r + dr_vec[1], t, i, wl)
        E_sum_plus[2] += dipole_efield_ref(r + dr_vec[2], t, i, wl)
        E_sum_minus[0] += dipole_efield_ref(r - dr_vec[0], t, i, wl)
        E_sum_minus[1] += dipole_efield_ref(r - dr_vec[1], t, i, wl)
        E_sum_minus[2] += dipole_efield_ref(r - dr_vec[2], t, i, wl)

    # sum of scattered fields
    for i in range(dip_num):
        if i != n:        
            E_sum_plus[0] += dipole_efield(r + dr_vec[0], t, i, wl)
            E_sum_plus[1] += dipole_efield(r + dr_vec[1], t, i, wl)
            E_sum_plus[2] += dipole_efield(r + dr_vec[2], t, i, wl)
            E_sum_minus[0] += dipole_efield(r - dr_vec[0], t, i, wl)
            E_sum_minus[1] += dipole_efield(r - dr_vec[1], t, i, wl)
            E_sum_minus[2] += dipole_efield(r - dr_vec[2], t, i, wl)

    # result is a MATRIX!
    return E_sum_plus, E_sum_minus


# may the Force be with you
# F = 1/2 Re{ mu*_i nabla E_i }
# (acceleration in fact)
def force_sum(r, t, n, wl = wave_length):
    # fields
    dEplus, dEminus = field_sum(r, t, n, wl)

    F_average = np.zeros(3)

    temp_const = np.zeros(3, dtype=complex)
    for alpha in range(3):
        for beta in range(3):
            temp_const[alpha] += np.conjugate(dip_mu[n, beta]) * (
                dEplus[alpha, beta] - dEminus[alpha, beta]) * 0.5 / dr
    F_average += np.real(temp_const)
    F_average *= C1_WAVE
    # motion in z axis is fixed 
    F_average[2] = 0.
    

    return F_average


# Specific functions
#   system_vector = {x1, v1, x2, v2, ... }
# d_system_vector = {v1, f1, v2, f2, ... }
def dsystem(t, system_vector):
    df = np.zeros([dip_num, 3])
    df[0] = force_sum(np.array([system_vector[0],
                                system_vector[1],
                                system_vector[2]]), t, 0)
    d_system_vector = np.array([system_vector[3], system_vector[4], system_vector[5],
                                df[0, 0] - gamma * system_vector[3],
                                df[0, 1] - gamma * system_vector[4],
                                df[0, 2] - gamma * system_vector[5]])
    for i in range(1, dip_num):
        df[i] = force_sum(np.array([system_vector[i * 6],
                                    system_vector[i * 6 + 1],
                                    system_vector[i * 6 + 2]]), t, i)
        d_system_vector = np.hstack((d_system_vector,
                                     np.array([system_vector[i * 6 + 3],
                                               system_vector[i * 6 + 4],
                                               system_vector[i * 6 + 5],
                                               df[i, 0] - gamma *
                                               system_vector[i * 6 + 3],
                                               df[i, 1] - gamma *
                                               system_vector[i * 6 + 4],
                                               df[i, 2] - gamma * system_vector[i * 6 + 5]])))

    return d_system_vector


def mu_single_solver(dipole_num, wl = wave_length):
    # recalc alpha
    alpha_eff = np.zeros(3, dtype=complex)
    alpha_eff[0] = polarizability(wl) / (1 - polarizability(wl) * CONST_FOR_GREF(wl) * GFref_xx_rr0(2 * dip_r[dipole_num, 2], wl))
    alpha_eff[1] = alpha_eff[0]
    alpha_eff[2] = polarizability(wl) / (1 - polarizability(wl) * CONST_FOR_GREF(wl) * GFref_zz_rr0(2 * dip_r[dipole_num, 2], wl))

    return(alpha_eff * ex_efield(dip_r[dipole_num], 0, wl))

# WARNING: only for 2 particles
def mu_solver(wl = wave_length):
    GREEN_M = np.zeros([dip_num, dip_num, 3, 3], dtype=complex)
    for i in range(dip_num):
        for j in range(dip_num):
            if j > i:
                GREEN_M[i, j] = green_func(dip_r[i], dip_r[j], wl)
                
    I = np.identity(3) / (C2_WAVE(wl) * polarizability(wl))
    A = np.bmat([[I - GFref(dip_r[0], dip_r[0], wl), -GREEN_M[0, 1] - GFref(dip_r[0], dip_r[1], wl)], 
                 [-GREEN_M[0, 1] - GFref(dip_r[1], dip_r[0], wl), I - GFref(dip_r[1], dip_r[1], wl)]])
    b = np.hstack([ex_efield(dip_r[0], 0., wl) / C2_WAVE(wl), 
    			   ex_efield(dip_r[1], 0., wl) / C2_WAVE(wl)])

    #        |    I - Gref_00         -(Gref_01 + G0_01) |
    # A[i] = |                                           |
    #        | -(Gref_10 + G0_10)         I - Gref_11    |
    
    # solving matrix equation
    x = np.linalg.solve(A, b)
    # fill results
    for i in range(dip_num):
        dip_mu[i] = np.array([x[3* i], x[3 * i + 1], x[3 * i + 2]], dtype=complex)


def do_experiment():
    # vector of initial values
    y0 = np.hstack([dip_r[0], dip_v[0]])
    for i in range(1, dip_num):
        y0 = np.hstack([y0, np.hstack([dip_r[i], dip_v[i]])])
    
    t0 = 0.
    r = ode(dsystem).set_integrator('dopri5')
    r.set_initial_value(y0, t0)
    
    # sys_vector_len = 6 * dip_num
    
    solution = np.zeros([dip_num, step_num + 1, 3])
    
    start_time = time.time()
    step = 0
    for t in time_space:
        if (dip_r[0, 2] < 1) or (dip_r[1, 2] < 1):
            break
        # ## step forward
        y0 = r.integrate(r.t + dt)
        for i in range(dip_num):
            # write a solution array
            solution[i, step] = np.array([y0[6 * i], y0[6 * i + 1], y0[6 * i + 2]])
            # renew dipole var
            dip_r[i] = np.array([y0[6 * i], y0[6 * i + 1], y0[6 * i + 2]])
        print('y0, z0 = ', y0[1], ', ', y0[2])
        print('y1, z1 = ', y0[7], ', ', y0[8])
        print('d = %.2f' % np.sqrt((y0[1] - y0[7])**2))
        # ## recalc dipole moment 
        mu_solver()
        # print('mu0 = (%.1e, %.1e, %.1e)' % (dip_mu[0, 0], dip_mu[0, 1], dip_mu[0, 2]))
        # print('mu1 = (%.1e, %.1e, %.1e)' % (dip_mu[1, 0], dip_mu[1, 1], dip_mu[1, 2]))
        # ## print progress
        # if step % 5 == 0:
        print("step = ", step)
        # ## change step counter
        step += 1
    
    time_spent = time.time() - start_time
    print("execution time: %s [s]" % time_spent)
    print("seconds per iteration: %s" % (time_spent / step))
    
    return(solution)


###################################################
########## PLOT FORCE #############################
###################################################

def calc_force_for_plot(y_space):
    Fy_space = np.zeros(len(y_space))
    
    F0 = 0.5 * (k0(wave_length) / a_charastic_nm * 1e9) * polarizability(wave_length).imag * 4 * np.pi * const.epsilon0 * (a_charastic_nm * 1e-9)**3 * E0_charastic**2
    print('F0 = %.1e' % F0)
    i = 0
    bar = progressbar.ProgressBar()
    for y in bar(y_space):
    #for y in y_space:
        dip_r[1, 1] = y / a_charastic_nm 
        mu_solver()
        #dip_mu[0] = mu_single_solver(0)
        #dip_mu[1] = mu_single_solver(1)
        Fy_space[i] = force_sum(dip_r[0], 0., 0)[1] * m_charastic*a_charastic/time_charastic**2 / F0
        i += 1
        
    return(Fy_space)


#y_space_plot = np.linspace(50, 800, 30)
#Fy_space = calc_force_for_plot(y_space_plot)
#
## y_space_plot_nonSPP = np.linspace(50, 800, 60)
## Fy_space_nonSPP = calc_force_for_plot(y_space_plot_nonSPP)
#
#dat_m = np.genfromtxt('data.csv')
#dat_m_nonSPP = np.genfromtxt('data_nonSPP.csv')
#
#plt.rcParams.update({'font.size': 14})
#plt.figure(figsize=(8.24, 3.56))
#plt.plot(y_space_plot, - Fy_space, label='current')
#plt.plot(dat_m[:, 0], dat_m[:, 1], label='article data')
#plt.plot(dat_m_nonSPP[:, 0], dat_m_nonSPP[:, 1], label='article data nonSPP x10')
##plt.plot(y_space_plot_nonSPP, - 10 * Fy_space_nonSPP, label='nonSPP x10')
#plt.legend()
#plt.ylim(-6, 8)
#plt.xlim(0, np.max(y_space_plot))
#plt.title(r'Optical Force, $\lambda = %.0f\ nm$, $z = %.1f \ nm$' % (wave_length, dip_r[1, 2] * 15))
#plt.xlabel(r'$y, nm$')
#plt.ylabel(r'$F_x/F_0$')
#plt.grid()
#
#plt.show()



###################################################
########## MIDULATE WITH INTERPOLATED FORCE #######
###################################################

def calc_force_space(x_space, y_space):
    F_space = np.zeros([len(x_space), len(y_space), 3])
    
    i = 0
    bar = progressbar.ProgressBar()
    for x in bar(x_space):
        dip_r[1, 0] = x / a_charastic_nm
        j = 0
        for y in y_space:
            dip_r[1, 1] = y / a_charastic_nm
            if np.linalg.norm(dip_r[0] - dip_r[1]) < (40 / a_charastic_nm):
                F_space[i, j] = np.array([0., 0., 0.])
            else:
                mu_solver()
                F_space[i, j] = force_sum(dip_r[0], 0., 0)
            j += 1
        i += 1 
    
    return(F_space)


def create_force_function(x_space, y_space, F_space, method='linear'):
    FX = F_space[:,:,0]
    FY = F_space[:,:,1]
    FZ = F_space[:,:,2]
    # надо бы пофиксить, чтобы не делать transpose
    fx = interpolate.interp2d(x_space / a_charastic_nm, y_space / a_charastic_nm, FX.transpose(), kind=method)
    fy = interpolate.interp2d(x_space / a_charastic_nm, y_space / a_charastic_nm, FY.transpose(), kind=method)
    fz = interpolate.interp2d(x_space / a_charastic_nm, y_space / a_charastic_nm, FZ.transpose(), kind=method)
    
    def f_vec(x, y):
        return(np.array([fx(x, y), fy(x, y), fz(x, y)]))
    
    return(f_vec)


# calc force space
#x_space = np.linspace(-300, 300, 40)
#y_space = np.linspace(-400, 400, 40)
#dip_r[0] *= 0
#dip_r[0, 2] = dip_r[1, 2]
#F_space = calc_force_space(x_space, y_space)
## save results
#np.save('F_space_wide.npy', F_space)
#np.save('x_space_wide.npy', x_space)
#np.save('y_space_wide.npy', y_space)


# Loading np arrays
x_space = np.load('x_space_wide.npy')
y_space = np.load('y_space_wide.npy')
# force calculated for E0 = 1e4 [V/m]
F_space = np.load('F_space_wide.npy')

# Default vaules (F_space was calculated with such prm)
E0_charastic_def = 1e+4  # [V/m]  
time_charastic_def = 1e-4  # [s]
# Scaling F \sim  T^2 E0^2
F_space *= (E0_charastic / E0_charastic_def)**2 * (time_charastic / time_charastic_def)**2

force_interpolated = create_force_function(x_space, y_space, F_space, 'linear')

#checking by plotting
#y_space_for_plot = np.linspace(-0, 400, 100)
#fy_interpolated = np.zeros(len(y_space_for_plot))
#i = 0
#for yy in y_space_for_plot:
#    fy_interpolated[i] = force_interpolated(0, yy/a_charastic_nm)[1]
#    i += 1
#    
#plt.plot(y_space_for_plot, -fy_interpolated)



# Specific functions
#   system_vector = {x1, v1, x2, v2, ... }
# d_system_vector = {v1, f1, v2, f2, ... }
def dsystem_interpolated(t, system_vector):
    df = np.zeros([dip_num, 3])
    # ВОЗМОЖНО ДРУГОЙ ЗНАК, ПРОВЕРИТЬ! r0 - r1 или r1 - r0
    x = system_vector[6] - system_vector[0]
    y = system_vector[7] - system_vector[1]
    # но вроде так
    # if стоит чтобы не крашилось при больших аргументах (x,y)
    if (x > -250 / a_charastic_nm) and (x < 250 / a_charastic_nm) and (y > 0) and (y < 400 / a_charastic_nm):
        df[0] = force_interpolated(system_vector[6] - system_vector[0], 
                               system_vector[7] - system_vector[1]).transpose()
    else:
        df[0] = force_interpolated(0, 0).transpose() * 0.
    d_system_vector = np.array([system_vector[3], system_vector[4], system_vector[5],
                                df[0, 0] - gamma * system_vector[3],
                                df[0, 1] - gamma * system_vector[4],
                                df[0, 2] - gamma * system_vector[5]])
    
    # F10 = - F01
    df[1] = - df[0]
    d_system_vector = np.hstack((d_system_vector,
                                 np.array([system_vector[6 + 3],
                                           system_vector[6 + 4],
                                           system_vector[6 + 5],
                                           df[1, 0] - gamma *
                                           system_vector[6 + 3],
                                           df[1, 1] - gamma *
                                           system_vector[6 + 4],
                                           df[1, 2] - gamma * system_vector[6 + 5]])))

    return d_system_vector


def do_experiment_inter(dt_local, tmax_local, r1_rand):
    dip_r[1] = r1_rand
    step_num = int(tmax_local / dt_local)
    time_space_local = np.linspace(0, tmax_local, step_num + 1)
    # vector of initial values
    y0 = np.hstack([dip_r[0], dip_v[0]])
    for i in range(1, dip_num):
        y0 = np.hstack([y0, np.hstack([dip_r[i], dip_v[i]])])
    
    t0 = 0.
    r = ode(dsystem_interpolated).set_integrator('dopri5')
    r.set_initial_value(y0, t0)
    
    # sys_vector_len = 6 * dip_num
    
    solution = np.zeros([dip_num, step_num + 1, 3])
    
    start_time = time.time()
    step = 0
    bar = progressbar.ProgressBar()
    for t in bar(time_space_local):
        if (dip_r[0, 2] < 1) or (dip_r[1, 2] < 1):
            break
        # ## step forward
        y0 = r.integrate(r.t + dt)
        for i in range(dip_num):
            # write a solution array
            solution[i, step] = np.array([y0[6 * i], y0[6 * i + 1], y0[6 * i + 2]])
            # renew dipole var
            dip_r[i] = np.array([y0[6 * i], y0[6 * i + 1], y0[6 * i + 2]])
       
        if np.linalg.norm(dip_r[0] - dip_r[1]) > 424 / a_charastic_nm:
            for i in range(dip_num):
                # write a solution array
                solution[i, step] = np.array([np.nan, np.nan, np.nan])
        
        step += 1
        
    
    time_spent = time.time() - start_time
    print("execution time: %s [s]" % time_spent)
    print("seconds per iteration: %s" % (time_spent / step))
    
    return(time_space_local, solution)


###############################
# mapping green zone
XMIN = -300
XMAX = 300
YMIN = -300
YMAX = 300
###############################
# type 1 -- pulling zone
def n(r0, r):
    dr = r0 - r
    norm = np.linalg.norm(dr)
    if norm == 0:
        return(dr)
    return(dr / norm)

x_gr = np.linspace(YMIN, YMAX, 200) / a_charastic_nm
y_gr = np.linspace(XMIN, XMAX, 200) / a_charastic_nm
green_zone = np.zeros([len(x_gr), len(y_gr)])
r_eq = np.array([0, 161, prm.hight]) / a_charastic_nm
r_eq_left = np.array([0, -161, prm.hight]) / a_charastic_nm

i = 0
for y in y_gr:
    j = 0
    for x in x_gr:
        r = np.array([x, y, prm.hight / a_charastic_nm])
        if y > 0:
            nn = n(r_eq, r)
        else:
            nn = n(r_eq_left, r)
        ff = - force_interpolated(x, y).transpose()[0]
        green_zone[i, j] = np.dot(nn, ff)
        if np.linalg.norm([x, y]) < 65/a_charastic_nm:
            green_zone[i, j] *= 0
        j += 1
    i += 1


green_zone /= np.max(green_zone)

# plotting
#plt.rcParams.update({'font.size': 14})
#plt.figure(figsize=(8.24, 5.24))
#cmap = plt.cm.seismic
#mid = 1 - np.max(green_zone)/(np.max(green_zone) + np.abs(np.min(green_zone)))
#shifted_cmap = extra.shiftedColorMap(cmap, midpoint=mid, name='shifted')
#plt.contourf(y_gr * a_charastic_nm, x_gr * a_charastic_nm, green_zone.transpose(), 100, cmap=shifted_cmap)
#plt.colorbar()
#plt.title('Stable zone')
#plt.xlabel(r'$x,\ nm$')
#plt.ylabel(r'$y,\ nm$')
#plt.savefig('fig/green_zone.svg', format='svg', dpi=1200)
#plt.savefig('fig/green_zone.eps', format='eps', dpi=1200)
#plt.show()

###############################
# type 2 -- color vector field
x_gr2 = np.linspace(YMIN, YMAX, 20) / a_charastic_nm
y_gr2 = np.linspace(XMIN, XMAX, 20) / a_charastic_nm
FX = np.zeros([len(x_gr2), len(y_gr2)])
FY = np.zeros([len(x_gr2), len(y_gr2)])
MAG = np.zeros([len(x_gr2), len(y_gr2)])

i = 0
for y in y_gr2:
    j = 0
    for x in x_gr2:
        ff = - force_interpolated(x, y).transpose()[0]
        FX[i, j] = ff[0]
        FY[i, j] = ff[1]
        MAG[i, j] = np.linalg.norm([ff[0], ff[1]])
        
        if np.linalg.norm([x, y]) < 70/a_charastic_nm:
            MAG[i, j] *= 0
        j += 1
    i += 1

MAG /= np.max(MAG)
FX /= MAG
FY /= MAG

#plotting
#plt.rcParams.update({'font.size': 14})
##plt.title('Force map')
#plt.figure(figsize=(8.24, 5.24))
#plt.quiver(y_gr2 * a_charastic_nm, x_gr2 * a_charastic_nm, 
#           FY.transpose(), FX.transpose(), MAG.transpose(), cmap="Greys", width=0.004)
#plt.colorbar()
#plt.xlabel(r'$x,\ nm$')
#plt.ylabel(r'$y,\ nm$')
#plt.savefig('fig/vector_field.svg', format='svg', dpi=1200)
#plt.savefig('fig/vector_field.eps', format='eps', dpi=1200)
#plt.show()


## myrand \in [-1,1]
#def myrand():
#    return((random.random() - 0.5) * 2)
#    
## rand initial positon
#disturtion = 150  # [nm]
#r1_rand = np.array([disturtion * myrand(), 187 + disturtion * myrand(), dip_r[1, 2] * a_charastic_nm]) / a_charastic_nm
#
## calculation
#dip_r[0] *= 0
#dip_r[0, 2] = dip_r[1, 2]
#time_space_1, sol = do_experiment_inter(dt, 25*tmax, r1_rand)
#np.save('solution.npy', sol)
#np.save('time_space_1.npy', time_space_1)


#plt.plot(time_space_1 * time_charastic * 1000, sol[0, :, 1] * a_charastic_nm)
#plt.plot(time_space_1 * time_charastic * 1000, sol[1, :, 1] * a_charastic_nm)
#plt.title('Dynamics')
#plt.xlabel(r'$t,\ ms$')
#plt.ylabel(r'$r_y,\ nm$')
#plt.grid()
#plt.show()


#fig, ax = plt.subplots()
#plt.rcParams.update({'font.size': 14})
#plt.figure(figsize=(8.24, 5.24))
##plt.plot(sol[0, :, 1] * a_charastic_nm * 0, sol[0, :, 0] * a_charastic_nm * 0)
##plt.plot((sol[1, :, 1] - sol[0, :, 1]) * a_charastic_nm, (sol[1, :, 0] - sol[0, :, 0]) * a_charastic_nm)
#plt.xlabel(r'$x,\ nm$')
#plt.ylabel(r'$y,\ nm$')
#plt.ylim(-200, 200)
#plt.xlim(0, 310)
##plt.grid()
##lc = extra.colorline(...)
#lc = extra.colorline((sol[1, :, 1] - sol[0, :, 1]) * a_charastic_nm, 
#               (sol[1, :, 0] - sol[0, :, 0]) * a_charastic_nm,
#               time_space_1*time_charastic*1000, cmap='jet', linewidth=4,
#               norm=plt.Normalize(0.0, np.max(time_space_1)*time_charastic*1000))
#plt.colorbar(lc)
#plt.savefig('fig/line.svg', format='svg', dpi=1200)
#plt.savefig('fig/line.eps', format='eps', dpi=1200)
#plt.show()

# np.set_printoptions(precision=2)


##################################################
# PLOT EVERYTHING IN ONE PLOT
# WORKS ONLY WHEN EVERYTHING IS CALCULATED
##################################################
r1_1 = np.array([-210, 180, dip_r[1, 2] * a_charastic_nm]) / a_charastic_nm
r1_2 = np.array([-140, 30, dip_r[1, 2] * a_charastic_nm]) / a_charastic_nm
gamma = 0.014

# calculation
dip_r[0] *= 0
dip_r[0, 2] = dip_r[1, 2]
time_space_1, sol = do_experiment_inter(dt, tmax, r1_1)
dip_r[0] *= 0
dip_r[0, 2] = dip_r[1, 2]
time_space_1, sol2 = do_experiment_inter(dt, tmax, r1_2)
sol2 = -sol2 


plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8.24, 5.24))
plt.ylim(YMIN, YMAX)
plt.xlim(XMIN, XMAX)
plt.xlabel(r'$x,\ nm$')
plt.ylabel(r'$y,\ nm$')
#plt.title('Stable zone')
#plt.grid()
# COLORED LINE
lc = extra.colorline((sol[1, :, 1] - sol[0, :, 1]) * a_charastic_nm, 
               (sol[1, :, 0] - sol[0, :, 0]) * a_charastic_nm,
               time_space_1*time_charastic*1000, cmap='jet', linewidth=3.5,
               norm=plt.Normalize(0.0, np.max(time_space_1)*time_charastic*1000))
lc = extra.colorline((sol2[1, :, 1] - sol2[0, :, 1]) * a_charastic_nm, 
               (sol2[1, :, 0] - sol2[0, :, 0]) * a_charastic_nm,
               time_space_1*time_charastic*1000, cmap='jet', linewidth=3.5,
               norm=plt.Normalize(0.0, np.max(time_space_1)*time_charastic*1000))
#plt.colorbar(lc)
clb = plt.colorbar(lc)
clb.ax.set_title('time, ms')
# GREEN ZONE
#cmap = plt.cm.PRGn
#cmap = plt.cm.RdBu
cmap = plt.cm.seismic
#cmap = plt.cm.rainbow
#cmap = plt.cm.gist_earth
#cmap = plt.cm.jet

mid = 1 - np.max(green_zone)/(np.max(green_zone) + np.abs(np.min(green_zone)))
shifted_cmap = extra.shiftedColorMap(cmap, midpoint=mid, name='shifted')
plt.contourf(y_gr * a_charastic_nm, x_gr * a_charastic_nm, green_zone.transpose(), 150, 
             cmap=shifted_cmap, alpha=0.4)
#plt.colorbar()
clb = plt.colorbar(ticks=[0])
clb.ax.set_title(r'$\bf{F}\cdot \bf{n}^{eq}$')
# VECTOR FIELD
plt.quiver(y_gr2 * a_charastic_nm, x_gr2 * a_charastic_nm, 
           65*FY.transpose(), 65*FX.transpose(), MAG.transpose(), cmap="Greys", 
           scale=5, scale_units='inches', width=0.008, headwidth=3, headlength=3, headaxislength=3)
#plt.colorbar()
plt.savefig('fig/dynamics.svg', format='svg', dpi=1200)
plt.savefig('fig/dynamics.eps', format='eps', dpi=1200)
plt.show()




