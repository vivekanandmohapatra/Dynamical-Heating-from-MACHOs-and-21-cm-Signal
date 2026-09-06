"""Background cosmology: parameters and redshift-dependent quantities.

Planck 2018 (TT,TE,EE+lowE+lensing+BAO) best-fit values, expressed in the
natural units defined in :mod:`units`.

Naming follows the physics symbols rather than PEP 8's ``UPPER_CASE``
module-constant convention; see the note in :mod:`units`.
"""

import numpy as np
from scipy.special import zeta

from units import (
    Centimeter,
    Kelv,
    Kilogram,
    Kmps,
    Meter,
    MeV,
    Mpc,
    Sec,
    alpha,
    kpc,
)

# ---------------------------------------------------------------------
# Cosmological parameters (Planck 2018)
# ---------------------------------------------------------------------
GN = 6.67e-11 * Meter**3 / Kilogram / Sec**2  # [GeV^-2]  Newton constant
h = 0.6766  # [0]  Reduced Hubble constant
H0 = 100 * h * (Kmps / Mpc)  # [GeV]  Hubble constant today
Tcmb0 = 2.725 * Kelv  # [GeV]  CMB temperature today
Neff = 3.044  # [0]  Effective number of neutrino species
nu_contri = Neff * (7 / 8) * (4 / 11) ** (4 / 3)  # [0]

rho_cmb0 = np.pi**2 * Tcmb0**4 / 15  # [GeV^4]  CMB energy density
rho_c0 = 3 * H0**2 / (8 * np.pi * GN)  # [GeV^4]  Critical density

omega_r = rho_cmb0 / rho_c0  # [0]  Radiation density parameter
omega_m = 0.30966  # [0]  Total matter density parameter
omega_b = 0.04897  # [0]  Baryon density parameter
omega_dm = omega_m - omega_b  # [0]  Cold dark matter
omega_de = 1 - omega_m - omega_r  # [0]  Cosmological constant

z_eq = 3402.0  # [0]  Matter-radiation equality
eta = 6e-10  # [0]  Baryon-to-photon ratio

# ---------------------------------------------------------------------
# Particle masses and atomic data
# ---------------------------------------------------------------------
m_p = 1.67261777e-27 * Kilogram  # [GeV]  Proton
m_e = 0.5109989461 * MeV  # [GeV]  Electron
m_mu = 0.106  # [GeV]  Muon
m_n = 0.939  # [GeV]  Neutron
m_H = 1.6735575e-27 * Kilogram  # [GeV]  Hydrogen

g_p = 2  # [0]  Spin d.o.f. of the proton
g_e = 2  # [0]  Spin d.o.f. of the electron
g_H = 4  # [0]  Spin d.o.f. of hydrogen

# ---------------------------------------------------------------------
# Baryon content
# ---------------------------------------------------------------------
Y_p = 0.245  # [0]  Primordial helium mass fraction
fHe = Y_p / (4 * (1 - Y_p))  # [0]  He/H by number
rho_b0 = omega_b * rho_c0  # [GeV^4]  Baryon energy density today
n_b0 = rho_b0 / m_H  # [GeV^3]  Baryon number density today
n_H0 = rho_c0 * omega_b * (1 - Y_p) / m_H  # [GeV^3]  Hydrogen today
rho_H0 = n_H0 * m_H  # [GeV^4]

# ---------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------
sigma_T = 6.6524616e-25 * Centimeter**2  # [GeV^-2]  Thomson cross section
alpha_em = alpha  # [0]  Fine-structure constant (no running below ~GeV)

# ---------------------------------------------------------------------
# Planck scale
# ---------------------------------------------------------------------
Mpl = GN ** (-1 / 2)  # [GeV]  Planck mass
Mpl_r = (8 * np.pi * GN) ** (-1 / 2)  # [GeV]  Reduced Planck mass

# ---------------------------------------------------------------------
# Galactic scales
# ---------------------------------------------------------------------
Rsun = 8.0 * kpc  # [GeV^-1]  Solar galactocentric radius
R200_MW = 200.0 * kpc  # [GeV^-1]  Milky Way virial radius

# ---------------------------------------------------------------------
# Recombination (RecFast++ fitting-function parameters)
# ---------------------------------------------------------------------
Rec_F = 1.14  # [0]  Fudge factor
Rec_a1, Rec_a2 = 4.309, -0.6166  # [0]  Case-B fit parameters
Rec_a3, Rec_a4 = 0.6703, 0.5300  # [0]  Case-B fit parameters

Lambda_2s = 8.22458 * Sec**-1  # [GeV]  H 2s-1s two-photon rate
E_ion2s = 3.39856e-9  # [GeV]  Ionization energy from H(2s)
E_ion = 13.6e-9  # [GeV]  Ionization energy from H(1s)
E_Lya = 10.28e-9  # [GeV]  Lyman-alpha energy
L_Lya = 2 * np.pi / E_Lya  # [GeV^-1]  Lyman-alpha wavelength


# ---------------------------------------------------------------------
# Redshift-dependent background quantities
# ---------------------------------------------------------------------
def T_CMB(z):
    """Return the CMB temperature at redshift ``z``. [GeV]

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        CMB temperature [GeV].
    """
    return Tcmb0 * (1 + z)


def H_z(z):
    """Return the Hubble rate at redshift ``z``. [GeV]

    Includes the massless-neutrino contribution in the radiation term.

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        Hubble rate [GeV].
    """
    rho_cmb0_corr = (1.0 + nu_contri) * np.pi**2 * Tcmb0**4 / 15
    omega_r_corr = rho_cmb0_corr / rho_c0
    return H0 * (
        omega_de + omega_m * (1 + z) ** 3 + omega_r_corr * (1 + z) ** 4
    ) ** (1 / 2)


def rho_c(z):
    """Return the critical density at redshift ``z``. [GeV^4]

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        Critical density [GeV^4].
    """
    return (3 / (8 * np.pi * GN)) * (H_z(z) ** 2)


def n_CMB(z):
    """Return the CMB photon number density at redshift ``z``. [GeV^3]

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        Photon number density [GeV^3].
    """
    return (2 * zeta(3)) / (np.pi**2) * (Tcmb0**3) * (1 + z) ** 3


def rho_CMB(z):
    """Return the CMB energy density at redshift ``z``. [GeV^4]

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        Photon energy density [GeV^4].
    """
    return rho_cmb0 * (1 + z) ** 4


def n_b(z):
    """Return the baryon number density at redshift ``z``. [GeV^3]

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        Baryon number density [GeV^3].
    """
    return n_b0 * (1 + z) ** 3


def n_BB(x):
    """Return the blackbody occupation number at ``x = E / T``. [0]

    Parameters
    ----------
    x : float or numpy.ndarray
        Dimensionless photon energy ``h nu / k_B T``.

    Returns
    -------
    float or numpy.ndarray
        Occupation number [0].
    """
    return 1 / (np.exp(x) - 1)


def n_Y(x):
    """Return the Compton-y spectral shape at ``x = E / T``. [0]

    Parameters
    ----------
    x : float or numpy.ndarray
        Dimensionless photon energy ``h nu / k_B T``.

    Returns
    -------
    float or numpy.ndarray
        y-distortion spectral shape [0].
    """
    coeff = (x * np.exp(x)) / ((np.exp(x) - 1) ** 2)
    return coeff * (x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4)
