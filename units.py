"""Natural-unit system with GeV as the base unit.

Every physical quantity in this project is stored as a plain float in
natural units (hbar = c = k_B = 1) with GeV as the base unit.  The
constants below are the conversion factors between SI/cgs/astronomical
units and powers of GeV.

Usage
-----
Multiply to convert *into* natural units, divide to convert *out of*
them::

    length_gev = 5.0 * Centimeter   # 5 cm  -> GeV^-1
    temp_kelvin = t_gev / Kelv      # GeV   -> K

Naming
------
Physical constants keep their conventional symbols (``GeV``, ``Kelv``,
``m_e``) rather than the ``UPPER_CASE`` module-constant convention of
PEP 8.  This is a deliberate, project-wide deviation: matching the
symbols used in the accompanying paper is worth more than the naming
rule here.  See ``pyproject.toml`` for the corresponding lint
exemptions.
"""

import numpy as np

# ---------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------
GeV = 1  # [0]  Base unit
eV = 10**-9 * GeV  # [GeV]
KeV = 10**-6 * GeV  # [GeV]
MeV = 10**-3 * GeV  # [GeV]
TeV = 10**3 * GeV  # [GeV]

# ---------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------
Sec = (1 / (6.582119 * 10**-16)) / eV  # [GeV^-1]
Day = 86400 * Sec  # [GeV^-1]
Year = 365 * Day  # [GeV^-1]
Hz = 1 / Sec  # [GeV]

# ---------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------
Centimeter = 5.067730716156396 * 10**13 / GeV  # [GeV^-1]
Meter = 100 * Centimeter  # [GeV^-1]
Km = 10**5 * Centimeter  # [GeV^-1]
Angstrom = 1e-10 * Meter  # [GeV^-1]
Mpc = 3.086 * 10**24 * Centimeter  # [GeV^-1]
kpc = 1.0e-3 * Mpc  # [GeV^-1]
pc = 1e-3 * kpc  # [GeV^-1]
barn = 1e-24 * Centimeter**2  # [GeV^-2]

# ---------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------
Kilogram = 5.6095883571872 * 10**35 * eV  # [GeV]
Gram = 1e-3 * Kilogram  # [GeV]
amu = 1.66053892 * 10**-27 * Kilogram  # [GeV]
M_s = 1.99 * 10**30 * Kilogram  # [GeV]  Solar mass

# ---------------------------------------------------------------------
# Velocity, energy and flux
# ---------------------------------------------------------------------
Kmps = 3.3356 * 10**-6  # [0]  1 km/s in units of c
joule = Kilogram * Meter**2 / Sec**2  # [GeV]
erg = 1e-7 * joule  # [GeV]
Kelv = 8.62e-14 * GeV  # [GeV]
k_B = 1.3806488e-16 * erg / Kelv  # [0]
Jy = 1e-23 * erg / Sec / Centimeter**2 / Sec**-1  # [GeV^3]
KgDay = Kilogram * Day  # [0]

# ---------------------------------------------------------------------
# Electromagnetism
# ---------------------------------------------------------------------
alpha = 1 / 137  # [0]  Fine-structure constant
q_e = np.sqrt(4 * np.pi * alpha)  # [0]  Charge (Heaviside-Lorentz)
Coulomb = 6.2415090741e18 * q_e  # [0]
Amp = Coulomb / Sec  # [GeV]
Wb = Kilogram * Meter**2 / Sec**2 / Amp  # [0]
Mx = 1e-8 * Wb  # [0]
Gauss = Mx / Centimeter**2  # [GeV^2]
nGauss = 1e-9 * Gauss  # [GeV^2]

# ---------------------------------------------------------------------
# 21-cm transition
# ---------------------------------------------------------------------
omega_21 = 5.9e-6 * eV  # [GeV]  Hyperfine splitting of H(1s)

# ---------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------
asctorad = np.pi / 648000.0  # [0]  arcsec -> rad
radtoasc = 648000.0 / np.pi  # [0]  rad -> arcsec
