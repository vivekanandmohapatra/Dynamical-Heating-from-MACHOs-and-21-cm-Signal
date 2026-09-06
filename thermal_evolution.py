"""
Thermal evolution of the IGM.

Solves the coupled ODEs for the ionization fraction X_e, matter temperature
T_M, and MACHO-baryon relative velocity v_rel, including:
  - Standard recombination (RecFast++ fitting functions)
  - Compton heating/cooling
  - X-ray heating from star-forming haloes (Press-Schechter; see
    the StarFormation class)
  - Dynamical friction heating from MACHOs (MACHOHeating class)

All quantities are in natural units (GeV-based) unless stated otherwise.

Sign convention for ODEs
------------------------
All ODEs are written as dY/dz and integrated from z_start (high) to z_end
(low), so dz < 0 at every internal step of the solver.  The signs of every
term follow from the chain rule dt/dz = -1 / (H * (1+z)).
"""

import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
from scipy.special import erfc

from cosmology_terms import (
    GN,
    T_CMB,
    E_ion2s,
    E_Lya,
    H_z,
    L_Lya,
    Lambda_2s,
    Rec_a1,
    Rec_a2,
    Rec_a3,
    Rec_a4,
    Rec_F,
    Y_p,
    fHe,
    h,
    m_e,
    m_H,
    n_b,
    omega_dm,
    omega_m,
    rho_b0,
    rho_c0,
    rho_CMB,
    sigma_T,
)
from units import (
    Kelv,
    M_s,
    Meter,
    Sec,
)

cosmology.setCosmology("planck18")


# ============================================================================
# Recombination Physics  (RecFast++ fitting functions)
# ============================================================================


def rec_alpha(T_M):
    """
    Case-B recombination coefficient (RecFast++ fit). [GeV^-2]

    Parameters
    ----------
    T_M : float
        Matter temperature [GeV].

    Returns
    -------
    float
        Alpha_B [GeV^-2].
    """
    t = T_M / (1e4 * Kelv)
    return (
        1e-19
        * Rec_F
        * Rec_a1
        * t**Rec_a2
        / (1.0 + Rec_a3 * t**Rec_a4)
        * Meter**3
        / Sec
    )


def rec_beta(T):
    """
    Photo-ionization rate (RecFast++ fit). [GeV]

    Uses T_CMB, not T_M — old implementations using T_M are incorrect.

    Parameters
    ----------
    T : float
        CMB temperature [GeV].

    Returns
    -------
    float
        Beta [GeV].
    """
    mu_red = 1.0 / (1.0 + 5.44617e-4)  # reduced mass ratio of e-p system
    return (
        rec_alpha(T)
        * ((m_e * mu_red * T) / (2.0 * np.pi)) ** 1.5
        * np.exp(-E_ion2s / T)
    )


def rec_BH(T):
    """
    Boltzmann suppression factor for Lyman-alpha (RecFast++ BH term). [0]

    Parameters
    ----------
    T : float
        CMB temperature [GeV].

    Returns
    -------
    float
        BH factor [0].
    """
    return np.exp(-E_Lya / T)


def rec_KH(z):
    """
    Sobolev escape-probability denominator (RecFast++ K_H). [GeV^-4]

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        K_H [GeV^-4].
    """
    return L_Lya**3 / (8.0 * np.pi * H_z(z))


# ============================================================================
# Star-Formation Physics — encapsulated in StarFormation class
# ============================================================================


class StarFormation:
    """
    Star-formation physics for X-ray and ionization heating of the IGM.

    Computes the collapsed fraction of baryons in dark-matter haloes above a
    virial-temperature threshold using the Press-Schechter formalism
    (via colossus), and derives the resulting X-ray heating and secondary
    ionization rates per unit redshift.

    Parameters
    ----------
    f_star : float
        Star-formation efficiency (fraction of collapsed baryons that form
        stars).  Typical range: 0.01 - 0.5.
    T_vir4 : float
        Minimum virial temperature for star-forming haloes in units of 1e4 K
        (e.g. T_vir4 = 1 means T_vir = 1e4 K, the atomic cooling threshold).
    f_X : float, optional
        X-ray ionization efficiency per baryon.  Encodes the fraction of
        X-ray energy deposited as secondary ionizations.  Default: 1.22.
    x_ray_norm : float, optional
        Normalisation of the X-ray heating rate per unit |df_coll/dz|
        in Kelvin.  Default: 5e5 K (standard semi-numerical value).
    dz_fd : float, optional
        Step size for the finite-difference derivative df_coll/dz.
        Default: 1e-2.
    z_xray : float, optional
        Redshift below which X-ray sources are assumed active.  Default: 60.
    """

    def __init__(
        self,
        f_star,
        T_vir4,
        f_X=1.22,
        x_ray_norm=5e5,
        dz_fd=1e-1,
        z_xray=60.0,
    ):
        if not (0.0 < f_star <= 1.0):
            raise ValueError(f"f_star must be in (0, 1], got {f_star}.")
        if T_vir4 <= 0:
            raise ValueError(f"T_vir4 must be positive, got {T_vir4}.")

        self.f_star = f_star
        self.T_vir4 = T_vir4
        self.f_X = f_X
        self.x_ray_norm = x_ray_norm  # [K]
        self.dz_fd = dz_fd
        self.z_xray = z_xray

    def __repr__(self):
        return (
            f"StarFormation(f_star={self.f_star}, T_vir4={self.T_vir4}, "
            f"f_X={self.f_X}, x_ray_norm={self.x_ray_norm:.2e} K, "
            f"z_xray={self.z_xray})"
        )

    # ------------------------------------------------------------------
    # Halo mass threshold
    # ------------------------------------------------------------------

    def halo_mass_min(self, z):
        """
        Minimum halo mass for star formation. [Solar Mass]

        Derived from the virial temperature criterion:
            M_min = 1e8 * (Omega_m h^2)^{-0.5}
                    * [10/(1+z) * (mu/mu_H) * T_vir / (1.98e4 K)]^{3/2}

        Returned in Solar Mass because colossus requires it.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Minimum halo mass [Solar Mass].
        """
        omega_mh2_inv_sqrt = (omega_m * h**2) ** (-0.5)
        mu_ratio = 0.6 / 1.22  # mean molecular weight ratio mu/mu_H
        t_vir_norm = self.T_vir4 / 1.98  # T_vir / (1.98e4 K)

        return (
            1e8
            * omega_mh2_inv_sqrt
            * (10.0 / (1.0 + z) * mu_ratio * t_vir_norm) ** 1.5
        )

    # ------------------------------------------------------------------
    # Collapsed fraction and its derivative
    # ------------------------------------------------------------------

    def f_coll(self, z):
        """
        Collapsed baryon fraction in haloes above M_min (Press-Schechter). [0]

        f_coll(z) = erfc(nu(M_min, z) / sqrt(2))

        where nu is the peak-height computed by colossus.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Collapsed fraction [0].
        """
        nu = peaks.peakHeight(self.halo_mass_min(z), z)
        return erfc(nu / np.sqrt(2.0))

    def df_coll_dz(self, z):
        """
        Numerical derivative of the collapsed fraction, df_coll/dz. [0]

        Uses a second-order centered finite difference with half-step
        self.dz_fd / 2, which gives O(dz^2) accuracy and avoids the
        asymmetry of the former forward-difference stencil.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            df_coll/dz [0].
        """
        h = self.dz_fd / 2.0
        return (self.f_coll(z + h) - self.f_coll(z - h)) / (2.0 * h)

    # ------------------------------------------------------------------
    # Star formation rate density
    # ------------------------------------------------------------------

    def sfrd(self, z):
        """
        Star formation rate density (SFRD). [GeV^5]

        SFRD = -(1+z) * f_star * rho_b0 * (df_coll/dz) * H(z)

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            SFRD [GeV^4].
        """
        return -(1.0 + z) * self.f_star * rho_b0 * self.df_coll_dz(z) * H_z(z)

    # ------------------------------------------------------------------
    # X-ray heating and ionization rates
    # ------------------------------------------------------------------

    def xray_heating_rate(self, z):
        """
        X-ray heating contribution to dT_M/dz. [GeV]

        Active only for z < self.z_xray.

            dT_xray/dz = x_ray_norm [K] * Kelv * f_star * |df_coll/dz|

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            dT_xray/dz [GeV].
        """
        if z >= self.z_xray:
            return 0.0
        return self.x_ray_norm * Kelv * self.f_star * abs(self.df_coll_dz(z))

    def xray_ionization_rate(self, z):
        """
        X-ray secondary ionization contribution to dX_e/dz. [0]

        Active only for z < self.z_xray.

            dXe_xray/dz = f_X * f_star * |df_coll/dz|

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            dXe_xray/dz [0].
        """
        if z >= self.z_xray:
            return 0.0
        return self.f_X * self.f_star * abs(self.df_coll_dz(z))


# ============================================================================
# MACHO Dynamical Heating — encapsulated in MACHOHeating class
# ============================================================================


class MACHOHeating:
    """
    Dynamical friction heating of the IGM by MACHOs.

    Computes the Chandrasekhar drag force on a MACHO moving through the baryon
    fluid and the resulting volumetric heating rate per baryon, following
    Eq. (1) of Bhalla, Ireland, Liu, Xiao & Xu, arXiv:2512.00169.

    Coulomb-log function I(Mach, Lambda): the full piecewise expression of
    Eq. (17) of arXiv:2512.00169 (SM), continuous through Mach = 1 and
    vanishing continuously as Lambda -> 1+ (see coulomb_log).

    Nonlinear regime (Kim & Kim 2009, arXiv:0908.1391): with
    nonlinear_cutoff=True the supersonic branch replaces the fiducial BHL
    b_min by the hydrostatic-envelope scale 2*G*M/(v_rel^2 - c_s^2),
    calibrated against the Kim & Kim simulations.  The two settings bracket
    the accreting (fiducial; Suzuguchi et al. 2024) and non-accreting
    (Kim & Kim) limits, and their difference is the nonlinear systematic.

    Supports monochromatic and three extended mass distributions:
    log-normal, critical-collapse, and power-law.

    Parameters
    ----------
    Mass : float
        MACHO mass [Solar Mass].
        For monochromatic (model=Monochromatic): the single MACHO mass.
        For model='power_law': must satisfy M_min <= Mass <= M_max.
    model : {None, 'log_normal', 'critical_collapse', 'power_law'}
        Mass function model.  None = monochromatic (delta function).
    frac : float
        Fraction of DM composed of MACHOs.  Range: (0, 1].
    M_min : float, optional
        Lower mass integration limit for extended distributions [Solar Mass].
        Default: 1e-2 M_sun.  Must be positive.
    M_max : float, optional
        Upper mass integration limit for extended distributions [Solar Mass].
        Default: 1e7 M_sun.  Must be greater than M_min.
    sigma_ln : float, optional
        Width of the log-normal distribution.  Default: 1.0.
    M_c : float, optional
        Central/characteristic mass for log_normal and critical_collapse
        distributions [Solar Mass].  Defaults to Mass if not provided.
    eos : float, optional
        Equation-of-state parameter w for the power_law model.
        Required when model='power_law'.  Must satisfy -1/3 <= eos <= 1.
        The spectral index is derived as g = -2w / (1 + w).
    nonlinear_cutoff : bool, optional
        If True, apply the Kim & Kim (2009) nonlinear-envelope cutoff to the
        supersonic Coulomb logarithm, b_min -> 2*G*M/(v_rel^2 - c_s^2).
        Default False (fiducial BHL cutoff).  Run the mass grid with both
        settings and quote the band as the nonlinear systematic.
    baryon_backreaction : bool, optional
        If True (default), include the baryon bulk-recoil term in the
        relative-velocity equation (momentum conservation; absent from
        Eq. (4) of arXiv:2512.00169 — set False to reproduce that paper).
        See recoil_acceleration().
    """

    # Numerical floor for velocities inside drag evaluations (units of c).
    # The physical v_rel never reaches zero (dv/dz -> 0 as v -> 0, because
    # I(Mach) -> Mach^3/3 kills the force faster than 1/v^2 grows it), but a
    # stiff solver can transiently step to a tiny or negative trial value.
    V_FLOOR = 1e-30

    # NOTE: 'power law' and 'power_law' are both accepted and normalised to
    # 'power_law' internally.  In the previous version VALID_MODELS advertised
    # 'power law' (with a space) while every downstream branch tested for
    # 'power_law' (underscore), so the power-law model could never actually
    # execute: it passed validation, skipped the eos check, and then fell
    # through every branch of _mass_distribution_gev returning None.
    VALID_MODELS = (
        "Monochromatic",
        "Log-normal",
        "Critical collapse",
        "power law",
        "power_law",
    )

    def __init__(
        self,
        Mass,
        model="Monochromatic",
        frac=1.0,
        M_min=None,
        M_max=None,
        sigma_ln=2,
        M_c=None,
        eos=None,
        n_bins=50,
        nonlinear_cutoff=True,
        baryon_backreaction=False,
    ):
        """
        All mass arguments (Mass, M_c, M_min, M_max) must be supplied in
        Solar Mass units [M_sun].  Internal physics is computed in GeV.

        n_bins : number of log-spaced mass-quadrature nodes used for
        extended distributions.  Forced to 1 for the monochromatic model.
        """
        # ------------------------------------------------------------------
        # Validate model
        # ------------------------------------------------------------------
        if model not in self.VALID_MODELS:
            raise ValueError(
                f"model must be one of {self.VALID_MODELS}, got '{model}'."
            )
        if model == "power law":
            model = "power_law"  # canonical internal spelling

        # ------------------------------------------------------------------
        # Validate frac
        # ------------------------------------------------------------------
        if not (0.0 < frac <= 20.0):
            raise ValueError(
                f"frac (DM fraction) must be in (0, 20], got {frac}.  "
                f"(Physical range is (0, 1]; values above 1 are accepted "
                f"only as a lever for heating-sensitivity tests.)"
            )

        # ------------------------------------------------------------------
        # Resolve and validate mass limits [Solar Mass]
        # ------------------------------------------------------------------
        M_min_resolved = float(M_min) if M_min is not None else 1e-2
        M_max_resolved = float(M_max) if M_max is not None else 1e7

        if M_min_resolved <= 0.0:
            raise ValueError(
                f"M_min must be positive, got {M_min_resolved} M_sun."
            )
        if M_max_resolved <= M_min_resolved:
            raise ValueError(
                f"M_max must be greater than M_min.  "
                f"Got M_min={M_min_resolved} M_sun, "
                f"M_max={M_max_resolved} M_sun."
            )

        # ------------------------------------------------------------------
        # power_law-specific validation
        # ------------------------------------------------------------------
        if model == "power_law":
            # eos is mandatory
            if eos is None:
                raise ValueError(
                    "eos must be provided when model='power_law'."
                )
            # Physical range: -1/3 <= w <= 1
            if not (-1.0 / 3.0 <= eos <= 1.0):
                raise ValueError(
                    f"eos must be in [-1/3, 1] for model='power_law', "
                    f"got {eos:.6g}.  "
                    f"(-1/3 corresponds to a string-dominated universe; "
                    f"1 corresponds to a stiff fluid.)"
                )
            # Mass must lie within the integration range
            if not (M_min_resolved <= float(Mass) <= M_max_resolved):
                raise ValueError(
                    f"For model='power_law', Mass must satisfy "
                    f"M_min <= Mass <= M_max.  "
                    f"Got Mass={float(Mass)} M_sun, "
                    f"M_min={M_min_resolved} M_sun, "
                    f"M_max={M_max_resolved} M_sun."
                )

        # ------------------------------------------------------------------
        # Store user-facing solar-mass values (for display and output)
        # ------------------------------------------------------------------
        self.Mass_Msun = float(Mass)
        self.M_c_Msun = float(M_c) if M_c is not None else self.Mass_Msun
        self.M_min_Msun = M_min_resolved
        self.M_max_Msun = M_max_resolved

        # ------------------------------------------------------------------
        # Store internal GeV values (used by all physics methods)
        # ------------------------------------------------------------------
        self.Mass = self.Mass_Msun * M_s
        self.M_c = self.M_c_Msun * M_s
        self.M_min = self.M_min_Msun * M_s
        self.M_max = self.M_max_Msun * M_s

        self.model = model
        self.frac = frac
        self.sigma_ln = sigma_ln
        self.eos = eos  # None for non-power_law models

        self.nonlinear_cutoff = bool(nonlinear_cutoff)
        self.baryon_backreaction = bool(baryon_backreaction)

        # ------------------------------------------------------------------
        # Discretise the mass function into quadrature nodes.  This is what
        # makes the per-mass velocity treatment possible: every node carries
        # its own velocity in the ODE state vector.
        # ------------------------------------------------------------------
        self._setup_bins(int(n_bins))

    def __repr__(self):
        base = (
            f"MACHOHeating(\n"
            f"  Mass     = {self.Mass_Msun:.3e} M_sun,\n"
            f"  model    = {self.model!r},\n"
            f"  frac     = {self.frac},\n"
            f"  M_c      = {self.M_c_Msun:.3e} M_sun,\n"
            f"  M_min    = {self.M_min_Msun:.3e} M_sun,\n"
            f"  M_max    = {self.M_max_Msun:.3e} M_sun,\n"
            f"  sigma_ln = {self.sigma_ln},\n"
            f"  nonlinear_cutoff    = {self.nonlinear_cutoff},\n"
            f"  baryon_backreaction = {self.baryon_backreaction}"
        )
        if self.model == "power_law":
            g = -2.0 * self.eos / (1.0 + self.eos)
            base += f",\n  eos      = {self.eos:.6g}  " f"(g = {g:.4f})"
        return base + "\n)"

    # ------------------------------------------------------------------
    # Physics of relative motion of MACHOs amidst baryonic fluid
    # ------------------------------------------------------------------

    def sound_speed(self, Tb):
        """
        Baryon thermal sound speed in units of c = 1. [0]

        c_s = sqrt(gamma * T_b / (mu * m_H))

        Appropriate for the post-decoupling IGM (after recombination),
        where the baryon fluid is no longer tightly coupled to photons.
        gamma = 5/3 (monatomic ideal gas), mu = 1.22 (mean molecular weight
        for a fully neutral primordial gas).

        Parameters
        ----------
        Tb : float
            Baryon (matter) temperature [GeV].

        Returns
        -------
        float
            Sound speed [0] (in units where c = 1).
        """
        gamma = 5.0 / 3.0
        mu = 1.22
        return np.sqrt(gamma * Tb / (mu * m_H))

    def mach_number(self, z, v_rel, Tb):  # noqa: ARG002
        """
        Mach number M = v_rel / c_s. [0]

        Parameters
        ----------
        z : float
            Redshift.
        v_rel : float
            MACHO-baryon relative velocity [0].
        Tb : float
            Baryon temperature [GeV].

        Returns
        -------
        float
            Mach number [0].
        """
        return v_rel / self.sound_speed(Tb)

    def b_max(self, z, v_rel):
        """
        Maximum impact parameter — Jeans length x Mach number. [GeV^-1]

        b_max = M * R_J = v_rel * sqrt(pi / (G_N * rho_B(z)))

        Following Eq.(18) of arXiv:2512.00169 and McQuinn & O'Leary (2012).
        """
        rho_Bz = rho_b0 * (1.0 + z) ** 3
        return v_rel * np.sqrt(np.pi / (GN * rho_Bz))

    def b_min(self, z, v_rel, Mass, Tb):  # noqa: ARG002
        """
        Minimum impact parameter: Bondi-Hoyle-Lyttleton radius. [GeV^-1]

        b_min = G*M / (c_s^2 + v_rel^2)

        Fiducial choice (Eq. 18 of arXiv:2512.00169), appropriate for
        MACHOs smaller than r_BHL and supported for ACCRETING objects by
        Suzuguchi et al. (2024), who find no net drag from gas inside
        r_BHL.  For non-accreting objects, coulomb_log overrides this in
        the supersonic regime with the Kim & Kim (2009) envelope scale
        when nonlinear_cutoff is set.

        Parameters
        ----------
        z : float
            Redshift.
        v_rel : float
            Relative velocity [0].
        Mass : float
            MACHO mass [GeV].
        Tb : float
            Baryon temperature [GeV].

        Returns
        -------
        float
            b_min [GeV^-1].
        """
        v2 = self.sound_speed(Tb) ** 2 + v_rel**2
        return GN * Mass / v2

    def coulomb_log(self, z, v_rel, Mass, Tb):
        """
        Full piecewise I(M, Lambda) — Eq. (17) of arXiv:2512.00169 (SM). [0]

        Derived by carefully integrating the gravitational deceleration of the
        baryon wake over all impact parameters from b_min to b_max, without
        any approximation near M = 1.  This replaces the two-regime
        approximation (M << 1 and M >> 1), which diverges at M = 1.

        Definition
        ----------
        Lambda   = b_max / b_min          (Coulomb logarithm argument)
        x_min    = (1 + M) / Lambda       (dimensionless near-sonic parameter)

        The four regimes (all continuous at their boundaries):

          Case 1 — Deep subsonic,  M < 1 - x_min:
            i_val = 0.5 * ln((1+M)/(1-M)) - M

          Case 2 — Near-sonic lower,  1 - x_min < M < sqrt(1 + x_min^2):
            i_val = x_min/4 - M/2 - (1-M^2)/(4*x_min) + 0.5*ln((1+M)/x_min)

          Case 3 — Near-sonic upper,  sqrt(1+x_min^2) < M < 1 + x_min:
            i_val = -1/(4*x_min) + (M-x_min)^2/(4*x_min) + 0.5*ln((1+M)/x_min)

          Case 4 — Supersonic,  M > 1 + x_min:
            i_val = 0.5*ln((M+1)/(M-1)) + ln((M-1)/x_min)

        Asymptotic limits (for reference):
          M << 1:  I ~ M^3 / 3   (DF strongly suppressed pre-reionization)
          M >> 1:  I ~ ln(Lambda)       (standard supersonic limit)

        In practice Lambda >> 1 so x_min << 1, and Cases 2 and 3 span an
        extremely narrow Mach-number window near M = 1.  The dominant physics
        is Cases 1 and 4, but all four are needed for a continuous and
        well-defined force across all redshifts.

        Optional nonlinear cutoff (self.nonlinear_cutoff = True)
        --------------------------------------------------------
        Kim & Kim (2009, arXiv:0908.1391) show that a massive point-like
        perturber develops a quasi-hydrostatic, front-back symmetric envelope
        bounded by a detached bow shock at delta = G*M/(c_s^2 (M^2 - 1)); the
        envelope contributes no net drag (their Figs. 8-9 and 16).  Excising
        r < 2*delta reproduces their fitted nonlinear force
        F/F_lin = (eta/2)^(-0.45) to a few percent at M = 1.5 over
        eta = 10-100 (and to ~20% at M = 3).  Implemented as a
        supersonic-only rescaling

            b_min -> b_min_BHL * 2*(M^2 + 1)/(M^2 - 1)
                   = 2*G*M / (v_rel^2 - c_s^2) ,

        i.e. Delta ln(Lambda) ~= -1.2 at M ~ 2.  The subsonic branch keeps
        the linear value, which Kim & Kim verify directly (F ~ F_lin up to
        A = 50).  NOTE: I is then discontinuous at M = 1+ (the envelope
        diverges and I drops toward zero); along physical trajectories this
        affects only bins hovering near the sonic point.  If it bothers the
        solver, cap I across 1 < M < 1 + x_min at its M = 1- value instead.
        The BHL (False) and Kim & Kim (True) settings bracket the accreting
        and non-accreting limits.

        Parameters
        ----------
        z : float
            Redshift.
        v_rel : float
            Relative velocity [0].
        Mass : float
            MACHO mass [GeV].
        Tb : float
            Baryon temperature [GeV].

        Returns
        -------
        float
            I(M, Lambda) [dimensionless].  Guarded to be >= 0.
        """
        mach = self.mach_number(z, v_rel, Tb)
        b_min = self.b_min(z, v_rel, Mass, Tb)

        # Optional Kim & Kim (2009) nonlinear-envelope cutoff: supersonic
        # b_min -> 2*G*M/(v^2 - c_s^2) = b_min_BHL * 2*(M^2+1)/(M^2-1).
        # See the docstring above.  Subsonic (Case 1) keeps the BHL cutoff.
        if self.nonlinear_cutoff and mach > 1.0:
            b_min = b_min * 2.0 * (mach**2 + 1.0) / (mach**2 - 1.0)

        # Coulomb logarithm argument.  The piecewise I below vanishes
        # continuously as Lambda -> 1+, so returning 0 for Lambda <= 1 is the
        # continuous limit, not a clamp: b_min >= b_max means the no-drag
        # region has swallowed the entire wake, the perturber dominates the
        # gas out to the Jeans scale, and the perturbative DF description no
        # longer applies (drag self-terminates; with the BHL cutoff this
        # happens below z ~ 40 / 170 / 500 for 1e5 / 1e6 / 1e7 M_sun on rms
        # streaming trajectories).
        Lambda = self.b_max(z, v_rel) / b_min
        if Lambda <= 1.0:
            return 0.0

        # Near-sonic parameter: x_min << 1 when Lambda >> 1
        x_min = (1.0 + mach) / Lambda

        # ---- Case 1: deep subsonic, M < 1 - x_min -------------------------
        if mach < 1.0 - x_min:
            if mach < 1e-3:
                # Taylor expansion prevents catastrophic cancellation
                # for very small M: 0.5*ln((1+M)/(1-M)) - M ≈ M^3 / 3
                i_val = (mach**3) / 3.0
            else:
                i_val = 0.5 * np.log((1.0 + mach) / (1.0 - mach)) - mach

        # ---- Case 2: near-sonic lower, 1 - x_min <= M < sqrt(1+x_min^2) ---
        elif mach < np.sqrt(1.0 + x_min**2):
            i_val = (
                x_min / 4.0
                - mach / 2.0
                - (1.0 - mach**2) / (4.0 * x_min)
                + 0.5 * np.log((1.0 + mach) / x_min)
            )

        # ---- Case 3: near-sonic upper, sqrt(1+x_min^2) <= M < 1 + x_min ---
        elif mach < 1.0 + x_min:
            i_val = (
                -1.0 / (4.0 * x_min)
                + (mach - x_min) ** 2 / (4.0 * x_min)
                + 0.5 * np.log((1.0 + mach) / x_min)
            )

        # ---- Case 4: supersonic, M >= 1 + x_min ----------------------------
        else:
            i_val = 0.5 * np.log((mach + 1.0) / (mach - 1.0)) + np.log(
                (mach - 1.0) / x_min
            )

        return max(
            i_val, 0.0
        )  # guard against small negative values near M ~ 0

    # ------------------------------------------------------------------
    # Drag force
    # ------------------------------------------------------------------

    def drag_force(self, z, v_rel, Mass, Tb):
        """
        Chandrasekhar dynamical friction force on a single MACHO. [GeV^2]

        F_DF = 4*pi*G^2 * M^2 * rho_b(z) / v_rel^2 * I(M)

        Parameters
        ----------
        z : float
            Redshift.
        v_rel : float
            Relative velocity [0].
        Mass : float
            MACHO mass [GeV].
        Tb : float
            Baryon temperature [GeV].

        Returns
        -------
        float
            Drag force magnitude [GeV^2].
        """
        rho_bz = rho_b0 * (1.0 + z) ** 3
        i_val = self.coulomb_log(z, v_rel, Mass, Tb)
        return 4.0 * np.pi * GN**2 * Mass**2 * rho_bz / v_rel**2 * i_val

    # ------------------------------------------------------------------
    # Mass distribution
    # ------------------------------------------------------------------

    def _mass_distribution_gev(self, Mass_gev):
        """
        Internal: normalised dN/dM evaluated at Mass_gev [GeV].

        This is called by _heating_integrand where the integration variable
        from quad is already in GeV.  Do NOT call this directly from user
        code — use the public mass_distribution() instead.

        Parameters
        ----------
        Mass_gev : float
            MACHO mass [GeV].

        Returns
        -------
        float
            dN/dM [GeV^-1], or dimensionless 1 for monochromatic.
        """
        if self.model == "Monochromatic":
            return 1.0

        if self.model == "Log-normal":
            s = self.sigma_ln
            ln_m = np.log(Mass_gev / self.M_c)
            return np.exp(-(ln_m**2) / (2.0 * s**2)) / (
                s * Mass_gev * np.sqrt(2.0 * np.pi)
            )

        if self.model == "Critical collapse":
            x = (Mass_gev / self.M_c) ** 2.85
            return (3.2 / self.M_c) * x * np.exp(-x)

        if self.model == "power_law":
            # Standard PBH convention (Carr 1975):  psi(M) ~ M^(g-1),
            # g = 2w/(1+w).  The previous line was np.abs(-2w/(1+w)), which
            # coincides with this for w > 0 but gives the WRONG SIGN of the
            # slope for w < 0.  This branch never executed before (see the
            # VALID_MODELS note above), so no published result changes.
            g = 2.0 * self.eos / (1.0 + self.eos)
            # Normalisation: ∫_{M_min}^{M_max} g*M^(g-1) dM = M_max^g - M_min^g
            norm = self.M_max**g - self.M_min**g
            return g * Mass_gev ** (g - 1) / norm

        msg = f"Unhandled mass-function model: {self.model!r}."
        raise ValueError(msg)

    def mass_distribution(self, Mass_Msun):
        """
        Public: normalised dN/dM for extended MACHO populations.

        Accepts mass in Solar Mass units and converts internally to GeV
        before evaluating the distribution.

        Models
        ------
        None               -> monochromatic; returns 1.
        'log_normal'       -> Dolgov & Silk (1993), Phys. Rev. D 47, 4244.
        'critical_collapse' -> Niemeyer & Jedamzik (1998) / Yokoyama (1998).
        'power_law'        -> dN/dM ∝ M^(g-1),  g = -2w / (1+w).

        Parameters
        ----------
        Mass_Msun : float or array_like
            MACHO mass [Solar Mass].  Converted to GeV internally.

        Returns
        -------
        float
            dN/dM [GeV^-1], or dimensionless 1 for monochromatic.
        """
        return self._mass_distribution_gev(Mass_Msun * M_s)

    # ------------------------------------------------------------------
    # Mass-function discretisation  (quadrature nodes for the multi-bin
    # velocity treatment)
    # ------------------------------------------------------------------

    def _effective_support(self):
        """
        Effective support of psi(M), intersected with [M_min, M_max]. [GeV]

        Placing the quadrature nodes only where psi(M) is non-negligible keeps
        the mass integral accurate at a modest number of bins.  Spreading
        n_bins uniformly across a range as wide as [1, 1e8] M_sun when psi(M)
        lives in a decade around M_c would resolve nothing.

        Returns
        -------
        (lo, hi) : tuple of float
            Lower and upper mass nodes [GeV].
        """
        if self.model == "Log-normal":
            lo = self.M_c * np.exp(-6.0 * self.sigma_ln)
            hi = self.M_c * np.exp(+6.0 * self.sigma_ln)

        elif self.model == "Critical collapse":
            # psi ~ x exp(-x) with x = (M/M_c)^2.85:
            # negligible below ~0.01 M_c and above ~4 M_c (x ~ 42 there).
            lo, hi = 1e-2 * self.M_c, 4.0 * self.M_c

        else:  # power_law: pure power law, no interior peak -> full range
            lo, hi = self.M_min, self.M_max

        lo = max(lo, self.M_min)
        hi = min(hi, self.M_max)

        if hi <= lo:
            raise ValueError(
                "Mass-function support does not overlap [M_min, M_max]: "
                f"support ~ [{lo / M_s:.3e}, {hi / M_s:.3e}] M_sun, "
                f"limits [{self.M_min_Msun:.3e}, {self.M_max_Msun:.3e}] M_sun."
            )
        return lo, hi

    def _setup_bins(self, n_bins):
        """
        Build the mass-quadrature grid used by the per-bin dynamics.

        Sets
        ----
        n_bins    : int             number of mass nodes (1 for monochromatic)
        M_nodes   : ndarray [GeV]   quadrature nodes M_i
        w_nodes   : ndarray [0]     trapezoidal weights in ln M
        psi_nodes : ndarray [GeV^-1] psi(M_i), the normalised mass function
        mass_norm : float [0]       sum_i w_i M_i psi_i ~ int_{M_min}^{M_max}
                                    psi dM.  This is the fraction of the mass
                                    function actually captured inside the
                                    integration limits.  It is REPORTED, not
                                    used to renormalise.  If it is not close
                                    to 1, the truncation is doing real work
                                    and the result depends on M_min / M_max.

        Monochromatic convention
        ------------------------
        A delta function in the mass-normalised convention is psi(M) =
        delta(M - M_0), so that int psi dM = 1 and rho_MACHO = frac * rho_DM.
        Setting  w_1 = 1  and  psi_1 = 1 / M_0  reproduces this exactly:

            Q_vol = rho_dm * frac * sum_i w_i psi_i v_i F_i
                  = (rho_dm / M_0) * frac * v * F     [monochromatic]
            mass_norm = w_1 * M_1 * psi_1 = 1

        so the monochromatic and extended code paths are literally the SAME
        expression -- no special-casing, and the narrow-distribution limit is
        continuous by construction.  (The previous version had two separate
        branches whose deceleration terms disagreed by a factor `frac`.)
        """
        if self.model == "Monochromatic":
            self.n_bins = 1
            self.M_nodes = np.array([self.Mass])
            self.w_nodes = np.array([1.0])
            self.psi_nodes = np.array([1.0 / self.Mass])
            self.mass_norm = 1.0
            return

        if n_bins < 4:
            raise ValueError(
                f"n_bins must be >= 4 for extended "
                f"distributions, got {n_bins}."
            )

        lo, hi = self._effective_support()

        u = np.linspace(np.log(lo), np.log(hi), n_bins)
        du = u[1] - u[0]

        w = np.full(n_bins, du)
        w[0] *= 0.5
        w[-1] *= 0.5  # trapezoidal rule in ln M

        self.n_bins = n_bins
        self.M_nodes = np.exp(u)
        self.w_nodes = w
        self.psi_nodes = np.array(
            [self._mass_distribution_gev(M) for M in self.M_nodes]
        )
        self.mass_norm = float(
            np.sum(self.w_nodes * self.M_nodes * self.psi_nodes)
        )

    def v0_vector(self, v_rel_ini):
        """
        Initial per-bin velocity vector. [0]

        Every mass starts at the same v_bc: the rms baryon-DM streaming
        velocity at recombination is mass-independent.  The histories diverge
        afterwards because the deceleration F_DF/M scales linearly with M.
        """
        return np.full(self.n_bins, float(v_rel_ini))

    # ------------------------------------------------------------------
    # Per-bin drag forces
    # ------------------------------------------------------------------

    def drag_forces(self, z, v_vec, Tb):
        """
        Chandrasekhar drag force on ONE object of each mass node. [GeV^2]

            F_i = 4 pi G^2 M_i^2 rho_b(z) / v_i^2 * I(Mach_i, Lambda_i)

        Evaluated at that node's OWN velocity v_i.  Note the Mach number
        Mach_i = v_i / c_s(T_M) uses the shared gas sound speed: this is the
        only channel through which the bins talk to each other.

        Parameters
        ----------
        z : float
            Redshift.
        v_vec : ndarray, shape (n_bins,)
            Per-bin relative velocities [0].
        Tb : float
            Baryon temperature [GeV].

        Returns
        -------
        ndarray, shape (n_bins,)
            Drag forces [GeV^2].
        """
        v = np.maximum(np.asarray(v_vec, dtype=float), self.V_FLOOR)
        return np.array(
            [
                self.drag_force(z, v[i], self.M_nodes[i], Tb)
                for i in range(self.n_bins)
            ]
        )

    def drag_acceleration_bins(self, z, v_vec, Tb, F=None):
        """
        Per-bin deceleration a_i = F_DF(M_i, v_i) / M_i. [GeV]

        Newton's second law, applied to each mass individually:

            a_i = 4 pi G^2 M_i rho_b(z) / v_i^2 * I(Mach_i)   ~  M_i

        There is NO `frac` and NO mass-function weighting here.  The
        deceleration of an individual MACHO cannot depend on how abundant
        other MACHOs are, nor on the shape of the rest of the mass function.
        (The previous population-averaged form, a = frac * int psi F/M dM,
        is the deceleration of the DM fluid CENTRE OF MASS -- a correct
        quantity, but the wrong one to feed back into F_DF, which needs the
        MACHO's own velocity.  The smooth DM feels no drag at all.)

        NOTE: with baryon_backreaction enabled, the RELATIVE-velocity EOM
        additionally carries the shared baryon-recoil term (one scalar for
        all bins; see recoil_acceleration).  frac enters the dynamics there
        and only there; this method remains the single-MACHO deceleration
        F_i / M_i.

        Parameters
        ----------
        z : float
            Redshift.
        v_vec : ndarray, shape (n_bins,)
            Per-bin relative velocities [0].
        Tb : float
            Baryon temperature [GeV].
        F : ndarray, optional
            Pre-computed drag forces from drag_forces(); supplied by
            drag_and_heating() to avoid evaluating the Coulomb logarithm
            twice per ODE right-hand side.

        Returns
        -------
        ndarray, shape (n_bins,)
            Deceleration magnitudes [GeV].
        """
        if F is None:
            F = self.drag_forces(z, v_vec, Tb)
        return F / self.M_nodes

    # ------------------------------------------------------------------
    # Baryon bulk recoil  (momentum conservation; shared by all bins)
    # ------------------------------------------------------------------

    def recoil_acceleration(self, z, F):
        """
        Baryon bulk-recoil contribution to d v_rel / dt. [GeV]

        The momentum each MACHO loses to its wake is gained by the baryon
        fluid, which accelerates toward the MACHO frame; the RELATIVE
        velocity therefore decays faster than the MACHO alone decelerates:

            du_b/dt = (1/rho_b) * sum_j n_j F_j
                    = (frac * rho_DM / rho_b) * sum_j w_j psi_j F_j ,

        with node number densities n_j = frac * rho_DM * psi_j * w_j.  This
        is ONE scalar added to every bin's deceleration: all bins push the
        same baryon fluid, and the streaming direction is coherent across
        the mass function.  Monochromatic limit:

            dv/dt = -H*v - (1 + frac * rho_DM/rho_b) * F/M ,

        i.e. a factor (1 + 5.35*frac) faster relative-velocity decay.
        Absent from arXiv:2512.00169 (their Eq. 4); negligible for
        frac <~ 0.05.  The heating rate Q is deliberately NOT modified:
        the dissipated power is sum_i n_i F_i v_i regardless of how the
        lost momentum is split between the two fluids.

        Parameters
        ----------
        z : float
            Redshift.
        F : ndarray, shape (n_bins,)
            Per-bin drag forces from drag_forces() [GeV^2].

        Returns
        -------
        float
            Recoil deceleration of the relative velocity [GeV].
        """
        rho_dm_z = omega_dm * rho_c0 * (1.0 + z) ** 3
        rho_b_z = rho_b0 * (1.0 + z) ** 3
        return (self.frac * rho_dm_z / rho_b_z) * float(
            np.sum(self.w_nodes * self.psi_nodes * np.asarray(F))
        )

    # ------------------------------------------------------------------
    # Heating rate  (frac and psi enter here and in recoil_acceleration)
    # ------------------------------------------------------------------

    def heating_rate_bins(self, z, v_vec, Tb, F=None):
        """
        Dynamical friction heating rate per baryon, Q. [GeV^2]

        Volumetric energy dissipation, with each mass evaluated at its own
        current velocity:

            dE/dV/dt = frac * rho_dm(z) * int dM/M psi(M) v(M) F_DF(M, v(M))
                     = frac * rho_dm(z) * int dlnM psi(M) v(M) F_DF(M, v(M))
                    ~= frac * rho_dm(z) * sum_i w_i psi(M_i) v_i F_i

        The substitution dM = M dlnM cancels the 1/M of the number density
        dn/dM = frac rho_dm psi(M) / M, which is why no explicit 1/M appears
        in the sum.  Then

            Q = (dE/dV/dt) / n_b(z)

        With the monochromatic convention w_1 = 1, psi_1 = 1/M_0 this reduces
        exactly to Q = (rho_dm/M_0) * frac * v * F_DF / n_b.

        Parameters
        ----------
        z : float
            Redshift.
        v_vec : ndarray, shape (n_bins,)
            Per-bin relative velocities [0].
        Tb : float
            Baryon temperature [GeV].
        F : ndarray, optional
            Pre-computed drag forces from drag_forces().

        Returns
        -------
        float
            Q [GeV^2].
        """
        v = np.maximum(np.asarray(v_vec, dtype=float), self.V_FLOOR)
        if F is None:
            F = self.drag_forces(z, v, Tb)

        rho_dm_z = omega_dm * rho_c0 * (1.0 + z) ** 3

        Qvol = (
            rho_dm_z
            * self.frac
            * np.sum(self.w_nodes * self.psi_nodes * v * F)
        )
        return Qvol / n_b(z)

    def drag_and_heating(self, z, v_vec, Tb):
        """
        Relative-velocity deceleration a_i and heating Q from a SINGLE
        evaluation of the drag forces. [GeV], [GeV^2]

        The Coulomb logarithm is the expensive part of the right-hand side;
        computing F_DF once and sharing it between the dynamics and the energy
        budget halves the cost of every solver step.

        The returned a_i is the deceleration of the RELATIVE velocity: the
        single-MACHO term F_i/M_i plus, when baryon_backreaction is True,
        the shared baryon bulk-recoil term from recoil_acceleration().

        Returns
        -------
        (a_vec, Q) : (ndarray shape (n_bins,), float)
        """
        v = np.maximum(np.asarray(v_vec, dtype=float), self.V_FLOOR)
        F = self.drag_forces(z, v, Tb)
        a_vec = self.drag_acceleration_bins(z, v, Tb, F=F)
        if self.baryon_backreaction:
            a_vec = a_vec + self.recoil_acceleration(z, F)
        Q = self.heating_rate_bins(z, v, Tb, F=F)
        return a_vec, Q


# ============================================================================
# Coupled ODEs:  X_e, T_M, v_1 ... v_N
# ============================================================================


def evolution(z, Y, sf: StarFormation, macho: MACHOHeating):
    """
    RHS of the coupled ODE system dY/dz.

    State vector Y = [X_e, T_M, v_1, ..., v_N]   (length 2 + macho.n_bins):
      Y[0]  : ionization fraction X_e                       [0]
      Y[1]  : matter (baryon) temperature T_M               [GeV]
      Y[2:] : per-bin MACHO-baryon relative velocities v_i  [0]

    For the monochromatic model n_bins = 1 and the state vector is the
    familiar [X_e, T_M, v_rel].

    All 2 + N components are advanced SIMULTANEOUSLY by the solver with a
    single common step: there is no operator splitting.  The bins are coupled
    through the gas,

        {v_i} -> Q -> dT_M/dz -> T_M -> c_s -> Mach_i -> I -> F_i -> a_i

    i.e. every mass bin heats the same reservoir, and the hotter gas raises
    c_s for everyone, suppressing every bin's Mach number and hence its drag —
    and, when macho.baryon_backreaction is True, also through the shared
    baryon-recoil term in dv_i/dz (every bin pushes the same baryon fluid).
    The system is therefore NOT linear in the bins: running N single-mass
    calculations and adding the resulting Delta T_M is not equivalent.

    Sign convention
    ---------------
    Integrated from z_start (high) to z_end (low), so dz < 0.
    All signs derived via dt/dz = -1 / (H * (1+z)).

    dv_i/dz: both the adiabatic (Hubble) and dynamical-friction terms are
    positive, because a physical deceleration dv/dt < 0 picks up a minus sign
    from dt/dz < 0.  With dz < 0 both terms correctly reduce v_i.

    Parameters
    ----------
    z : float
        Current redshift.
    Y : array_like, shape (2 + n_bins,)
        Current state [X_e, T_M [GeV], v_1 ... v_N [0]].
    sf : StarFormation
        Configured StarFormation instance.
    macho : MACHOHeating
        Configured MACHOHeating instance.

    Returns
    -------
    dY : ndarray, shape (2 + n_bins,)
        Derivatives [dX_e/dz, dT_M/dz, dv_1/dz ... dv_N/dz].
    """
    if not isinstance(sf, StarFormation):
        raise TypeError(
            f"sf must be a StarFormation instance, got {type(sf).__name__}."
        )
    if not isinstance(macho, MACHOHeating):
        raise TypeError(
            f"macho must be a MACHOHeating instance, "
            f"got {type(macho).__name__}."
        )
    if len(Y) != 2 + macho.n_bins:
        raise ValueError(
            f"State vector length {len(Y)} does not match 2 + n_bins = "
            f"{2 + macho.n_bins}."
        )

    dY = np.zeros_like(Y)

    X_e = Y[0]
    T_M = Y[1]
    # Clip once, here, so the RHS is a well-defined function of the state even
    # if the stiff solver takes a trial step to a tiny or negative velocity.
    v_vec = np.maximum(Y[2:], macho.V_FLOOR)

    T_cmb = T_CMB(z)
    H = H_z(z)
    n_bz = n_b(z)
    rho_cmb = rho_CMB(z)

    # ----------------------------------------------------------------
    # Peebles C_H coefficient
    # ----------------------------------------------------------------
    kh = rec_KH(z)
    n_H = (1.0 - Y_p) * n_bz  # hydrogen number density [GeV^3]

    C_H = (1.0 + kh * Lambda_2s * (1.0 - X_e) * n_H) / (
        1.0 + kh * (Lambda_2s + rec_beta(T_cmb)) * (1.0 - X_e) * n_H
    )

    # ----------------------------------------------------------------
    # Compton (Thomson) coupling coefficient  [GeV^-1]
    # ----------------------------------------------------------------
    thom_coeff = (X_e / (1.0 + X_e + fHe)) * (8.0 * sigma_T) / (3.0 * m_e)

    # ----------------------------------------------------------------
    # Star-formation X-ray terms (delegated to StarFormation instance)
    # ----------------------------------------------------------------
    dT_xray = sf.xray_heating_rate(z)  # [GeV]
    dXe_xray = sf.xray_ionization_rate(z)  # [0]

    # ----------------------------------------------------------------
    # MACHO drag + heating  (single evaluation of F_DF, shared by both)
    # ----------------------------------------------------------------
    a_vec, Q = macho.drag_and_heating(z, v_vec, T_M)

    # ----------------------------------------------------------------
    # dX_e / dz
    # ----------------------------------------------------------------
    recomb = rec_alpha(T_M) * n_H * X_e**2
    ioniz = rec_beta(T_cmb) * (1.0 - X_e) * rec_BH(T_cmb)

    dY[0] = (C_H / (H * (1.0 + z))) * (recomb - ioniz) - dXe_xray

    # ----------------------------------------------------------------
    # dT_M / dz
    # ----------------------------------------------------------------
    adiabatic = 2.0 * T_M / (1.0 + z)
    compton = thom_coeff * rho_cmb / ((1.0 + z) * H) * (T_M - T_cmb)
    # DF heat is shared by ALL free particles (H, He, e-): the heat capacity
    # is (3/2) n_tot with n_tot = n_H * (1 + fHe + X_e).  heating_rate_bins()
    # returns Q = Q_vol / n_b, so undo that with Q * n_bz = Q_vol and divide
    # by n_tot.  Written via n_H (= (1 - Y_p) * n_bz above), this is
    # independent of the convention used for n_b(z) in Cosmology_term.  The
    # Compton term already carries the matching 1/(1 + fHe + X_e) inside
    # thom_coeff; X-ray heating needs no factor because x_ray_norm is a
    # phenomenological dT/dz normalisation, not an energy bookkeeping.
    macho_heat = (
        (2.0 / 3.0) * (Q * n_bz) / (n_H * (1.0 + fHe + X_e) * H * (1.0 + z))
    )

    dY[1] = adiabatic + compton - macho_heat - dT_xray

    # ----------------------------------------------------------------
    # dv_i / dz  --  one equation per mass node
    #
    # Physical EOM:  dv_i/dt = -H*v_i - F_DF(M_i, v_i)/M_i - a_recoil
    #   a_recoil = (frac * rho_DM/rho_b) * sum_j w_j psi_j F_j  is the
    #   baryon bulk recoil (momentum conservation), one scalar shared by
    #   every bin; drag_and_heating() folds it into a_vec when
    #   macho.baryon_backreaction is True.  Set it False to reproduce
    #   Eq. (4) of arXiv:2512.00169, which omits it.
    # Chain rule:    dv_i/dz = (dv_i/dt) * (-1/(H*(1+z)))
    #                        = v_i/(1+z) + a_i / (H*(1+z))
    # ----------------------------------------------------------------
    dY[2:] = v_vec / (1.0 + z) + a_vec / (H * (1.0 + z))

    return dY
