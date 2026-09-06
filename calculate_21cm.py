"""
calculate_21cm.py
=================
Computes the 21-cm global brightness temperature T_21(z) from the IGM
thermal and ionization histories produced by calculate_TM_Xe.py (multi-bin
MACHO treatment).

T_21 depends only on (z, X_e, T_M).  The per-bin MACHO velocities enter
solely through the thermal history, so NONE of the 21-cm formulae below
change relative to the single-velocity version -- only the plumbing does.

Physics
-------
    T_21 = 27 * x_HI * [(1-Y_p)/0.76] * [Omega_b h^2 / 0.023]
           * sqrt[0.15*(1+z) / (10*Omega_m*h^2)]
           * (1 - T_CMB/T_s)                                    [mK]

Spin temperature:
    T_s^{-1} = (T_CMB^{-1} + x_alpha/T_alpha + x_c/T_M)
               / (1 + x_alpha + x_c)

with T_alpha = T_M (Wouthuysen-Field approximation, valid for T_M > 1 K).

Collisional coupling x_c:
    x_c = T_* * n_H * [(1-X_e)*k_HH + X_e*k_eH] / (A_10 * T_CMB)

Lyman-alpha coupling x_alpha (SFRD-based):
    J_alpha = f_alpha * (1/4pi) * z^2 / m_H
              * integral_{z}^{z_max} emissivity(z') * SFRD(z') / H(z') dz'
    J_0(z)  = 5.54e-8 * z    [Mittal & Kulkarni 2021, Eq.24]
    x_alpha = J_alpha / J_0   for z < z_star, else 0

All quantities in GeV natural units.  Output T_21 in mK.

Unit translations from snippet (SI) → GeV
------------------------------------------
k_HH, k_eH  [m^3/s]   →  [GeV^{-2}]   via * Meter**3/Sec
T_*         [K]        →  [GeV]        = omega_21/k_B
A_10        [s^{-1}]   →  [GeV]        = 2.85e-15 * Sec**(-1)
n_H(z)      [GeV^3]                    = n_H0*(1+z)^3
SFRD(z)     [GeV^4]                    from StarFormation.sfrd(z)
H(z)        [GeV]                      = H_z(z)
J_alpha     dimensionless ratio; J_0 dimensionless → x_alpha dimensionless
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import integrate as scint
from tqdm import tqdm

from calculate_tm_xe import (
    N_BINS,
    OUTPUT_DIR,
    PARAM_GRID,
    Z_END,
    Z_START,
    make_filestem,
    make_label,
    run_evolution,
)
from cosmology_terms import (
    T_CMB,
    H_z,
    Y_p,
    h,
    m_H,
    n_H0,
    omega_b,
    omega_m,
)
from thermal_evolution import MACHOHeating, StarFormation
from units import (
    Kelv,
    Meter,
    Sec,
    eV,
    q_e,
)

# ============================================================================
# 21-cm physical constants  (GeV framework)
# ============================================================================

f_alpha_value = 10.0

# T_* = h*nu_21 / k_B  ~ 0.0682 K
Tstar = 0.0682 * Kelv  # [GeV]

# Einstein A coefficient for 21-cm hyperfine transition
A10 = 2.85e-15 * Sec ** (-1)  # [GeV]


def n_H(z):
    """Neutral hydrogen number density [GeV^3]."""
    return n_H0 * (1.0 + z) ** 3


# ============================================================================
# Collisional rate coefficients  (fitting functions, converted from SI m^3/s)
# ============================================================================


def k_HH(T_K):
    """
    H-H spin-flip rate coefficient [GeV^{-2}].
    Fitting function from Zygelman (2005).

    Parameters
    ----------
    T_K : float or array
        Gas temperature [K]  (= T_M / Kelv).
    """
    k_si = (
        3.1e-17 * (T_K / Kelv) ** 0.357 * np.exp(-32.0 * Kelv / T_K)
    )  # [m^3/s]
    return k_si * Meter**3 / Sec  # [GeV^{-2}]


def k_eH(T_K):
    """
    e-H spin-flip rate coefficient [GeV^{-2}].
    Fitting function from Furlanetto & Furlanetto (2007).

    Parameters
    ----------
    T_K : float or array
        Gas temperature [K]  (= T_M / Kelv).
    """
    log10T = np.log10(T_K / Kelv)
    k_si = 10.0 ** (
        -15.607 + 0.5 * log10T * np.exp(-np.abs(log10T) ** 4.5 / 1800.0)
    )

    return k_si * Meter**3 / Sec  # [GeV^{-2}]


# ============================================================================
# Collisional coupling  x_c  [dimensionless]
# ============================================================================


def x_coll(z, X_e, T_M):
    """
    Collisional coupling coefficient x_c [0].

    x_c = T_* * n_H * [(1-X_e)*k_HH + X_e*k_eH] / (A_10 * T_CMB)

    Unit check:
      [GeV] * [GeV^3] * [GeV^{-2}] / ([GeV] * [GeV]) = [0]  ✓

    Parameters
    ----------
    z   : float or array   Redshift.
    X_e : float or array   Free electron fraction [0].
    T_M : float or array   Gas temperature [GeV].
    """
    T_K = T_M  # [K] (dimensionless)
    nH = n_H(z)  # [GeV^3]
    T_cmb = T_CMB(z)  # [GeV]

    kappa = (1.0 - X_e) * k_HH(T_K) + X_e * k_eH(T_K)  # [GeV^{-2}]
    return (Tstar * nH * kappa) / (A10 * T_cmb)  # [0]


# ============================================================================
# Lyman-alpha coupling  x_alpha  (SFRD-based, from snippet)
# ============================================================================


def x_alpha(z, sf: StarFormation, f_alpha=f_alpha_value, z_star=60.0):
    """
    Lyman-alpha (Wouthuysen-Field) coupling coefficient x_alpha [0].

    Follows the SFRD-based integral from the provided snippet, translated
    into the GeV framework:

        J_alpha(z) = f_alpha / (4*pi) * z^2 / m_H
                     * integral_{z}^{z_max} N_alpha(z'/z)
                       * SFRD(z') / H(z') dz'

        J_0(z) = 5.54e-8 * z          [Mittal & Kulkarni 2021, Eq.24]

        x_alpha = J_alpha / J_0        for z < z_star, else 0

    where N_alpha(u) = 2902.91 * (0.75*u)^{-0.86} is the dimensionless
    number of Lyman-alpha photons emitted per baryon per unit frequency
    (fitting function from the snippet), and z_max = (32/27) * z is the
    upper redshift limit set by the Lyman-alpha photon horizon.

    Unit analysis:
      J_alpha ~ [0] * z^2 * [GeV^{-1}] * [GeV^4] / [GeV] * dz
              = [0] * z^2 * [GeV^2] * dz
      J_0     = 5.54e-8 * z  [dimensionless by convention of Mittal+2021]
      x_alpha = J_alpha / J_0  [0]  ✓

    Parameters
    ----------
    z       : float or array   Redshift.
    sf      : StarFormation    Instance used for SFRD(z').
    f_alpha : float            Lyman-alpha efficiency factor. Default: 1.0.
    z_star  : float            Redshift below which sources are active.
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    xa_out = np.zeros_like(z_arr)

    for idx, zi in enumerate(z_arr):
        if zi >= z_star:
            xa_out[idx] = 0.0
            continue

        # Integration range: z' from z to z_max = (32/27)*z
        z_max = (32.0 / 27.0) * zi
        z_integ = np.linspace(zi, z_max, 30)

        # Emissivity: N_alpha = number of Ly-alpha photons per baryon
        # N_alpha(z'/z) = 2902.91 * (0.75 * z'/z)^{-0.86}   [GeV^-1]
        N_alpha = 2902.91 * (0.75 * (z_integ + 1) / (zi + 1)) ** (-0.86) / eV

        # SFRD(z') [GeV^5],  H(z') [GeV]
        sfrd_arr = np.array([sf.sfrd(zp) for zp in z_integ])  # [GeV^5]
        H_arr = H_z(z_integ)  # [GeV]

        # Integrand: N_alpha [GeV-1] * SFRD / H -->  [GeV^3]
        integrand = N_alpha * sfrd_arr / H_arr

        # J_alpha = f_alpha / (4*pi) * z^2 / m_H * integral  [GeV^2]
        J_alpha = (
            f_alpha
            / q_e
            / (4.0 * np.pi)
            * (zi + 1) ** 2
            / 1.22
            / m_H
            * scint.trapezoid(integrand, z_integ)
        )

        # J_0 = 5.54e-8 * z  [GeV^2 by Mittal & Kulkarni 2021 Eq.24]
        J_0 = 5.54e-8 * (zi + 1) * Meter**-2

        xa_out[idx] = J_alpha / J_0

    return xa_out if xa_out.size > 1 else float(xa_out[0])


# ============================================================================
# Spin temperature  [GeV]
# ============================================================================


def spin_temperature(
    z, X_e, T_M, sf: StarFormation, f_alpha=f_alpha_value, z_star=60.0
):
    """
    Spin temperature T_s [GeV].

    T_s^{-1} = (T_CMB^{-1} + x_alpha/T_alpha + x_c/T_M)
               / (1 + x_alpha + x_c)

    T_alpha = T_M  (Wouthuysen-Field approximation, valid for T_M > 1 K).

    Parameters
    ----------
    z       : float or array   Redshift.
    X_e     : float or array   Free electron fraction [0].
    T_M     : float or array   Gas temperature [GeV].
    sf      : StarFormation    For Lyman-alpha coupling.
    f_alpha : float            Lyman-alpha efficiency.
    z_star  : float            Redshift where Lyman-alpha sources switch on.
    """
    T_cmb = T_CMB(z)
    xc = x_coll(z, X_e, T_M)
    xa = x_alpha(z, sf, f_alpha, z_star)

    numerator = 1.0 + xa + xc
    denominator = 1.0 / T_cmb + xa / T_M + xc / T_M

    return numerator / denominator  # [GeV]


# ============================================================================
# 21-cm brightness temperature  [mK]
# ============================================================================


def T21_mK(z, X_e, T_M, sf: StarFormation, f_alpha=f_alpha_value, z_star=60.0):
    """
    21-cm differential brightness temperature T_21 [mK].

    T_21 = 27 * x_HI * [(1-Y_p)/0.76] * [Omega_b h^2/0.023]
           * sqrt[0.15*(1+z)/(10*Omega_m*h^2)]
           * (1 - T_CMB/T_s)

    Parameters
    ----------
    z       : float or array   Redshift.
    X_e     : float or array   Free electron fraction [0].
    T_M     : float or array   Gas temperature [GeV].
    sf      : StarFormation    For Lyman-alpha coupling.
    f_alpha : float            Lyman-alpha efficiency.
    z_star  : float            Redshift where Lyman-alpha sources switch on.

    Returns
    -------
    float or array
        T_21 [mK].
    """
    x_HI = 1.0 - X_e
    T_s = spin_temperature(z, X_e, T_M, sf, f_alpha, z_star)
    T_cmb = T_CMB(z)

    prefactor = (
        27.0
        * x_HI
        * ((1.0 - Y_p) / 0.76)
        * (omega_b * h**2 / 0.023)
        * np.sqrt(0.15 * (1.0 + z) / (10.0 * omega_m * h**2))
    )
    return prefactor * (1.0 - T_cmb / T_s)  # [mK]


# ============================================================================
# Main driver
# ============================================================================


def main():
    keys = list(PARAM_GRID.keys())
    value_lists = [PARAM_GRID[k] for k in keys]
    all_combos = list(itertools.product(*value_lists))
    n_total = len(all_combos)

    print(f"Total parameter combinations: {n_total}")
    results = []

    for combo in tqdm(all_combos, desc="Parameter sweep", unit="run"):
        params = dict(zip(keys, combo, strict=False))

        sf = StarFormation(
            f_star=params["f_star"],
            T_vir4=params["T_vir4"],
            f_X=params["f_X"],
        )
        macho = MACHOHeating(
            Mass=params["Mass"],
            model=params["model"],
            frac=params["frac"],
            M_c=params["M_c"],
            M_min=params["M_min"],
            M_max=params["M_max"],
            eos=params["eos"],
            n_bins=N_BINS,
        )

        res = run_evolution(sf, macho, v_rel_ini=params["v_rel_ini"])
        if not res["success"]:
            print(f"\n  [WARN] {make_label(params)}: {res.get('message','')}")
            continue

        z_arr = res["z"]
        Xe_arr = res["Xe"]
        TM_arr = res["TM_K"] * Kelv  # [GeV]

        T21_arr = T21_mK(z_arr, Xe_arr, TM_arr, sf)

        res["T21_mK"] = T21_arr
        res["params"] = params
        res["label"] = make_label(params)
        results.append(res)

        stem = make_filestem(params)
        np.save(
            OUTPUT_DIR / f"{stem}_21cm.npy",
            np.array([z_arr, Xe_arr, res["TM_K"], res["vrel"], T21_arr]),
        )

    print(f"\nSuccessful runs: {len(results)} / {n_total}")
    _print_T21_at_redshifts(results)
    _plot_T21(results)
    _plot_spin_temp(results)
    _plot_xc_xa(results)


# ============================================================================
# Tabulate T_21 at selected redshifts
# ============================================================================


def _print_T21_at_redshifts(results, z_probes=(17.2, 89, 225, 550, 700, 900)):
    """
    Print T_21 [mK] at a set of specified redshifts for every run.

    Uses linear interpolation on the (z_arr, T21_arr) arrays stored in each
    result dict.  If a probe redshift lies outside the computed range the
    entry is reported as NaN with a warning flag.

    Parameters
    ----------
    results  : list of dict   Output of the main parameter sweep.
    z_probes : tuple of float Redshifts at which to report T_21.
    """
    z_probes = list(z_probes)

    # Header
    header_z = "  ".join(f"z={zp:<7g}" for zp in z_probes)
    sep = "-" * (len("Run") + 4 + len(header_z) + 2)
    print("\n" + sep)
    print("  T_21 [mK] at selected redshifts")
    print(sep)
    print(f"  {'Run':<45s}  {header_z}")
    print(sep)

    for res in results:
        z_arr = res["z"]  # may be ascending or descending
        T21_arr = res["T21_mK"]
        label = res["label"]

        # np.interp requires x to be increasing
        if z_arr[0] > z_arr[-1]:
            z_arr = z_arr[::-1]
            T21_arr = T21_arr[::-1]

        z_min, z_max = z_arr[0], z_arr[-1]
        values = []
        for zp in z_probes:
            if zp < z_min or zp > z_max:
                values.append("  OORange ")
            else:
                t21 = float(np.interp(zp, z_arr, T21_arr))
                values.append(f"{t21:+10.3f} ")

        row = "  ".join(values)
        print(f"  {label:<45s}  {row}")

    print(sep + "\n")


# ============================================================================
# Plotting helpers
# ============================================================================


def _plot_T21(results):
    fig, ax = plt.subplots()
    colors = cm.jet(np.linspace(0, 1, max(len(results), 1)))
    for res, col in zip(results, colors, strict=False):
        ax.semilogx(
            res["z"], res["T21_mK"], color=col, lw=1.5, label=res["label"]
        )
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Redshift $z$", fontsize=15)
    ax.set_ylabel(r"$T_{21}$ [mK]", fontsize=15)
    ax.set_xlim(Z_END, Z_START)
    ax.legend(fontsize=13, ncol=1, loc="lower right")
    # ax.set_title("21-cm brightness temperature", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "T21_evolution.pdf", dpi=300)
    # plt.show()


def _plot_spin_temp(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.plasma(np.linspace(0, 1, max(len(results), 1)))
    z_ref = np.linspace(Z_END, Z_START, 500)
    for res, col in zip(results, colors, strict=False):
        Ts_arr = (
            spin_temperature(
                res["z"],
                res["Xe"],
                res["TM_K"] * Kelv,
                StarFormation(
                    f_star=res["params"]["f_star"],
                    T_vir4=res["params"]["T_vir4"],
                    f_X=res["params"]["f_X"],
                ),
            )
            / Kelv
        )
        ax.semilogy(res["z"], Ts_arr, color=col, lw=1.5, label=res["label"])
    ax.semilogy(
        z_ref, T_CMB(z_ref) / Kelv, "k--", lw=1.5, label=r"$T_\mathrm{CMB}$"
    )
    ax.set_xlabel("Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$T_s$ [K]", fontsize=13)
    ax.set_xlim(Z_END, Z_START)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("Spin temperature", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Ts_evolution.pdf", dpi=250)
    # plt.show()


def _plot_xc_xa(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, max(len(results), 1)))
    for res, col in zip(results, colors, strict=False):
        sf_tmp = StarFormation(
            f_star=res["params"]["f_star"],
            T_vir4=res["params"]["T_vir4"],
            f_X=res["params"]["f_X"],
        )
        z_arr = res["z"]
        xc_arr = x_coll(z_arr, res["Xe"], res["TM_K"] * Kelv)
        xa_arr = x_alpha(z_arr, sf_tmp)
        ax.semilogy(
            z_arr, xc_arr, color=col, lw=1.5, label=rf"$x_c$: {res['label']}"
        )
        ax.semilogy(
            z_arr,
            np.maximum(xa_arr, 1e-30),
            color=col,
            lw=1.5,
            ls="dashed",
            label=rf"$x_\alpha$: {res['label']}",
        )
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Redshift $z$", fontsize=13)
    ax.set_ylabel("Coupling coefficients", fontsize=13)
    ax.set_xlim(Z_END, min(Z_START, 200))
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(r"$x_c$ and $x_\alpha$", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coupling_evolution.pdf", dpi=250)
    # plt.show()


if __name__ == "__main__":
    main()
