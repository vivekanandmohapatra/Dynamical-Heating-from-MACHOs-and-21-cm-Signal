"""
calculate_TM_Xe.py
==================
Driver script for IGM thermal and ionization evolution in the presence of
MACHOs, with the MULTI-BIN (per-mass velocity) treatment.

Computes X_e(z), T_M(z), and v_i(z) by integrating the coupled ODEs in
Thermal_evolution.py over a user-defined parameter grid spanning:
  - MACHO mass (Mass)
  - Characteristic mass (M_c) for extended distributions
  - Initial relative velocity (v_rel_ini)
  - MACHO dark-matter fraction (frac)
  - Mass function model (model)
  - Virial temperature threshold (T_vir4)
  - Star-formation efficiency (f_star)
  - X-ray ionization efficiency (f_X)

Results are saved to NumPy .npy files and plotted (T_M, X_e and v_rel vs
redshift), overlaid against a RecFast reference solution.

Multi-bin treatment
-------------------
The mass function is discretised into N_BINS log-spaced nodes M_i.  The ODE
state vector is

    Y = [X_e, T_M, v_1, ..., v_N]

Every mass node starts at the same v_bc (the rms streaming velocity is
mass-independent) but decelerates at its OWN rate F_DF(M_i, v_i)/M_i, so the
velocity histories diverge.  frac and psi(M) enter only the heating rate.
For the monochromatic model N = 1 and the state vector reduces to the
familiar [X_e, T_M, v_rel].

Because there is no longer a single v_rel, the scalar "vrel" reported in the
plots and in the .npy files is a SUMMARY of the velocity family (see
VREL_SUMMARY below).  It reduces exactly to the single velocity for the
monochromatic model.  The full per-bin matrix v_i(z) is saved separately;
that is the object that actually carries the physics.

Units
-----
All internal quantities use natural units (GeV-based) defined in units.py.
Temperatures are converted to Kelvin only for output and plotting.
"""

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.integrate import solve_ivp
from tqdm import tqdm

from cosmology_terms import (
    T_CMB,
)
from thermal_evolution import MACHOHeating, StarFormation, evolution
from units import (
    Kelv,
    Kmps,
    M_s,
)

# ============================================================================
# Output directory
# ============================================================================

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Redshift grid
# ============================================================================

Z_START = 1060  # Starting redshift (high-z, tight coupling)
Z_END = 10  # Ending redshift (no reionization module included)
Z_NPTS = 10_000  # Number of evaluation points

z_grid = np.linspace(Z_END, Z_START, Z_NPTS)  # ascending, for t_eval flip


# ============================================================================
# Initial conditions at z = Z_START
# ============================================================================

# X_e at z_start: fully ionized hydrogen
Xe_ini = 0.097638161

# T_M at z_start: tight-coupling approximation valid for 1000 < z < 1500
# T_M ~ T_CMB * [1 + H / gamma_C]^{-1}
# _gamma_c_ini = ((8.0 * sigma_T * rho_CMB(Z_START))
#                / ((1.0 + Xe_ini + fHe) * 3.0 * m_e))
# TM_ini = T_CMB(Z_START) * (1.0 + H_z(Z_START) / _gamma_c_ini)**(-1)
TM_ini = 2891.9355 * Kelv


# ============================================================================
# Multi-bin settings
# ============================================================================

# Number of log-spaced mass-quadrature nodes for extended mass functions.
# Ignored (forced to 1) for the monochromatic model.  Convergence check:
# re-run with N_BINS = 96 and confirm T_M(z) does not move.
N_BINS = 48

# How to collapse the family {v_i(z)} into the single "vrel" curve that is
# plotted and stored in the .npy (for backwards compatibility with the
# monochromatic output format).  All three reduce to v_1 when n_bins == 1.
#   "mass"    : weights w_i M_i psi_i   -- velocity of the average unit of
#               MACHO *mass*.  psi is a mass PDF, so this is the natural one.
#   "number"  : weights w_i psi_i       -- number-weighted; dominated by the
#               light end, which contributes almost no heating.
#   "heating" : weights w_i psi_i v_i F_i -- what the energy integral actually
#               sees.  Time-dependent weights; the most physically meaningful
#               summary, but not a simple average of a conserved population.
VREL_SUMMARY = "mass"

# Save the full per-bin velocity matrix v_i(z) and the bin masses M_i.
# This is the single most useful diagnostic the multi-bin scheme buys: it
# tells you WHICH mass bin went subsonic WHEN.  Set to False to write only
# the four legacy arrays.
SAVE_VBINS = True


# ============================================================================
# Parameter grid
# ============================================================================
# Each entry is a list of values to sweep.  All combinations are iterated.
# To run a single value, wrap it in a one-element list.
# color=col,
PARAM_GRID = {
    # MACHO / PBH mass [Solar Mass]
    "Mass": 1 * 10 ** np.array([0]),
    # Characteristic mass M_c for log_normal / critical_collapse [Solar Mass]
    # (ignored when model=None)
    "M_c": 1 * 10 ** np.array([7]),
    # Lower mass integration limit for extended distributions [Solar Mass]
    # (ignored when model=None)
    "M_min": [1],  # None → default 1e-2 M_sun in MACHOHeating
    # Upper mass integration limit for extended distributions [Solar Mass]
    # (ignored when model=None)
    "M_max": [1e8],  # None → default 1e7 M_sun in MACHOHeating
    # Initial relative velocity v_rel at z = Z_START [units of c]
    # Typical RMS value at z~1000 from Tseliakhovich & Hirata (2010): ~1e-4
    "v_rel_ini": [30 * Kmps],
    # MACHO DM fraction
    "frac": [0.04],
    # Mass function model: None = monochromatic
    # NOTE: if 'power_law' is included, eos must be set to a value in [-1/3, 1]
    "model": [
        "Log-normal"
    ],  # "Monochromatic" 'Log-normal', 'Critical collapse'],
    # Equation-of-state parameter w for power_law model [-1/3, 1]
    # Must be provided (non-None) when model='power_law'; ignored otherwise.
    "eos": np.linspace(-1 / 3, 1, 1),
    # Minimum virial temperature [units of 1e4 K]
    "T_vir4": [4],
    # Star-formation efficiency
    "f_star": [0.1],
    # X-ray ionization efficiency
    "f_X": [1.22],
}


# ============================================================================
# ODE solver settings
# ============================================================================

SOLVER_METHOD = "Radau"  # stiff solver — appropriate for recombination ODEs
RTOL = 1e-8
ATOL = 1e-10


# ============================================================================
# RecFast reference
# ============================================================================

# Optional RecFast++ reference curve, overplotted for comparison.  The file
# is not required to run the code: if it is absent the reference is simply
# omitted from the plots.
RECFAST_FILE = Path(__file__).with_name("standard.dat")


def load_recfast(path=RECFAST_FILE):
    """Load the RecFast++ reference history, if it is available.

    Parameters
    ----------
    path : pathlib.Path, optional
        Three-column whitespace-separated file of ``z``, ``X_e``, ``T_M``
        with ``T_M`` in Kelvin.  Defaults to ``standard.dat`` next to this
        module.

    Returns
    -------
    pandas.DataFrame or None
        The reference table, or ``None`` if ``path`` does not exist.
    """
    if not Path(path).is_file():
        warnings.warn(
            f"RecFast reference {path} not found; "
            "plots will omit the reference curve.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return pd.read_csv(path, header=None, names=["z", "Xe", "TM_K"])


recfast = load_recfast()


# ============================================================================
# Helper: build initial state vector
# ============================================================================


def build_y0(macho: MACHOHeating, v_rel_ini: float) -> np.ndarray:
    """
    Construct the initial state vector Y0 = [X_e, T_M, v_1 ... v_N].

    Every mass bin starts at the same v_rel_ini: the rms baryon-DM streaming
    velocity at recombination is mass-independent.

    Parameters
    ----------
    macho : MACHOHeating
        Configured instance (supplies n_bins).
    v_rel_ini : float
        Initial relative velocity at z = Z_START [units of c].

    Returns
    -------
    np.ndarray, shape (2 + macho.n_bins,)
    """
    return np.concatenate([[Xe_ini, TM_ini], macho.v0_vector(v_rel_ini)])


# ============================================================================
# Helper: collapse the per-bin velocities into one representative curve
# ============================================================================


def summarise_vrel(macho: MACHOHeating, v_bins: np.ndarray) -> np.ndarray:
    """
    Collapse v_i(z) into a single representative v_rel(z). [units of c]

    Parameters
    ----------
    macho : MACHOHeating
    v_bins : ndarray, shape (n_bins, n_z)
        Per-bin velocity histories [units of c].

    Returns
    -------
    ndarray, shape (n_z,)
    """
    if macho.n_bins == 1:
        return v_bins[0]

    w, psi, M = macho.w_nodes, macho.psi_nodes, macho.M_nodes

    if VREL_SUMMARY == "mass":
        wt = (w * M * psi)[:, None]
    elif VREL_SUMMARY == "number":
        wt = (w * psi)[:, None]
    elif VREL_SUMMARY == "heating":
        # Force ~ M^2 / v^2 * I; use the supersonic scaling M^2/v as a cheap,
        # monotone stand-in so the weights need no Coulomb-log evaluation.
        wt = (w * psi)[:, None] * (M**2)[:, None] / np.maximum(v_bins, 1e-30)
    else:
        raise ValueError(f"Unknown VREL_SUMMARY: {VREL_SUMMARY!r}")

    return (wt * v_bins).sum(axis=0) / wt.sum(axis=0)


# ============================================================================
# Helper: run a single ODE integration
# ============================================================================


def run_evolution(
    sf: StarFormation, macho: MACHOHeating, v_rel_ini: float
) -> dict:
    """
    Integrate the IGM evolution ODEs for one parameter combination.

    The full system [X_e, T_M, v_1 ... v_N] is advanced by a single stiff
    solve_ivp call: all 2 + N components share one step size, so the mass
    bins, the gas temperature and the ionization fraction evolve strictly
    simultaneously.  (The previous chunked integration with N_CHUNKS = 1 was
    a no-op; the scaffolding has been removed.)

    Parameters
    ----------
    sf : StarFormation
        Configured StarFormation instance.
    macho : MACHOHeating
        Configured MACHOHeating instance.
    v_rel_ini : float
        Initial relative velocity [units of c].

    Returns
    -------
    dict with keys:
        "z"       : redshift array (ascending, Z_END -> Z_START)
        "Xe"      : ionization fraction
        "TM_K"    : matter temperature [Kelvin]
        "vrel"    : representative relative velocity [km/s]  (see VREL_SUMMARY)
        "vbins"   : per-bin velocities, shape (n_bins, n_z) [km/s]
        "Mbins"   : quadrature node masses, shape (n_bins,) [Solar Mass]
        "success" : bool
    """
    t_eval = z_grid[::-1]  # descending z: Z_START -> Z_END

    sol = solve_ivp(
        evolution,
        [Z_START, Z_END],
        build_y0(macho, v_rel_ini),
        method=SOLVER_METHOD,
        t_eval=t_eval,
        args=(sf, macho),
        rtol=RTOL,
        atol=ATOL,
        dense_output=True,
    )

    if not sol.success:
        return {"success": False, "message": sol.message}

    v_bins = sol.y[2:]  # (n_bins, n_z) [c]
    v_rel = summarise_vrel(macho, v_bins)  # (n_z,)      [c]

    return {
        "z": sol.t[::-1],
        "Xe": sol.y[0][::-1],
        "TM_K": (sol.y[1] / Kelv)[::-1],
        "vrel": (v_rel / Kmps)[::-1],
        "vbins": v_bins[:, ::-1] / Kmps,
        "Mbins": macho.M_nodes / M_s,
        "success": True,
    }


# ============================================================================
# Helper: build a short human-readable label for a parameter combination
# ============================================================================


def make_label(params: dict) -> str:
    """
    Build a compact legend label from a parameter dictionary.

    Parameters
    ----------
    params : dict
        Keys: Mass, M_c, M_min, M_max, v_rel_ini, frac, model, eos,
              T_vir4, f_star, f_X.

    Returns
    -------
    str
        Label string.
    """
    model_str = params["model"] if params["model"] is not None else " "
    _exp = int(np.round(np.log10(params["Mass"])))
    _man = params["Mass"] / 10**_exp
    if abs(_man - 1.0) < 1e-9:
        _mass_str = rf"$10^{{{_exp}}}\,M_\odot$"
    else:
        _mass_str = rf"${_man:.4g}\times10^{{{_exp}}}\,M_\odot$"

    _exp1 = int(np.round(np.log10(params["M_c"])))
    _man1 = params["M_c"] / 10**_exp
    if abs(_man1 - 1.0) < 1e-9:
        _mass_str1 = rf"$10^{{{_exp1}}}\,M_\odot$"
    else:
        _mass_str1 = rf"${_man1:.4g}\times10^{{{_exp1}}}\,M_\odot$"
    label = (
        # f"M={_mass_str} "
        rf"$M_c$={_mass_str1} "
        # f"v0={params['v_rel_ini']/Kmps:.0e}Kmps "
        # f"f={params['frac']:.2f} "
        f"{model_str} "
        # f"Tv={params['T_vir4']:.1f} "
        # f"fs={params['f_star']:.2f}"
    )
    if params["model"] == "power_law" and params["eos"] is not None:
        label += f"w={params['eos']:.3g} "
    return label


# ============================================================================
# Helper: build a filesystem-safe filename stem
# ============================================================================


def make_filestem(params: dict) -> str:
    """
    Build a filesystem-safe filename stem from a parameter dictionary.

    Parameters
    ----------
    params : dict
        Keys: Mass, M_c, M_min, M_max, v_rel_ini, frac, model, eos,
              T_vir4, f_star, f_X.

    Returns
    -------
    str
    """
    model_str = params["model"] if params["model"] is not None else "mono"
    eos_str = f"_eos{params['eos']:.3g}" if params["eos"] is not None else ""
    return (
        (
            f"M{params['Mass']:.2e}_"
            f"Mc{params['M_c']:.2e}_"
            f"v{params['v_rel_ini']:.2e}_"
            f"frac{params['frac']:.2f}_"
            f"{model_str}{eos_str}_"
            f"Tv{params['T_vir4']:.1f}_"
            f"fs{params['f_star']:.3f}_"
            f"fX{params['f_X']:.2f}"
        )
        .replace("+", "")
        .replace("-", "n")
    )


# ============================================================================
# Main loop over parameter grid
# ============================================================================


def main():
    # Build all combinations from the parameter grid
    keys = list(PARAM_GRID.keys())
    value_lists = [PARAM_GRID[k] for k in keys]
    all_combos = list(itertools.product(*value_lists))
    n_total = len(all_combos)

    print(f"Total parameter combinations: {n_total}")

    # Collect results for plotting
    results = []

    for combo in tqdm(all_combos, desc="Parameter sweep", unit="run"):
        params = dict(zip(keys, combo, strict=False))

        # Instantiate physics objects for this combination
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

        # Report the fraction of the mass function actually captured inside
        # [M_min, M_max].  If this is not close to 1 the truncation is doing
        # real work and the result depends on the integration limits.
        if macho.model != "Monochromatic":
            tqdm.write(
                f"  [{params['model']}] n_bins={macho.n_bins}, "
                f"nodes {macho.M_nodes[0]/M_s:.3e} - "
                f"{macho.M_nodes[-1]/M_s:.3e} M_sun, "
                f"captured mass fraction = {macho.mass_norm:.4f}"
            )

        # Run ODE integration
        res = run_evolution(sf, macho, v_rel_ini=params["v_rel_ini"])

        if not res["success"]:
            print(
                f"\n  [WARN] Solver failed for {make_label(params)}: "
                f"{res.get('message', '')}"
            )
            continue

        res["params"] = params
        res["label"] = make_label(params)
        results.append(res)

        # Save individual run to .npy  (same four arrays as before)
        stem = make_filestem(params)
        np.save(
            OUTPUT_DIR / f"{stem}.npy",
            np.array([res["z"], res["Xe"], res["TM_K"], res["vrel"]]),
        )

        # Per-bin velocity history + node masses (extended models only)
        if SAVE_VBINS and macho.n_bins > 1:
            np.save(OUTPUT_DIR / f"{stem}_vbins.npy", res["vbins"])
            np.save(OUTPUT_DIR / f"{stem}_Mbins.npy", res["Mbins"])

    print(f"\nSuccessful runs: {len(results)} / {n_total}")

    # ========================================================================
    # Plotting
    # ========================================================================

    _plot_temperature(results)
    _plot_ionization(results)
    _plot_velocity(results)
    _plot_vbins(results)


# ============================================================================
# Plot: T_M vs redshift
# ============================================================================


def _plot_temperature(results: list):
    fig, ax = plt.subplots()

    colors = cm.jet(np.linspace(0, 1, len(results)))

    for res, col in zip(results, colors, strict=False):
        z = res["z"]
        ax.loglog(z, res["TM_K"], color=col, lw=1.5, label=res["label"])

    # CMB temperature reference
    z_ref = np.linspace(Z_END, Z_START, 500)
    ax.loglog(z_ref, T_CMB(z_ref) / Kelv, "k--", lw=1.5, label=r"$T_\gamma$")

    # RecFast reference (skipped when standard.dat is unavailable)
    if recfast is not None:
        ax.plot(
            recfast["z"],
            recfast["TM_K"],
            color="red",
            ls="--",
            lw=1.5,
            label="Standard",
        )

    ax.set_xlabel("Redshift $z$", fontsize=15)
    ax.set_ylabel(r"$T_{\rm gas}$ [K]", fontsize=15)
    ax.set_xlim(Z_END, Z_START)
    ax.set_ylim(2, 4e3)
    ax.legend(fontsize=12, ncol=1, loc="lower right")
    # ax.set_title("Matter temperature evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "TM_evolution.pdf", dpi=300)
    plt.show()


# ============================================================================
# Plot: X_e vs redshift
# ============================================================================


def _plot_ionization(results: list):
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = cm.plasma(np.linspace(0, 1, len(results)))

    for res, col in zip(results, colors, strict=False):
        z = res["z"]
        ax.loglog(
            z, res["Xe"], color=col, lw=1.0, alpha=0.8, label=res["label"]
        )

    # RecFast reference
    # RecFast reference (skipped when standard.dat is unavailable)
    if recfast is not None:
        ax.plot(
            recfast["z"],
            recfast["Xe"],
            color="red",
            ls="--",
            lw=1.5,
            label="RecFast",
        )

    ax.set_xlabel("Redshift $z$", fontsize=13)
    ax.set_ylabel("Ionization fraction $X_e$", fontsize=13)
    ax.set_xlim(Z_END, Z_START)
    ax.set_ylim(1e-4, 1.1)
    ax.legend(fontsize=10, ncol=2, loc="upper left")
    ax.set_title("Ionization fraction evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Xe_evolution.pdf", dpi=250)
    # plt.show()


# ============================================================================
# Plot: v_rel vs redshift
# ============================================================================


def _plot_velocity(results: list):
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = cm.plasma(np.linspace(0, 1, len(results)))

    for res, col in zip(results, colors, strict=False):
        z = res["z"]
        ax.loglog(
            z, res["vrel"] / 10, color=col, lw=1.5, alpha=1, label=res["label"]
        )

    ax.set_xlabel("Redshift $z$", fontsize=13)
    ax.set_ylabel(r"Relative velocity $v_{\rm rel}/10$ [Km/s]", fontsize=13)
    ax.set_xlim(Z_END, Z_START)
    ax.legend(fontsize=10, ncol=2, loc="upper left")
    ax.set_title("MACHO-baryon relative velocity evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vrel_evolution.pdf", dpi=250)
    # plt.show()


# ============================================================================
# Plot: per-bin velocity histories v_i(z)  (extended mass functions only)
# ============================================================================


def _plot_vbins(results: list):
    """
    The diagnostic the multi-bin scheme exists to produce: the family of
    velocity histories, one per mass node, coloured by log10(M_i).  Heavy
    bins peel off early (a ~ M) and go subsonic; light bins coast.  If every
    curve lies on top of every other, the mass function is too narrow for the
    multi-bin treatment to matter and the monochromatic run would do.
    """
    ext = [r for r in results if r["vbins"].shape[0] > 1]
    if not ext:
        return

    fig, axes = plt.subplots(
        1, len(ext), figsize=(6 * len(ext), 5), squeeze=False
    )

    for ax, res in zip(axes[0], ext, strict=False):
        z = res["z"]
        vb = res["vbins"]  # (n_bins, n_z) [km/s]
        logM = np.log10(res["Mbins"])
        norm = plt.Normalize(logM.min(), logM.max())
        cmap = cm.viridis

        for i in range(vb.shape[0]):
            ax.loglog(z, vb[i], color=cmap(norm(logM[i])), lw=0.9, alpha=0.85)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label(r"$\log_{10}(M_i / M_\odot)$", fontsize=11)

        ax.set_xlabel("Redshift $z$", fontsize=13)
        ax.set_ylabel(r"$v_i$ [km/s]", fontsize=13)
        ax.set_xlim(Z_END, Z_START)
        ax.set_title(res["label"], fontsize=11)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vbins_evolution.pdf", dpi=250)
    # plt.show()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()
