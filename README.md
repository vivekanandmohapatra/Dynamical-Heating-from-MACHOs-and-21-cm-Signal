# Dark-ages 21-cm signal with MACHO dynamical-friction heating

Computes the global 21-cm brightness temperature `T_21(z)` for an IGM
heated by dynamical friction on MACHOs (compact dark-matter objects),
including standard recombination, Compton coupling, and X-ray heating
from the first star-forming haloes.

All internal quantities use natural units with GeV as the base unit;
temperatures are converted to Kelvin only for output and plotting.

## Modules

| Module | Purpose |
| --- | --- |
| `units.py` | GeV-based natural-unit conversion factors. |
| `cosmology_terms.py` | Planck 2018 parameters and background quantities `H(z)`, `T_CMB(z)`, `n_b(z)`. |
| `thermal_evolution.py` | `StarFormation` (Press–Schechter collapse, X-ray heating), `MACHOHeating` (Chandrasekhar drag, multi-bin mass functions), and the `evolution` ODE right-hand side. |
| `calculate_tm_xe.py` | Integrates `[X_e, T_M, v_1 … v_N]` from `Z_START` to `Z_END` over `PARAM_GRID`; saves and plots the thermal history. |
| `calculate_21cm.py` | Turns `(z, X_e, T_M)` into spin temperature and `T_21` in mK. |

The dependency order is
`units → cosmology_terms → thermal_evolution → calculate_tm_xe → calculate_21cm`.

## Installation

```bash
git clone https://github.com/<user>/dark-ages-21cm.git
cd dark-ages-21cm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

Edit `PARAM_GRID` in `calculate_tm_xe.py`, then run either stage:

```bash
python calculate_tm_xe.py    # thermal and ionization history only
python calculate_21cm.py     # full sweep through to T_21
```

Both write `.npy` arrays and PDF figures into `results/`.

### Optional reference curve

`calculate_tm_xe.py` will overplot a RecFast++ reference history if a
file named `standard.dat` is present next to the module — three
comma-separated columns of `z`, `X_e`, `T_M [K]` with no header. If the
file is absent the code emits a `RuntimeWarning` and omits the curve;
it is not required to run.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
black . && ruff check .
```

Formatting is enforced at 79 columns (`black` + `ruff`, configured in
`pyproject.toml`) and checked in CI.

### Naming deviation from PEP 8

Physical constants keep their conventional symbols — `GeV`, `Kelv`,
`m_e`, `T_M`, `X_e`, `M_c` — rather than PEP 8's `UPPER_CASE` module
constants and `lower_case` locals. This is a deliberate, project-wide
choice so that the code reads like the equations in the paper; the
corresponding `pep8-naming` rules are disabled in `pyproject.toml`
rather than silenced case by case. Everything else — layout, imports,
line length, whitespace, docstrings — follows PEP 8 and PEP 257.

## Known issues

These are open questions in the physics, not style problems. They are
listed here because the numbers this code produces should not be taken
at face value until they are resolved.

1. **`J_0` units in `x_alpha` (`calculate_21cm.py`).** The critical
   Lyman-α flux is coded as `5.54e-8 * (1 + z) * Meter**-2`. The
   constant is quoted in the literature in
   cm⁻² s⁻¹ Hz⁻¹ sr⁻¹, which would make the correct factor
   `Centimeter**-2`. As written `J_0` is 10⁴ too small and `x_alpha`
   10⁴ too large. The module docstring is also self-inconsistent about
   whether `J_0` is dimensionless or `[GeV^2]`.
2. **Undocumented factors in `J_alpha`.** The prefactor contains
   `/ q_e` (≈ 0.303, an electromagnetic coupling with no obvious role
   in a photon-number flux) and `/ 1.22` (presumably the mean mass per
   hydrogen atom), neither of which appears in the formula in the
   docstring. Combined with `f_alpha_value = 10.0` — while the
   docstring claims a default of 1.0 — there are three stacked
   multiplicative factors whose provenance is unclear.
3. **Lyman-α horizon.** `z_max = (32 / 27) * z` should be
   `1 + z_max = (32 / 27) * (1 + z)`. The integrand uses `(1 + z')`
   ratios, so limit and integrand are inconsistent. Only `n = 2` is
   included; the standard treatment sums the Lyman series to
   `n_max ≈ 23` with recycling fractions.
4. **SFRD normalisation.** `StarFormation.sfrd` is documented as
   `[GeV^5]` in one docstring and `[GeV^4]` in another, and uses the
   comoving `rho_b0`. Check this against the `(1 + z)**2` factor in
   `J_alpha` for consistent comoving/proper bookkeeping.
5. **`k_eH` is uncapped above 10⁴ K.** The Furlanetto & Furlanetto fit
   is only valid to `T = 10⁴ K`; with X-ray heating switched on, gas
   temperatures can exceed that.
6. **`T_alpha = T_M`.** No Chen & Miralda-Escudé `S_alpha` correction
   is applied.

The collisional coupling `x_c` was checked independently and behaves
correctly (`x_c ≈ 2.9` at `z = 100`, `0.27` at `z = 50`, crossing unity
near `z ≈ 70`).

## License

MIT — see `LICENSE`.
