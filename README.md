# Global 21-cm signal in the presence of MACHO dynamical-friction heating

Computes the global 21-cm brightness temperature `T_21(z)` for an IGM heated by dynamical friction on MACHOs (MAssive Compact Halo Objects), including standard recombination, Compton coupling, and  Ly-alpha & X-ray heating from the first star-forming haloes. Furthermore, we have incorporated physically motivated mass spectra such as the Log-normal distribution and critical collapse, in addition to the monochromatic distribution.


# Primary idea: 

- Unlike the particle dark matter paradigm, we consider that dark matter may exist as compact halo objects. The most studied  MACHO candidate in the literature to have left imprints on pre- and post-recombination physics is the Primordial Black Hole (PBH). Furthermore, non-PBH-like MACHOs have been studied either through enhanced dark matter annihilation due to their compact structure or by invoking axion-like particles, which require Beyond the Standard Model (BSM) physics. In this work, we consider dark matter to be composed of compact halo objects formed from primordial fluctuations, and they interact with the IGM purely gravitationally. In this minimalistic framework, the MACHOs tend to induce enhanced-density wakes (overdense regions in the IGM) surrounding them, and as they stream through the IGM, the frictional force between the MACHOs and the IGM slows down the objects, dissipating their kinetic energy and heating the IGM.
- This heating of the IGM leaves an imprint on the global 21-cm signal. 
- Note that the kinetic energy of MACHOs originates from the baryon-dark matter relative velocity ($v_{bc}$).
- Furthermore, frictional force depends on the ratio $v_{bc}/c_s$, where $c_s$ is the sound speed. Interestingly, this ratio remains subsonic before recombination; hence, the MACHOs enter the supersonic regime right after recombination, thus leaving little to no imprint on the CMB. This makes the dark ages 21-cm signal the most important probe for such dark matter candidates, providing a cosmological window free from astrophysical uncertainties.
  
# References:
- ### Dark ages bounds on nonaccreting massive compact halo objects (https://arxiv.org/abs/2604.17083) --- If you find this paper relevant and this repository helpful, please add this paper to your references.
- Dynamical Heating from Dark Compact Objects and Axion Minihalos: Implications for the 21-cm Signal (https://arxiv.org/abs/2512.00169)
- Nonlinear Dynamical Friction in a Gaseous Medium (https://arxiv.org/abs/0908.1391)
- Unified gas heating constraints on extended dark matter compact objects (https://arxiv.org/abs/2508.18344)

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

All internal quantities use natural units with GeV as the base unit; temperatures are converted to Kelvin only for output and plotting.


## Installation
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



### Naming deviation

Physical constants keep their conventional symbols — `GeV`, `Kelv`,
`m_e`, `T_M`, `X_e`, `M_c`


## License

MIT — see `LICENSE`.
