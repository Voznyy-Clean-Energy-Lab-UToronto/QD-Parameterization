"""Data-derived Stillinger-Weber force-field fitter for CdSe quantum dots.

Pipeline (see each module's docstring for detail):
  data.py             - load DFT trajectories, assign subtypes, build edges/triplets,
                        and measure the per-pair geometry scales (sigma, cutoff, r0).
  models.py           - the SW model: 2-body + 3-body forces; only epsilon is trainable.
  fitter.py           - force-match epsilon (run: `python -m sw_fitter_v9 config.yaml`).
  lammps_export.py    - write the fitted parameters to a LAMMPS .sw potential file.
  consistency_check.py- verify SWModel forces == LAMMPS forces on a real config.
  plotting.py         - training curves and force-parity plots.
  utils.py            - unit conversions (Hartree/Bohr <-> eV/Angstrom) and helpers.
"""
