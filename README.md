# SW Fitter

SW Fitter fits a generalized Stillinger-Weber potential to atomic forces stored in extended XYZ trajectories. It supports multiple datasets with different compositions, estimates bond scales and angular initial values from the trajectories, and exports the fitted potential in LAMMPS `.sw` format.

The fitted pair parameters are `eps`, `A`, `B`, `p`, `q`, and `gamma`. Each centered triplet has its own `lambda` and equilibrium angle. Pair equilibrium distances and cutoffs are estimated from radial distributions and remain fixed during training.

## Requirements

- Python 3.10 or newer
- Reference trajectories in extended XYZ format
- Positions in angstrom and forces readable by ASE in eV/angstrom
- LAMMPS only if the export consistency check will be run

All frames within one trajectory must contain the same atoms in the same order. The current graph construction is intended for nonperiodic clusters.

## Installation

Clone or download the repository, enter its root directory, and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

PyTorch is installed as a dependency. If a platform requires a particular CUDA build, install the appropriate PyTorch build before installing SW Fitter.

The editable installation provides two commands:

```text
sw-fitter
sw-fitter-check
```

The program can also be run without the command-line entry point:

```bash
python -m sw_fitter config.yaml
```

## Configuration

Copy the example configuration and edit the trajectory paths:

```bash
cp config.example.yaml config.yaml
```

Relative trajectory paths are resolved relative to the configuration file. A minimal configuration has three sections:

- `datasets`: trajectories and optional frame slicing
- `scope`: elements included in the fitted force field
- `training`: optimizer and validation settings

Each dataset accepts:

- `name`: label used in logs
- `xyz`: path to an extended XYZ trajectory
- `skip_frames`: number of initial frames to discard
- `first_n_frames`: number of frames retained after skipping

`scope.fit_elements` lists the framework elements whose force components enter the loss. `scope.ligand_bond_elements` adds selected ligand elements to framework-ligand pair and triplet construction without fitting pure ligand interactions.

`eps_init` contains optional pair-specific initial energy scales in eV. Missing pairs use `eps_init_default`. These are initial values, not fixed parameters. The remaining initial values are generated internally:

- Bond distances and cutoffs are estimated from radial distributions in the training frames.
- Equilibrium angle cosines are initialized from sampled triplet geometries.
- Angular strengths are initialized from the angular variance and training temperature.

The validation set is the final `validation_split` fraction of each trajectory. It is separated before bond scales and angular initial values are calculated.

## Running a fit

```bash
sw-fitter config.yaml
```

By default, output is written to a `results` directory beside the configuration file. A different parent directory can be selected with:

```bash
sw-fitter config.yaml --output-dir path/to/run_directory
```

The output directory contains:

- `training.log`: epoch history, final parameters, and timing
- `checkpoint.pt`: fitted parameters, configuration, scales, and loss history
- `<formula>_universal.sw`: LAMMPS parameter file
- `plots/training_curves.png`
- `plots/force_parity_training.png`
- `plots/force_parity_validation.png`
- `plots/sw_potentials.png`

Set `training.random_seed` to reproduce dataset and batch ordering. Training currently starts from the configured initial values; checkpoint resume is not implemented.

## LAMMPS consistency check

The consistency check compares the Python forces from the initialized model with forces produced from the exported LAMMPS file:

```bash
sw-fitter-check config.yaml --lammps /path/to/lmp
```

If `lmp` is already on `PATH`, the `--lammps` argument is unnecessary. Generated files are written to `consistency_results` by default. The number of frames used to estimate the initial parameters can be changed with `--frames`.

## Parameter interpretation

The two-body force depends on the product `A * eps`, while a triplet strength depends on `lambda * sqrt(eps_ca * eps_cb)`. Consequently, force fitting alone does not make `eps`, `A`, and `lambda` individually unique. The checkpoint and training log include the triplet strength in eV so that the directly relevant combined scale is available alongside `lambda`.

This does not prevent the model from fitting forces, but individual scale parameters should not be interpreted as independently measured physical quantities without an additional normalization convention or external constraint.

## Notes on the cutoff estimate

For each candidate pair, the code locates the largest radial-distribution peak and the minimum within the following 2 angstrom. The SW cutoff is that minimum multiplied by `TAIL_FACTOR`, currently 1.20. The cutoff is therefore a trajectory-derived modeling choice and should be checked when applying the fitter to a new chemical system.
