# LAMMPS Rerun Force Comparison — Agent Instructions

## Context

You are helping Alex, a computational chemistry researcher, validate force field parameters for CdSe quantum dot systems. He has a PyTorch-based fitter (`LJ_coul_fitter.py`) that optimizes Lennard-Jones + Coulomb parameters against DFT forces from CP2K. The fitter reports F_RMSE of ~0.18 eV/Å on DFT geometries, but when those parameters are used in LAMMPS MD simulations, the QD structure falls apart. Meanwhile, literature parameters from Infante (which report ~0.46 eV/Å in the fitter) produce stable MD.

**The critical question:** Does the fitter's 0.18 eV/Å hold up when LAMMPS evaluates forces at the exact same DFT geometries? If not, the fitter is solving a different optimization problem than what LAMMPS actually computes (due to differences in Coulomb handling, PBC, etc.), and that gap is the root cause of failure.

## What We're Doing

Using LAMMPS `rerun` to evaluate forces on DFT snapshots without doing MD. This gives us LAMMPS forces at the exact same geometries the fitter trained on, allowing a direct apples-to-apples comparison.

## File Layout

The working directory should contain:
```
./
├── Cd68Se55_300K.xyz       # DFT trajectory (CP2K xyz, positions in Angstrom)
├── Cd68Se55_300K.forces    # DFT forces (CP2K format, Ha/Bohr)
├── Cd68Se55.data           # LAMMPS data file (atom types, box, topology)
├── setup_rerun.py          # Converts DFT xyz → LAMMPS dump, generates input template
├── compare_forces.py       # Compares DFT forces vs LAMMPS rerun forces
├── dft_frames.dump         # (generated) DFT positions in LAMMPS dump format
├── rerun.lammps            # (generated) LAMMPS input template
└── xyz/                    # (optional) rerun output dumps go here, or in cwd
```

## Workflow

### Step 1: Setup
```bash
python setup_rerun.py
```
- Reads `*.xyz` and `*.data` from cwd
- Produces `dft_frames.dump` (DFT trajectory in LAMMPS dump format)
- Produces `rerun.lammps` (LAMMPS input template — needs parameters pasted in)
- Verifies atom ordering between xyz and data file

### Step 2: Edit rerun.lammps
Paste the force field parameters into the marked section. Three runs are needed:

**Run A — Infante locked (baseline):**
Use Infante's published LJ parameters and charges exactly as-is. This is the known-good parameter set that produces stable MD.

**Run B — Trained from Infante initial:**
Use the parameters output by `LJ_coul_fitter.py` when initialized from Infante's values and trained with F_RMSE loss.

**Run C — Trained from smart initial:**
Use the parameters from the fitter when initialized with RDF-based sigma guesses.

For each run, rename the output so they don't overwrite:
```bash
lammps -in rerun.lammps
mv rerun_forces.dump rerun_infante_locked.dump
# (edit parameters, rerun for next set)
```

### Step 3: Compare
```bash
python compare_forces.py
```
- Looks for `*.forces` in cwd (DFT reference)
- Looks for `rerun_*.dump` in cwd and `*.dump` in `xyz/`
- For each dump: computes per-element F_RMSE, produces hexbin parity plot
- Frame-by-frame comparison is valid here because geometries are identical

## What to Look For

### Scenario A: LAMMPS rerun reproduces fitter's RMSE
- Infante locked: ~0.46 eV/Å (matches fitter)
- Trained: ~0.18 eV/Å (matches fitter)

This would mean the fitter's force calculation matches LAMMPS, and the training genuinely improved force accuracy. The problem is then purely that better per-frame forces don't lead to stable dynamics — the optimizer may be overfitting to many-body noise.

### Scenario B: LAMMPS rerun does NOT reproduce fitter's RMSE
- Infante locked: ~0.46 eV/Å
- Trained: ~0.40-0.46 eV/Å (much worse than fitter's 0.18)

This would mean the fitter and LAMMPS compute forces differently, and the optimizer exploited that gap. The 0.18 eV/Å was achieved by fitting to artifacts of the simplified force calculation (bare Coulomb direct sum, no PBC, no Ewald) rather than to real physics.

**Known differences between fitter and LAMMPS force calculation:**
- Fitter: bare Coulomb k·q·q/r² direct sum, no periodic images
- LAMMPS with `coul/long`: Ewald/PPPM, periodic boundary conditions
- Fitter: 60 Å cutoff on all interactions (effectively no cutoff for this box)
- LAMMPS: pair cutoff + long-range Coulomb correction

### Scenario C: Mixed
Some intermediate result, e.g. LAMMPS rerun gives 0.30 eV/Å for trained params. This means training helped somewhat, but part of the improvement was from exploiting fitter-LAMMPS differences.

## Units

- DFT forces file: Ha/Bohr (CP2K native) — compare_forces.py converts to eV/Å (× 51.42)
- LAMMPS dump forces: eV/Å (metal units) — used as-is
- Positions: Angstrom everywhere
- The conversion factor Ha/Bohr → eV/Å = 27.211 / 0.5292 ≈ 51.42

## Key Insight

The fitter uses a simplified force calculator (no Ewald, no PBC). If the rerun reveals a large gap, the solution is to either:
1. Make the fitter's force calculation match LAMMPS exactly (hard)
2. Use LAMMPS itself as the force calculator during training (slow but correct — this is essentially what Infante does with iterative MD + RDF matching)
3. Add a regularization or constraint that prevents the optimizer from exploiting the gap (e.g., constraining parameters to stay near Infante's, or matching force distributions)

## Notes

- `setup_rerun.py` reads atom types from the `.data` file and maps them to elements via the Masses section. If comments aren't present, it falls back to mass-matching.
- The rerun uses `box no` so LAMMPS uses the box from the data file, not from the dump frames.
- Atom ordering must match between the xyz trajectory and the data file. `setup_rerun.py` verifies this.
- The `--max-frames` flag on both scripts limits frames for faster testing.
