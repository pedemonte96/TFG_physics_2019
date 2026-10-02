# Reproducing and exploring the results

## Featured figure

Follow the [README setup](../README.md#quick-start), then run:

```bash
python scripts/plot_reconstruction.py
```

The script reads three two-column tables in `harmonic_temp_annealing/square_bonds_python/Plots/`:

| File                    | Experiment                         | Relative error |
| ----------------------- | ---------------------------------- | -------------- |
| `M25000_L4_T1_S1_j.txt` | 4 x 4 lattice, sample 1, T = 0.5   | 0.053624       |
| `M25000_L4_T2_S1_j.txt` | 4 x 4 lattice, sample 1, T ≈ 1.083 | 0.024088       |
| `M25000_L4_T3_S1_j.txt` | 4 x 4 lattice, sample 1, T = 2.0   | 0.024136       |

Each table contains the original coupling in column 1 and its inferred value in column 2. The reported metric is the Euclidean norm of the coupling difference divided by the norm of the original couplings, matching the relative error used in the associated plotting notebooks.

The command recreates the scatter plots from saved estimates; it does not rerun inference. Its output is `output/coupling-reconstruction.png`. The README displays a committed copy in `docs/figures/`.

## Original notebooks

Install the additional dependencies into the same virtual environment:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyterlab
```

Open a notebook through JupyterLab. The notebooks use paths relative to their own directories, so keep each notebook alongside its original data folders. For example, open `harmonic_temp_annealing/square_bonds_python/Plots/square_bonds_L4_T2_S1.ipynb` to inspect the middle panel's experiment.

The original notebook metadata records **Python 3.7.0**. The dependency files provide a modern environment for exploration; they are not a reconstruction of the original environment. Only the featured plotting script has been validated with the documented setup.

Several plotting cells enable `rc('text', usetex=True)` and require a working LaTeX installation. For exploration without LaTeX, change that setting to `False` in your working copy.

Review an experiment's parameter and output cells before running it. These notebooks are research workbooks: some cells refer to earlier interactive variables or files that are absent from the archive, and some write over saved outputs. Filename labels can differ from the parameters in the cells. Long runs also use random initialization and sample selection without a consistently fixed seed, so a new run can produce different estimates.

## Data conventions

| Name                                               | Meaning                                                                                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `L4`, `L8`, `L32`                                  | Lattice side length; the number of spins is L².                                                                           |
| `sample1`, `sample2`, `sample3` / `S1`, `S2`, `S3` | Different bond samples.                                                                                                   |
| `T1`, `T2`, `T3`                                   | Physical sampling temperatures: 0.5, approximately 1.083, and 2.0. These differ from the optimizer's cooling temperature. |
| `M25000`, `m12500`, etc.                           | Number of configurations used for inference.                                                                              |
| `bonds.dat`                                        | Original nearest-neighbor couplings, indexed by spin and bond direction.                                                  |
| `configurations_T*.dat`                            | Binary spin configurations; the notebook readers map 0 to −1 and 1 to +1 and reshape them into rows of L² spins.          |
| `energies_T*.dat`                                  | Associated simulation energy records.                                                                                     |
| `*_j.txt`                                          | Saved original and inferred coupling pairs.                                                                               |
| `*_avg.txt`, `*_std.txt`                           | Saved averages and standard deviations across optimization runs.                                                          |

Some lattice notebooks subsample configurations with `spins[::4]`. Follow the selected notebook's preprocessing and parameter cells when interpreting its outputs. The square-lattice data were supplied by the advisor; their simulation generator is not part of this archive.
