# COGITO - Crystal Orbital Guided Iteration To atomic-Orbitals

[![Website](https://img.shields.io/badge/website-live-green?style=flat-square)](https://cogito-dft.readthedocs.io/)
[![COGITO](https://img.shields.io/badge/main%20repo-COGITO-blue?style=flat-square)](https://github.com/olipemil/COGITO-dft)

This repository hosts the code for COGITO, a tool for obtaining quantum chemistry from plane wave DFT calculations. The main codes files are COGITO.py, COGITOpost.py, COGITOanalyze.py, and COGITOico.py.

## Installation

```bash
pip install --upgrade pip
pip install cogito-dft
```

To install optional dependences (scikit-image, dash, dash-ag-grid) that are using in some COGITOpost functions

```bash
pip install "cogito-dft[plot]"
```

To avoid thread oversubscription and possible stalls **(especially on HPC)**, set:

```bash
export OMP_NUM_THREADS=1
```

## Quick Start

**Visit the live website:** [**cogito-dft.readthedocs.io**](https://cogito-dft.readthedocs.io/)

| Section | Description |
|---------|-------------|
| [Tutorial](https://cogito-dft.readthedocs.io/en/latest/tutorial.html) | Step-by-step analysis workflow |
| [File Guide](https://cogito-dft.readthedocs.io/en/latest/file_struc.html) | Interactive guide of inputs and outputs |
| [API Docs](https://cogito-dft.readthedocs.io/en/latest/api_ref.html) | Complete function reference |

**Basic Workflow:**

1. **Run VASP** - Static calculation with saved wavefunctions and high NBANDS
2. **Generate COGITO model** - Creates COGITO basis functions and tight binding parameters
3. **Verify quality** - Check convergence, charge spilling, orbital mixing, and band interpolation
4. **Analyze chemistry** - COHP, bonding, charge analysis

```bash
# run VASP 
vasp_std

# Check 'COGITO --help' to see variable options
COGITO --dir './'

# Checks that quality metrics are within range (check band interpolation after COGITOpost)
COGITOanalyze --dir './'

# Generate atom and bond partition of charge / band energies, make bond plots, and more
COGITOpost --dir './'
# Customize a runTBmodel.py file to get specific plots
python runTBmodel.py
```

