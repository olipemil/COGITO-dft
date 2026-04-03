# COGITO - Crystal Orbital Guided Iteration To atomic-Orbitals

[![Website](https://img.shields.io/badge/website-live-green?style=flat-square)](https://olipemil.github.io/cogito-website)
[![COGITO](https://img.shields.io/badge/main%20repo-COGITO-blue?style=flat-square)](https://github.com/olipemil/COGITO)

This repository hosts the code for COGITO, a tool for obtaining quantum chemistry from plane wave DFT calculations. The main codes file are COGITO.py, COGITOpost.py, and COGITOico.py.

## Quick Start

**Visit the live website:** [**cogito-website.github.io**](https://olipemil.github.io/cogito-website)

| Section | Description |
|---------|-------------|
| [Installation](https://olipemil.github.io/cogito-website/examples/installation_setup.html) | Get started with COGITO setup |
| [Tutorial](https://olipemil.github.io/cogito-website/tutorial/) | Step-by-step analysis workflow |
| [File Guide](https://olipemil.github.io/cogito-website/examples/) | Interactive guide of inputs and outputs |
| [API Docs](https://olipemil.github.io/cogito-website/api/) | Complete function reference |

**Basic Workflow:**

1. **Run VASP** - Static calculation with saved wavefunctions
2. **Generate COGITO model** - Creates tight binding parameters
3. **Verify quality** - Check convergence, spill, and orbital mixing (analyze band interpolation after runTBmodel.py)
4. **Analyze chemistry** - COHP, bonding, charge analysis

```bash
vasp_std
# Check 'python COGITO.py --help' to see variable options
python COGITO.py
# Checks that error metrics are within range and plots radius changes
python COGITOanalyze.py
# Edit runTBmodel.py to use functionality in COGITOpost.py
python runTBmodel.py
```

