# Overview

**COGITO** (Chemically Interpretable Tight-binding from Orbitals) is a Python toolkit for
chemically interpretable tight-binding and bonding analysis from DFT calculations.

::::{grid} 2

:::{grid-item-card} COGITO
:link: api/COGITO
:link-type: doc

Core class for reading VASP output, iterating adaptable atomic orbitals, and creating their local Hamiltonians.
:::

:::{grid-item-card} COGITOpost
:link: api/COGITOpost
:link-type: doc

Process results of COGITO tight binding model. Make key json files for custom analysis. Get orbital/COHP/COOP-projected bandstructure and DOS.
:::

:::{grid-item-card} COGITOanalyze
:link: api/COGITOanalyze
:link-type: doc

Analyze convergence and quality of the COGITO basis. Check the quality of TB interpolation.
:::

:::{grid-item-card} COGITOico
:link: api/COGITOico
:link-type: doc

Partial version of COGITOpost which uses pre-calculated integrated COHP/COOP to quickly calculate 3D bond plots.
:::

::::
