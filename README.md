# Nanoscale Heat Transport - Gray Model approach
Python little library to simulate nanoscale hat transport from the Gray Model approach.

# Overview
The aim of this program is to simulate thermal transport at nanoscale. The method used is known as the Gray Model, which considers that phonon properties (shuch as energy, average velocity, average frequency...) are based only on local sub-cell temperature. 

Two simulation types have been implemented:

1. `GrayModel` : Considers specular reflection with the domain walls.
2. `GrayModel_diffusive_walls` : Considers diffusive reflection with the domain walls.

Table of contents
=================

<!--ts-->
   * [Overview](#overview)
   * [Table of contents](#table-of-contents)
   * [Requeriments](#requeriments)
   * [Usage](#usage)
   * [Examples](#examples)
   * [Authors](#authors)
   * [License](#license)
   * [Acknowledgments](#acknowledgments)
<!--te-->

# Requeriments
- NumPy
- Matplotlib

# Usage
To initialise each of the available classes the following parameters must be passed in:

- `Lx` (float) - Domain lenght (x-direction).
- `Ly` (float) - Domain width (y-direction).
- `Lz` (float) - Domain height (z-direction).

- `Lx_subcell` (float) - Subcell length (x-direction).
- `Ly_subcell` (float) - Subcell width (y-direction).
- `Lz_subcell` (float) - Subcell height (z-direction).

- `T0` (float) - Hot boundary (first cell or array of cells).
- `Tf` (float) - Cold boundary (last cell or array of cells).
- `Ti` (float) - Other cells initial temperature.

- `t_MAX` (float) - Maximum simulation time.
- `dt` (float) - Integration time step.

- `W` (float) - Weighting factor.
- `every_flux` (int) - Flux calculation period in frame units.

- `init_restart` : (bool, optional) - Set to true to initialise a simulation from a restart. 
- `folder_restart` : (string, optional) - Specify the restart folder to start with.

Moreover, simulation function admit this optional parameters

- `every_restart` (int, optional) - Restart writting period in frame units.
- `folder_outputs` : (string, optional) - Folder name to save output files.

# Examples

## New simulation

```python
from GrayModelClasses import *

#Define the variables: Lx, Ly, Lz, etc.

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.simulation(every_restart, folder_outputs)
```

## Simulation from restart

```python
from GrayModelClasses import *

#Define the variables, although the values won't be used

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.simulation()
```

# Authors
* **A. Giménez-Romero**

# License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/agimenezromero/NHT-Gray-Model/blob/master/LICENSE) file for details

# Acknowledgments
