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
       - [New simulation](#new-simulation)
       - [Simulation from restart](#simulation-from-restart)
       - [Animation](#animation)
       - [Animation from restart](#animation-from-restart)
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

To perform a new simulation, all the system parameters must be initialised.

```python
from GrayModelClasses import *

Lx = 10e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 0.5e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 11.88
Tf = 3
Ti = 5

t_MAX = 10e-9
dt = 0.1e-12

W = 0.05
every_flux = 5
every_restart = 1000

#Optional: Default are 100 and 'OUTPUTS'
every_restart = 1000
folder_outputs = 'EXAMPLE_OUTPUTS'

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.simulation(every_restart, folder_outputs)
```

## Simulation from restart

To run a simulation from a restart only the `init_restart` and `restart_folder` are needed to initialise the class. If the class is initialized with some of the other arguments of the exemple above they will be simply ignored.

```python
from GrayModelClasses import *

#Define the variables, although the values won't be used

gray_model = GrayModel(init_restart=True, folder_restart='restart_100')
gray_model.simulation_from_restart()
```
However optional arguments can be passed to the `simulation_from_restart` function

```python
from GrayModelClasses import *

#Define the variables, although the values won't be used

gray_model = GrayModel(init_restart=True, folder_restart='restart_100')
gray_model.simulation_from_restart(every_restart=1000, folder='EXEMPLE_OUTPUTS')
```
## Animation

## Animation from restart

# Authors
* **A. Giménez-Romero**

# License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/agimenezromero/NHT-Gray-Model/blob/master/LICENSE) file for details

# Acknowledgments
