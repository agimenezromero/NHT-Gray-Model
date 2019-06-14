# Nanoscale Heat Transport - Gray Model approach
Python little library to simulate nanoscale hat transport from the Gray Model approach. It has been used to study nanoscale heat transport in the following final degree project: https://github.com/agimenezromero/NHT-Gray-Model/tree/master/Final%20Degree%20Project

# Overview
The aim of this program is to simulate thermal transport at nanoscale. The method used is known as the Gray Model, which considers that phonon properties (shuch as energy, average velocity, average frequency...) are based only on local sub-cell temperature. 

Two simulation types have been implemented:

1. `GrayModel` : Considers specular reflection with the domain walls.
2. `GrayModel_diffusive_walls` : Considers diffusive reflection with the domain walls.

More information about the model can be found in the [final degree project](https://github.com/agimenezromero/NHT-Gray-Model/tree/master/Final%20Degree%20Project). 

Table of contents
=================

<!--ts-->
   * [Overview](#overview)
   * [Table of contents](#table-of-contents)
   * [Requeriments](#requeriments)
   * [Usage](#usage)
   * [Examples](#examples)
       - [Create input arrays](#create-input-arrays)
       - [New simulation](#new-simulation)
       - [Simulation from restart](#simulation-from-restart)
       - [Animation](#animation)
       - [Animation from restart](#animation-from-restart)
   * [Authors](#authors)
   * [License](#license)
<!--te-->

# Requeriments
- NumPy
- Matplotlib
- SciPy

# Usage
First of all the input arrays dependent on temperature need to be created. To do so the `ThermalProperties` class has been developed. For the study in the final degree project germanium and silicon have been simulated, so two simple functions have been implemented to create and storage the corresponding arrays easily: 

* `save_arrays_germanium(init_T, final_T, n)` 
* `save_arrays_silicon(init_T, final_T, n)`.

  - `ìnit_T` (float) - Initial temperature for the computed properties.
  - `final_T` (float) - Final temperature for the computed properties.
  - `n` (int) - Number of points between initial and final temperatures.

To initialise both of the available simulation classes (`GrayMode`, `GrayModel_diffusive_walls`) the following parameters must be passed in:

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

## Create input arrays
To create and storage the input arrays needed by the simulation software the `ThermalProperties` class has been developed. Then, two functions have been built to use it to create the input arrays for germanium and silicon. 

```python
from GrayModelClasses import *
#also from GrayModelClasses import save_arrays_germanium, save_arrays_silicon

init_T = 1
final_T = 500
n = 10000

save_arrays_germanium(init_T, final_T, n)
save_arrays_silicon(init_T, final_T, n)
```

With this simple code the input arrays for these materials will be created  and stored in the automatically created `Input_arrays` folder.

## New simulation

To perform a new simulation, all the system parameters must be initialised.

```python
from GrayModelClasses import *
#also from GrayModelClasses import GrayModel

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

gray_model = GrayModel(init_restart=True, folder_restart='restart_100')
gray_model.simulation_from_restart()
```
However optional arguments can be passed to the `simulation_from_restart` function

```python
from GrayModelClasses import *

gray_model = GrayModel(init_restart=True, folder_restart='restart_1000')
gray_model.simulation_from_restart(every_restart=1000, folder='EXAMPLE_OUTPUTS')
```
## Animation
A real time animation of the sub-cell temperature evolution is also available, which makes all the calculation needed *on the fly*. Runing the animation for a new system is as simple as running a simulation.

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

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.animation()
```

## Animation from restart
And one can also start the animation from an existing restart

```python
from GrayModelClasses import *

gray_model = GrayModel(init_restart=True, folder_restart='restart_1000')
gray_model.animation_from_restart()
```

## Diffusive boundary walls
To considere diffusive boundary walls just call the other implemented class named `GrayModel_diffusive_walls` which is used in the same way as the `GrayModel` class. Anyway a single example is presented

```python
from GrayModelClasses import *
#also from GrayModelClasses import GrayModel_diffusive_walls

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

#Optional: Default are 100 and 'OUTPUTS'
every_restart = 1000
folder_outputs = 'EXAMPLE_OUTPUTS'

gray_model = GrayModel_diffusive_walls(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)
gray_model.simulation(every_restart, folder_outputs)
```

# Authors
* **A. Giménez-Romero**

# License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/agimenezromero/NHT-Gray-Model/blob/master/LICENSE) file for details

