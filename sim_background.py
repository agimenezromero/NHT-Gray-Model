import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import matplotlib.colors
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patches as mpatches
from scipy import stats

import time
import os
from shutil import copyfile

from GrayModelClasses import *


#Run script

Lx = 100e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 10e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 20
Tf = 10
Ti = 10

t_MAX = 5000e-12
dt = 1e-12

W = 1

current_dir = os.getcwd()
folder = current_dir + '/Diffussive_%.i_ps_W_%.i' % (t_MAX * 1e12, W)

if not os.path.exists(folder): os.mkdir(folder)

os.chdir(folder)

f = open('parameters_used.txt', 'w')

f.write('Lx: ' + str(Lx) + '\n')
f.write('Ly: ' + str(Ly) + '\n')
f.write('Lz: ' + str(Lz) + '\n\n')

f.write('Lx_subcell: ' + str(Lx_subcell) + '\n')
f.write('Ly_subcell: ' + str(Ly_subcell) + '\n')
f.write('Lz_subcell: ' + str(Lz_subcell) + '\n\n')

f.write('T0: ' + str(T0) + '\n')
f.write('Tf: ' + str(Tf) + '\n')
f.write('Ti: ' + str(Ti) + '\n\n')

f.write('t_MAX: ' + str(t_MAX) + '\n')
f.write('dt: ' + str(dt) + '\n\n')

f.write('W: ' + str(W))

f.close()

gray_model = PhononGas(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W)

os.chdir(folder)
gray_model.simulation()