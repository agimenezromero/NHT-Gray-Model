import os
from GrayModelClasses import *
import time

current_time = lambda: round(time.time(), 2)

#Run script

Lx = 20e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 1e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 11.88
Tf = 3
Ti = 5

t_MAX = 10000e-12
dt = 0.1e-12

W = 0.01

current_dir = os.getcwd()
folder = current_dir + '/Flux_Ballistic_Lx_20_nm'

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

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W)

t0 = current_time()

os.chdir(folder)
gray_model.simulation()

tf = current_time()

elapsed_time = round(tf - t0, 2)

f = open('parameters_used.txt', 'a')

f.write('\nElapsed time: ' + str(elapsed_time) + ' s')

f.close()

