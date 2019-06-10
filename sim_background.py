import os
from GrayModelClasses import *
import time

current_time = lambda: round(time.time(), 2)

##########################################################################################################
												#PARAMETERS										 		 #
##########################################################################################################

Lx = 1e-10
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 0.5e-11
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 300
Tf = 280
Ti = 280

t_MAX = 2000e-12
dt = 0.01e-12

W = 5e-1
every_flux = 5
every_restart = 1000

init_restart = False
folder_restart = 'restart_3000'

folder_outputs = 'OUTPUTS'

##########################################################################################################
##########################################################################################################

#RUN SIMULATION

current_dir = os.getcwd()
folder = current_dir + '/' + folder_outputs

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

f.write('W: ' + str(W) + '\n')
f.write('Every_flux: ' + str(every_flux))

f.close()

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux, init_restart, folder_restart)

t0 = current_time()

gray_model.simulation(every_restart, folder_outputs)

tf = current_time()

elapsed_time = round(tf - t0, 2)

f = open('parameters_used.txt', 'a')

f.write('\nElapsed time: ' + str(elapsed_time) + ' s')

f.close()

