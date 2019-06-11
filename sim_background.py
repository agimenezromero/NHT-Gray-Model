import os
from GrayModelClasses import *
import time

current_time = lambda: round(time.time(), 2)

##########################################################################################################
												#PARAMETERS										 		 #
##########################################################################################################

Lx = 200e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 10e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

T0 = 300
Tf = 280
Ti = 290

t_MAX = 1000e-12
dt = 1e-12

W = 50
every_flux = 5
every_restart = 1000

#Optional: Default are 100 and 'OUTPUTS'
every_restart = 1000
folder_outputs = 'OUTPUTS'

#Just needed if starting from restart: default are False and None
init_restart = True
folder_restart = 'restart_100'

##########################################################################################################
##########################################################################################################

#RUN SIMULATION

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)

t0 = current_time()

gray_model.simulation(every_restart)

tf = current_time()

elapsed_time = round(tf - t0, 2)

f = open('parameters_used.txt', 'a')

f.write('\nElapsed time: ' + str(elapsed_time) + ' s')

f.close()

