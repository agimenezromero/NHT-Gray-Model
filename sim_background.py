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

t_MAX = 2e-12
dt = 0.01e-12

W = 5e-1
every_flux = 5
every_restart = 100

folder_outputs = 'OUTPUTS'

#Just needed if starting from restart: default are False and None
init_restart = True
folder_restart = 'restart_100'

##########################################################################################################
##########################################################################################################

#RUN SIMULATION

gray_model = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux)

t0 = current_time()

gray_model.simulation(every_restart, folder_outputs)

tf = current_time()

elapsed_time = round(tf - t0, 2)

f = open('parameters_used.txt', 'a')

f.write('\nElapsed time: ' + str(elapsed_time) + ' s')

f.close()

