import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

from scipy import stats

import os

def diffussive_T(T, T0, Tf, xf):
	'''
		Computes the steady state temperature for the diffussive regime
		from the Fourier Law (Boundaries at x0=0 and xf=xf)
	'''

	k = (Tf - T0) / xf
	return k * T + T0

def balistic_T(T0, Tf):
	'''
		Computes the steady state temperature for the balistic regime
		from the Boltzmann Law
	'''
	return ((T0**4 + Tf**4)/2)**(0.25)

def get_parameters(filename):
	f = open(filename, 'r')

	i = 0

	for line in f:

		try:

			cols = line.split()

			if len(cols) > 0:
				value = float(cols[1])

			if i == 0:
				Lx = value

			elif i == 1:
				Ly = value

			elif i == 2:
				Lz = value

			elif i == 4:
				Lx_subcell = value

			elif i == 5:
				Ly_subcell = value

			elif i == 6:
				Lz_subcell = value

			elif i == 8:
				T0 = value

			elif i == 9:
				Tf = value

			elif i == 10:
				Ti = value

			elif i == 12:
				t_MAX = value

			elif i == 13:
				dt = value

			elif i == 15:
				W = value

			elif i == 16:
				every_flux = value

			i += 1

		except:
			pass

	return Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux

current_dir  = os.getcwd()

def Ballistic_regime_1D(folder):
	os.chdir(current_dir + '/' + folder)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')
	elapsed_time = np.load('Elapsed_time.npy')

	time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	equil = 7500

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.mean(T_cells[equil : , i]))

	total_avg_T = np.mean(T_cells[equil:])

	#plt.plot(x, average_T_cells, ls='-', color='k', marker='s', label='Average cell T %.2f' % total_avg_T)
	plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at %.2f ps' % float(time[1]))
	plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ps' % float(time[1000]))
	plt.plot(x, T_cells[5000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ps' % float(time[5000]))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1] * 1e-3))

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

def Diffusive_regime_1D(folder_filename):
	os.chdir(current_dir + '/' + folder_filename)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))

	time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	equil = 2000

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.mean(T_cells[equil : , i]))


	#plt.plot(x, average_T_cells, ls='-', color='purple', marker='None', label='Average cell T')
	plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at %.2f ps' % float(time[0]))
	plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ns' % float(time[1000]*1e-3))
	plt.plot(x, T_cells[2000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ns' % float(time[2000]*1e-3))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1]*1e-3))

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

def Ballistic_to_diffusive_1D_roughness(folder_filenames):
	plt.figure(figsize=(10, 6))

	colors = ['y','r', 'b', 'purple', 'g']
	markers = ['x','*', 'o', '^', 's']

	k = 0

	current_dir_now = current_dir + '/Ballistic_to_diffusive_boundary'

	for name in folder_filenames:

		os.chdir(current_dir_now + '/' + name)

		Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

		E = np.load('Energy.npy')
		N = np.load('Phonons.npy')
		T_cells = np.load('Subcell_Ts.npy')
		scattering_events = np.load('Scattering_events.npy')
		temperatures = np.load('Temperatures.npy')
		elapsed_time = np.load('Elapsed_time.npy')

		time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

		#T plot
		x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

		#Average for each subcell in equilibrium

		average_T_cells = []
		equil = 7500

		for i in range(int(round(Lx/Lx_subcell, 0))):
			average_T_cells.append(np.mean(T_cells[equil : , i]))

		total_avg_T = np.mean(T_cells[equil:])

		label = r'Diffusive $L_y=L_z=%.f$ nm' % (Ly * 1e9)

		if k == 0:
			label = r'Specular $L_y=L_z=%.f$ nm' % (Ly * 1e9)

		elif k == 4:
			label = r'Diffusive $L_y=L_z=%.1f$ nm' % (Ly * 1e9)

		plt.plot(x, average_T_cells, ls='-', color=colors[k], marker=markers[k], label=label)
		#plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color=colors[k], marker=markers[k], label=label)
	
		k += 1

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
			balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Steady state temperature')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

def Ballistic_to_diffusive_Lx(folder_filenames):

	colors = ['y','r', 'b', 'g']
	markers = ['x','*', 'o', 's']

	k = 0

	current_dir_now = current_dir + '/Ballistic_to_diffusive_Lx'

	for name in folder_filenames:

		os.chdir(current_dir_now + '/' + name)

		Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

		E = np.load('Energy.npy')
		N = np.load('Phonons.npy')
		T_cells = np.load('Subcell_Ts.npy')
		scattering_events = np.load('Scattering_events.npy')
		temperatures = np.load('Temperatures.npy')
		elapsed_time = np.load('Elapsed_time.npy')

		time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

		#T plot
		x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0))) * 1e9

		#Average for each subcell in equilibrium

		average_T_cells = []
		equil = 5000

		for i in range(int(round(Lx/Lx_subcell, 0))):
			average_T_cells.append(np.mean(T_cells[equil : , i]))

		total_avg_T = np.mean(T_cells[equil:])

		label = r'$t=%.f$ ns' % (t_MAX * 1e9)

		plt.figure(figsize=(10, 6))

		plt.plot(x, average_T_cells, ls='-', color='g', marker='None', label='Average cell T')
		plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='None', 
			label=label)

		plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
			balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
		plt.plot(x, diffussive_T(x*1e-9, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')
	
		plt.title('Steady state temperature')
		plt.xlabel('Domain length [nm]')
		plt.ylabel('Temperature [K]')

		plt.legend()
		plt.show()

		k += 1

	#plt.suptitle('Domain temperature evolution')
	#plt.show()

def all_plots(folder):
	os.chdir(current_dir + '/' + folder)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))
	print('Every_flux: ' + str(every_flux))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')
	elapsed_time = np.load('Elapsed_time.npy')
	flux = np.load('Flux.npy')

	time = np.linspace(0, t_MAX*1e12, int(round(t_MAX/dt, 0)))
	time_flux = np.linspace(0, t_MAX*1e12, int(round(t_MAX/(dt * every_flux))))

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	equil = 7500

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.mean(T_cells[equil : , i]))

	total_avg_T = np.mean(T_cells[equil:])

	#plt.plot(x, average_T_cells, ls='-', color='k', marker='s', label='Average cell T %.2f' % total_avg_T)
	plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at %.2f ps' % float(time[1]))
	#plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ps' % float(time[1000]))
	#plt.plot(x, T_cells[5000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ps' % float(time[5000]))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1] * 1e-3))

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

	plt.plot(time_flux, flux, ls='--', color='c', label='')
	plt.title('Flux')
	plt.show()

	plt.plot(time, elapsed_time, ls='', marker='x', color='g')

	slope, intercept, r_value, p_value, std_error = stats.linregress(time, elapsed_time)

	y_regre = np.array(time) * slope + intercept

	plt.plot(time, y_regre, color='r', label='y=%.2fx+%.2f' % (slope, intercept))

	plt.title('Elapsed time')
	plt.legend()
	plt.show()

def plots_restart(folder, t_max):
	os.chdir(current_dir + '/' + folder)

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

	print('Parameters used')
	print('---------------------------------------\n')

	print('Lx: ' + str(Lx))
	print('Ly: ' + str(Ly))
	print('Lz: ' + str(Lz) + '\n')

	print('Lx_subcell: ' + str(Lx_subcell))
	print('Ly_subcell: ' + str(Ly_subcell))
	print('Lz_subcell: ' + str(Lz_subcell) + '\n')

	print('T0: ' + str(T0))
	print('Tf: ' + str(Tf))
	print('Ti: ' + str(Ti) + '\n')

	print('t_MAX: ' + str(t_MAX))
	print('dt: ' + str(dt) + '\n')

	print('W: ' + str(W))
	print('Every_flux: ' + str(every_flux))

	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')
	elapsed_time = np.load('Elapsed_time.npy')
	flux = np.load('Flux.npy')

	time = np.linspace(0, len(E), len(E))
	time_flux = np.linspace(0, len(flux), len(flux))

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(time, E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(time, N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(time, temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(time, scattering_events)

	plt.title('Scattering events in time')
	plt.xlabel('Time [ps]')
	plt.ylabel('Scattering events [#]')

	plt.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.94, wspace=0.2, hspace=0.34)
	plt.show()

	#T plot
	plt.figure(figsize=(10, 6))
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	#Average for each subcell in equilibrium

	average_T_cells = []
	equil = 7500

	for i in range(int(round(Lx/Lx_subcell, 0))):
		average_T_cells.append(np.mean(T_cells[equil : , i]))

	total_avg_T = np.mean(T_cells[equil:])

	plt.plot(x, average_T_cells, ls='-', color='k', marker='s', label='Average cell T %.2f' % total_avg_T)
	plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at %.2f ps' % float(time[1]))
	#plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at %.2f ps' % float(time[1000]))
	#plt.plot(x, T_cells[5000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at %.2f ps' % float(time[5000]))
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at %.2f ns' % float(time[-1] * 1e-3))

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()

	plt.plot(time_flux, flux, ls='--', color='c', label='')
	plt.title('Flux')
	plt.show()

	plt.plot(time, elapsed_time, ls='', marker='x', color='g')

	slope, intercept, r_value, p_value, std_error = stats.linregress(time, elapsed_time)

	y_regre = np.array(time) * slope + intercept

	plt.plot(time, y_regre, color='r', label='y=%.2fx+%.2f' % (slope, intercept))

	plt.title('Elapsed time')
	plt.legend()
	plt.show()

def conductivity_plots(folders):

	conductivities = []
	lengths = []

	for name in folders:

		os.chdir(current_dir + '/' + name)

		Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

		E = np.load('Energy.npy')
		N = np.load('Phonons.npy')
		T_cells = np.load('Subcell_Ts.npy')
		scattering_events = np.load('Scattering_events.npy')
		temperatures = np.load('Temperatures.npy')
		elapsed_time = np.load('Elapsed_time.npy')
		flux = np.load('Flux.npy')

		time = np.linspace(0, t_MAX*1e12, int(round(t_MAX/dt, 0)))
		time_flux = np.linspace(0, t_MAX*1e12, int(round(t_MAX/(dt * every_flux))))

		flux_avg = np.mean(flux)
		delta_T = abs(T0 - Tf)
		A = Lz * Ly

		conductivity = flux_avg * Lx / (delta_T)

		conductivities.append(conductivity)
		lengths.append(Lx*1e9)

	plt.plot(lengths, conductivities, ls='', marker='s', color='r')
	
	plt.xlabel('Length [nm]')
	plt.show()

def conductivity_plots_nosim(folders):
	conductivities = []
	lengths = []

	for name in folders:

		os.chdir(current_dir + '/' + name)

		Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

		V = Lx_subcell * Ly_subcell * Lz_subcell

		v_avg = np.load('v_avg.npy')
		CV = np.load('C_V.npy')
		MFP = np.load('MFP.npy')


		k = []
		conductivity = 0

		for i in range(len(MFP)):
			conductivity += v_avg[i] * CV[i] * MFP[i]

		conductivities.append((1/3) * conductivity)
		lengths.append(Lx*1e9)

	plt.plot(lengths, conductivities, ls='', marker='s', color='r')
	
	plt.xlabel('Length [nm]')
	plt.show()

def Ballistic_to_diffusive_Lx_scalled(folder_filenames):

	colors = ['y','r', 'b', 'g']
	markers = ['x','*', 'o', 's']

	k = 0

	current_dir_now = current_dir + '/Ballistic_to_diffusive_Lx'

	plt.figure(figsize=(10, 6))

	for name in folder_filenames:

		os.chdir(current_dir_now + '/' + name)

		Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

		E = np.load('Energy.npy')
		N = np.load('Phonons.npy')
		T_cells = np.load('Subcell_Ts.npy')
		scattering_events = np.load('Scattering_events.npy')
		temperatures = np.load('Temperatures.npy')
		elapsed_time = np.load('Elapsed_time.npy')

		time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

		#T plot

		N_cells = int(round(Lx/Lx_subcell, 0))
		x = np.linspace(0, 1, N_cells)

		#Average for each subcell in equilibrium

		average_T_cells = []
		equil = 5000

		for i in range(int(round(Lx/Lx_subcell, 0))):
			average_T_cells.append(np.mean(T_cells[equil : , i]))

		total_avg_T = np.mean(T_cells[equil:])

		label = r'$t=%.f$ ns' % (t_MAX * 1e9)

		plt.plot(x, average_T_cells, ls='-', color=colors[k], marker='None', label='Lx=%.f nm' % (Lx * 1e9))
		#plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='None', label=label)

		k += 1

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), 1), color='k', label='Diffusive regime')

	plt.title('Steady state temperature')
	plt.xlabel(r'$x/L_x$')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()


	#plt.suptitle('Domain temperature evolution')
	#plt.show()

def Ballistic_to_diffusive_high_T_scalled(folder_filenames):

	colors = ['y','r', 'b', 'g', 'c', 'purple']
	markers = ['x','*', 'o', 's']

	k = 0

	current_dir_now = current_dir + '/Ballistic_to_diffusive_high_T'

	plt.figure(figsize=(10, 6))

	for name in folder_filenames:

		os.chdir(current_dir_now + '/' + name)

		Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux = get_parameters('parameters_used.txt')

		E = np.load('Energy.npy')
		N = np.load('Phonons.npy')
		T_cells = np.load('Subcell_Ts.npy')
		scattering_events = np.load('Scattering_events.npy')
		temperatures = np.load('Temperatures.npy')
		elapsed_time = np.load('Elapsed_time.npy')

		time = np.linspace(0, t_MAX*1e12, int(t_MAX/dt))

		#T plot

		N_cells = int(round(Lx/Lx_subcell, 0))
		x = np.linspace(0, 1, N_cells)

		#Average for each subcell in equilibrium

		average_T_cells = []
		equil = 1000

		for i in range(int(round(Lx/Lx_subcell, 0))):
			average_T_cells.append(np.mean(T_cells[equil : , i]))

		total_avg_T = np.mean(T_cells[equil:])

		label = r'$t=%.f$ ns' % (t_MAX * 1e9)

		if k == 0:
			plt.plot(x, average_T_cells, ls='-', color=colors[k], marker='None', label='Lx=%.1f nm' % (Lx * 1e9))

		else:
			plt.plot(x, average_T_cells, ls='-', color=colors[k], marker='None', label='Lx=%.f nm' % (Lx * 1e9))			
		#plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='None', label=label)

		k += 1

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), 1), color='k', label='Diffusive regime')

	plt.title('Steady state temperature')
	plt.xlabel(r'$x/L_x$')
	plt.ylabel('Temperature [K]')

	plt.legend()
	plt.show()


	#plt.suptitle('Domain temperature evolution')
	#plt.show()

def comparison_elapsed_time():

	os.chdir('Ballistic_1D_GrayModel')
	elapsed_time = np.load('Elapsed_time.npy')

	os.chdir(current_dir)
	os.chdir('Ballistic_1D_GrayModel_new')
	elapsed_time_new = np.load('Elapsed_time.npy')	

	plt.plot(np.linspace(0, len(elapsed_time), len(elapsed_time)), elapsed_time, label='GrayModel')
	plt.plot(np.linspace(0, len(elapsed_time_new), len(elapsed_time_new)), elapsed_time_new, label='GrayModel_new')

	plt.title('Elapsed time comparison')
	plt.legend()
	plt.show()

def animated_ballistic_1D():
	os.chdir(current_dir + '/Ballistic_1D')

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W = get_parameters('parameters_used.txt')

	T_cells = np.load('Subcell_Ts.npy')

	def update(t, x, lines):

		lines[0].set_data(x, T_cells[int(t)][:, int(Ly / 2), int(Lz/2)])
		lines[1].set_data(x, diffussive_T(x, T0, Tf, Lx))
		lines[2].set_data(x, np.linspace(balistic_T(T0, Tf), balistic_T(T0, Tf), len(x)))
		lines[3].set_text('Time step %i of %i' % (t, len(T_cells)))

		return lines

	# Attaching 3D axis to the figure
	fig, ax = plt.subplots()

	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	lines = [ax.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, T0, Tf, Lx), label='Diffusive')[0],
	ax.plot(x, np.linspace(balistic_T(T0, Tf), balistic_T(T0, Tf), len(x)), ls='--', color='k', label='Ballistic')[0], ax.text(0, Tf, '', color='k', fontsize=10)]

	ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, len(T_cells), len(T_cells)),
	                    blit=True, interval=1, repeat=False)
	#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
	plt.legend(loc = 'upper right')
	plt.show()


folders_diffusive = ['Diffusive_1D_T_low', 'Diffusive_1D_T_high/OUTPUTS', 'Diffusive_1D_T_high/OUTPUTS_2']

folders_ballistic = ['Ballistic_1D', 'Ly_10_nm', 'Ly_5_nm', 'Ly_1_nm', 'Ly_0.5_nm/OUTPUTS',]

folders_conductivity = ['Ballistic_to_diffusive_Lx/Lx_100_nm/OUTPUTS', 'Ballistic_to_diffusive_Lx/Lx_500_nm/OUTPUTS',
						'Ballistic_to_diffusive_Lx/Lx_1000_nm/OUTPUTS']

folders_diffusive_high_T = ['Lx_0.1_nm/OUTPUTS', 'Lx_1_nm/OUTPUTS', 'Lx_10_nm/OUTPUTS', 'Lx_50_nm/OUTPUTS', 
							'Lx_100_nm/OUTPUTS']

folders_conductivity_T_high = ['Ballistic_to_diffusive_high_T/Lx_1_nm/OUTPUTS', 'Ballistic_to_diffusive_high_T/Lx_10_nm/OUTPUTS',
						'Ballistic_to_diffusive_high_T/Lx_50_nm/OUTPUTS', 'Ballistic_to_diffusive_high_T/Lx_100_nm/OUTPUTS',
						'Ballistic_to_diffusive_high_T/Lx_0.1_nm/OUTPUTS', 'Ballistic_to_diffusive_high_T/Lx_200_nm/OUTPUTS', 'Ballistic_to_diffusive_high_T/Lx_300_nm/OUTPUTS',
						'Ballistic_to_diffusive_high_T/Lx_400_nm/OUTPUTS', 'Ballistic_to_diffusive_high_T/Lx_450_nm/OUTPUTS', 'Ballistic_to_diffusive_high_T/Lx_500_nm/OUTPUTS']

folders_conductivity_T_high_restart = ['Ballistic_to_diffusive_high_T/Lx_1_nm/last_restart', 'Ballistic_to_diffusive_high_T/Lx_10_nm/last_restart',
						'Ballistic_to_diffusive_high_T/Lx_50_nm/last_restart', 'Ballistic_to_diffusive_high_T/Lx_100_nm/last_restart',
						'Ballistic_to_diffusive_high_T/Lx_0.1_nm/last_restart', 'Ballistic_to_diffusive_high_T/Lx_200_nm/last_restart', 'Ballistic_to_diffusive_high_T/Lx_300_nm/last_restart',
						'Ballistic_to_diffusive_high_T/Lx_400_nm/last_restart', 'Ballistic_to_diffusive_high_T/Lx_500_nm/last_restart']

#Ballistic_regime_1D('Ballistic_boundary_diff_Ly_1_nm')
#Diffusive_regime_1D('Ballistic_to_diffusive_high_T/Lx_100_nm/OUTPUTS')

#Ballistic_to_diffusive_1D_roughness(folders_ballistic)
#Ballistic_to_diffusive_Lx(['Ballistic_1D', 'Lx_100_nm/OUTPUTS', 'Lx_500_nm/OUTPUTS', 'Lx_1000_nm/OUTPUTS'])
#Ballistic_to_diffusive_Lx_scalled(['Ballistic_1D', 'Lx_100_nm/OUTPUTS', 'Lx_500_nm/OUTPUTS', 'Lx_1000_nm/OUTPUTS'])
#Ballistic_to_diffusive_high_T_scalled(folders_diffusive_high_T)

#all_plots('/DIFFUSIVE/500_nm/OUTPUTS')
plots_restart('/DIFFUSIVE/500_nm/restart_1000', 1000)

#conductivity_plots(folders_conductivity_T_high)
#conductivity_plots_nosim(folders_conductivity_T_high_restart)

#comparison_elapsed_time()

#animated_ballistic_1D()

#plt.contourf(T_cells[:, :, 0], cmap='hot')
#plt.colorbar()
#plt.show()