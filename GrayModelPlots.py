import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

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

		i += 1

	return Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W

current_dir  = os.getcwd()

def Ballistic_regime_1D():
	os.chdir(current_dir + '/Ballistic_1D')

	Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W = get_parameters('parameters_used.txt')

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

	#Subplots
	plt.figure(figsize=(10, 6))

	plt.subplot(2, 2, 1)
	plt.plot(np.linspace(0, len(E), len(E)), E)

	plt.title('Energy evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Energy [J]')

	plt.subplot(2, 2, 2)
	plt.plot(np.linspace(0, len(N), len(N)), N)

	plt.title('Number of phonons evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Phonons [#]')

	plt.subplot(2, 2, 3)
	plt.plot(np.linspace(0, len(temperatures), len(temperatures)), temperatures)

	plt.title('Average temperature evolution')
	plt.xlabel('Time [ps]')
	plt.ylabel('Temperature [K]')

	plt.subplot(2, 2, 4)
	plt.plot(np.linspace(0, len(scattering_events), len(scattering_events)), scattering_events)

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


	#plt.plot(x, average_T_cells, ls='-', color='k', marker='s', label='Average cell T')
	plt.plot(x, T_cells[0][:, int(Ly / 2), int(Lz/2)], ls='-', color='c', marker='*', label='T at 1 ps')
	plt.plot(x, T_cells[1000][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^', label='T at 1 ns')
	plt.plot(x, T_cells[5000][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='o', label='T at 5 ns')
	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='g', marker='s', label='T at 10 ns')

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), 
		balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k', label='Ballistic regime')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k', label='Diffusive regime')

	plt.title('Domain temperature evolution')
	plt.xlabel('Domain length [m]')
	plt.ylabel('Temperature [K]')

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


#Ballistic_regime_1D()
#animated_ballistic_1D()

#plt.contourf(T_cells[:, :, 0], cmap='hot')
#plt.colorbar()
#plt.show()