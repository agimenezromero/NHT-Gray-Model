import numpy as np
import math
import random
import os
from scipy import integrate
from numpy import linalg as LA

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

import time

current_dir = os.getcwd()
data_folder = current_dir + '/Data'
array_folder = data_folder + '/Arrays'
final_arrays_folder = data_folder + '/Final_arrays'

if not os.path.exists(data_folder): os.mkdir(data_folder)
if not os.path.exists(array_folder): os.mkdir(array_folder)
if not os.path.exists(final_arrays_folder): os.mkdir(final_arrays_folder)

current_time = lambda: round(time.time(), 2)

############################################
#										   #
#			 General Functions			   #
#										   #
############################################


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

def balistic_T_2(T0, Tf):
	'''
		Computes the steady state temperature for the balistic regime
		from the Boltzmann Law
	'''
	return ((T0**4.1 + Tf**4.1)/2)**(1/4)


############################################
#										   #
#				  Classes			       #
#										   #
############################################

class PhononGas(object):
	def __init__(self, Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W):
		self.Lx = float(Lx) #x length of the box
		self.Ly = float(Ly) #y length of the box
		self.Lz = float(Lz) #z lenght of the box
		self.T0 = float(T0) #Temperature of the initial sub-cell (Boundary)
		self.Tf = float(Tf) #Temperature of the last sub-cell (Boundary)
		self.Ti = float(Ti) #Initial temperature of studied subcells
		self.t_MAX = float(t_MAX) #Maximum simulation time
		self.dt = float(dt) #Step size
		self.W = float(W) #Weighting factor

		self.Nt = int(self.t_MAX / self.dt) #Number of simulation steps/iterations

		self.Lx_subcell = float(Lx_subcell) #x length of each subcell
		self.Ly_subcell = float(Ly_subcell) #x length of each subcell
		self.Lz_subcell = float(Lz_subcell) #x length of each subcell

		self.N_subcells = int(self.Lx / self.Lx_subcell) #Number of subcells

		self.V_subcell = self.Ly_subcell * self.Lz_subcell * self.Lx_subcell

		self.r = [] #list of the positions of all the phonons
		self.v = [] #list of the velocities of all the phonons

		self.E = []
		self.N = []
		self.w_avg = []
		self.v_avg = []
		self.C_V = []
		self.MFP = []

		self.scattering_time = []
		self.subcell_Ts = []

		#self.subcell_Ts_2 = np.ones((int(self.Lx/self.Lx_subcell), int(self.Ly/self.Ly_subcell), int(self.Lz / self.Lz_subcell)))

		#Load arrays
		os.chdir(array_folder)

		#Silicon
		self.N_si = np.load('N_si.npy')
		self.E_si = np.load('E_si.npy')
		self.w_si = np.load('w_si.npy')
		self.v_si = np.load('v_si.npy')
		self.CV_si = np.load('CV_si.npy')
		self.MFP_si = np.load('MFP_si.npy')
		self.Etot_si = np.load('Etot_si.npy')

		#Germanium
		self.N_ge = np.load('N_ge.npy')
		self.E_ge = np.load('E_ge.npy')
		self.w_ge = np.load('w_ge.npy')
		self.v_ge = np.load('v_ge.npy')
		self.CV_ge = np.load('CV_ge.npy')
		self.MFP_ge = np.load('MFP_ge.npy')
		self.Etot_ge = np.load('Etot_ge.npy')

		#Temperature array
		self.Ts = np.load('T_ge.npy')

		#Account for the different volumes
		self.N_ge *= self.V_subcell 
		self.CV_ge *= self.V_subcell
		self.Etot_ge *= self.V_subcell

		self.N_si *= self.V_subcell 
		self.CV_si *= self.V_subcell
		self.Etot_si *= self.V_subcell


		#Maximum energies
		self.E_max_si = 3.1752e-21
		self.E_max_ge = 1.659e-21

	def create_phonons(self, N, subcell, T):
		r = np.zeros((N, 3)) #Array of vector positions
		v = np.zeros((N, 3)) #Array of vector velocities

		rx = np.random.random((N,)) * self.Lx_subcell + subcell * self.Lx_subcell
		ry = np.random.random((N,)) * self.Ly
		rz = np.random.random((N,)) * self.Lz

		pos = self.find_T(T, self.Ts)

		for j in range(N):
			r[j][0] = rx[j]
			r[j][1] = ry[j]
			r[j][2] = rz[j]

			self.E.append(self.E_ge[pos])
			self.v_avg.append(self.v_ge[pos])
			self.w_avg.append(self.w_ge[pos])
			self.C_V.append(self.CV_ge[pos])
			self.MFP.append(self.MFP_ge[pos])
			self.scattering_time.append(0.)

		v_polar = np.random.random((N, 2))

		v[:,0] = (np.sin(v_polar[:,0] * np.pi) * np.cos(v_polar[:,1] * 2 * np.pi)) 
		v[:,1] = (np.sin(v_polar[:,0] * np.pi) * np.sin(v_polar[:,1] * 2 * np.pi)) 
		v[:,2] = np.cos(v_polar[:,0] * np.pi)

		v *= self.v_ge[pos]

		self.r += list(r)
		self.v += list(v)

	def init_particles(self):

		for i in range(self.N_subcells):

			if i == 0:
				T_i = self.T0
				self.subcell_Ts.append(self.T0)

			elif i == self.N_subcells - 1:
				T_i = self.Tf
				self.subcell_Ts.append(self.Tf)

			else:
				T_i = self.Ti
				self.subcell_Ts.append(self.Ti)

			pos = self.find_T(T_i, self.Ts)

			N = int(self.N_ge[pos] / self.W)

			self.create_phonons(N, i, T_i)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

		return self.r, self.v, self.E, self.v_avg, self.w_avg, self.C_V, self.MFP

	def check_boundaries(self, i, Lx, Ly, Lz):

		if self.r[i][0] >= Lx or self.r[i][0] < 0:
			self.v[i][0] *= -1.

			if self.r[i][0] > Lx:
				self.r[i][0] = Lx
			else:
				self.r[i][0] = 0

		if self.r[i][1] > Ly or self.r[i][1] < 0:
			self.v[i][1] *= -1.

			if self.r[i][1] > Ly:
				delta_y = self.r[i][1] - Ly
				self.r[i][1] = self.r[i][1] - 2*delta_y
			else:
				delta_y = -self.r[i][1] 
				self.r[i][1] = delta_y

		if self.r[i][2] > Lz or self.r[i][2] < 0:
			self.v[i][2] *= -1.

			if self.r[i][2] > Lz:
				delta_z = self.r[i][2] - Lz
				self.r[i][2] = self.r[i][2] - 2*delta_z
			else:
				delta_z = -self.r[i][2] 
				self.r[i][2] = delta_z

	def find_T(self, value, T): #For a given value of temperature returns the position in the T array
		for i in range(len(T)):
			if T[i] >= value:
				return i
		
	def match_T(self, value, E, T):
		for i in range(len(E)):
			if E[i] == value:
				return T[i]

			elif E[i] > value: #If we exceed the value, use interpolation
				return T[i] * value /  E[i]

	def calculate_subcell_T(self, first_cell, last_cell):
		E_subcells = []
		N_subcells = []

		for i in range(first_cell, last_cell): #Don't take into acount the hot and cold cells
			E = 0
			N = 0
			for j in range(len(self.r)):
				if self.r[j][0] >= i * self.Lx_subcell and self.r[j][0] < (i + 1) * self.Lx_subcell:
					E += self.W * self.E[j]
					N += self.W

			if N == 0:
				E_N = 0
			else:
				E_N = E / N

			self.subcell_Ts[i] = (self.match_T(E_N, self.E_ge, self.Ts))	
			E_subcells.append(E)
			N_subcells.append(N)

		return E_subcells, N_subcells

	def calculate_subcell_T_2(self, first_cell, last_cell):
		N_subcells_right = []
		N_subcells_left = []
		Ts_right = []
		Ts_left = []

		for i in range(first_cell, last_cell): #Don't take into acount the hot and cold cells
			E_right = 0
			E_left = 0
			N_right = 0
			N_left = 0

			for j in range(len(self.r)):
				if self.r[j][0] >= i * self.Lx_subcell and self.r[j][0] <= (i + 1) * self.Lx_subcell and self.v[j][0] > 0: #dreta
					E_right += self.W * self.E[j]
					N_right += self.W

				elif self.r[j][0] >= i * self.Lx_subcell and self.r[j][0] <= (i + 1) * self.Lx_subcell and self.v[j][0] < 0: #esquerra
					E_left += self.W * self.E[j]
					N_left += self.W

			if N_right == 0:
				E_N_right = 0

			if N_left == 0:
				E_N_left = 0

			if N_right > 0 :
				E_N_right = E_right / N_right

			if N_left > 0:
				E_N_left = E_left / N_left

			Ts_right.append(self.match_T(E_N_right, self.E_ge, self.Ts))
			Ts_left.append(self.match_T(E_N_left, self.E_ge, self.Ts))

			N_subcells_right.append(N_right)
			N_subcells_left.append(N_left)

		return N_subcells_right, N_subcells_left, Ts_right, Ts_left

	def find_subcell(self, i):
		for j in range(1, self.N_subcells - 1):
			if self.r[i][0] >=  j * self.Lx_subcell and self.r[i][0] <= (j + 1) * self.Lx_subcell: #It will be in the j_th subcell
				return j	

	def find_subcell_flux(self, i, r):
		for j in range(0, self.N_subcells):
			if r[i][0] >=  j * self.Lx_subcell and r[i][0] <= (j + 1) * self.Lx_subcell: #It will be in the j_th subcell
				return j

	def scattering(self):
		scattering_events = 0

		for i in range(len(self.r)):
			if self.r[i][0] < self.Lx_subcell or self.r[i][0] > (self.N_subcells - 1) * self.Lx_subcell :
				pass #Avoid scattering for phonons in hot and cold boundary cells

			else:
				prob = 1 - np.exp(-self.v_avg[i] * self.scattering_time[i] / self.MFP[i])
				
				dice = random.uniform(0, 1)

				if prob > dice :#Scattering process

					v_polar = np.random.random((1, 2))

					self.v[i][0] = (np.sin(v_polar[:,0] * np.pi) * np.cos(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][1] = (np.sin(v_polar[:,0] * np.pi) * np.sin(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][2] = np.cos(v_polar[:,0] * np.pi)

					current_subcell = self.find_subcell(i)
					current_T = self.subcell_Ts[current_subcell]
					pos = self.find_T(current_T, self.Ts)

					self.v[i] *= self.v_ge[pos]

					self.v_avg[i] = self.v_ge[pos]
					self.w_avg[i] = self.w_ge[pos]
					self.E[i] = self.E_ge[pos]
					self.C_V[i] = self.CV_ge[pos]
					self.MFP[i] = self.MFP_ge[pos]

					self.scattering_time[i] = 0. #Re-init scattering time

					scattering_events += self.W

				else:
					self.scattering_time[i] += self.dt #Account for the scattering time

		return scattering_events

	def energy_conservation(self, delta_E):
		for i in range(1, self.N_subcells - 1):

			if delta_E[i - 1] > self.E_max_ge: #Deletion of phonons
				E_sobrant = delta_E[i - 1]

				T = self.subcell_Ts[i]
				pos_T = self.find_T(T, self.Ts)

				E_phonon_T = self.E_ge[pos_T] #Energy per phonon for this subcell T

				N_phonons = int(round(E_sobrant / (E_phonon_T * self.W), 0)) #Number of phonons to delete

				if N_phonons != 0:

					array_position_phonons_ith_subcell = []
					counter = 0

					for j in range(len(self.r)):
						if counter == N_phonons: 
								break
						if self.r[j][0] > i * self.L_subcell and self.r[j][0] < (i + 1) * self.L_subcell: #is in the i_th subcell

							counter += 1
							array_position_phonons_ith_subcell.append(j) #position in self.r array

					self.r = np.delete(self.r, array_position_phonons_ith_subcell, 0)
					self.v = np.delete(self.v, array_position_phonons_ith_subcell, 0)
					self.E = np.delete(self.E, array_position_phonons_ith_subcell, 0)
					self.v_avg = np.delete(self.v_avg, array_position_phonons_ith_subcell, 0)
					self.w_avg = np.delete(self.w_avg, array_position_phonons_ith_subcell, 0)
					self.C_V = np.delete(self.C_V, array_position_phonons_ith_subcell, 0)
					self.MFP = np.delete(self.MFP, array_position_phonons_ith_subcell, 0)
					self.scattering_time = np.delete(self.scattering_time, array_position_phonons_ith_subcell, 0)

			elif -delta_E[i - 1] > self.E_max_ge: #Production of phonons
				E_sobrant = -delta_E[i - 1]

				T = self.subcell_Ts[i]
				pos_T = self.find_T(T, self.Ts)

				E_phonon_T = self.E_ge[pos_T] #Energy per phonon for this subcell T

				N_phonons = int(round(E_sobrant / (E_phonon_T * self.W), 0)) #Number of phonons to create

				if N_phonons != 0:

					self.r = list(self.r)
					self.v = list(self.v)
					self.v_avg = list(self.v_avg)
					self.w_avg = list(self.w_avg)
					self.E = list(self.E)
					self.C_V = list(self.C_V)
					self.MFP = list(self.MFP)
					self.scattering_time = list(self.scattering_time)

					self.create_phonons(N_phonons, i, T)

					self.r = np.array(self.r)
					self.v = np.array(self.v)
					self.v_avg = np.array(self.v_avg)
					self.w_avg = np.array(self.w_avg)
					self.E = np.array(self.E)
					self.C_V = np.array(self.C_V)
					self.MFP = np.array(self.MFP)
					self.scattering_time = np.array(self.scattering_time)

	def re_init_boundary(self): #Eliminar tots i posar tots nous
		pos_T0 = self.find_T(self.T0, self.Ts)
		pos_Tf = self.find_T(self.Tf, self.Ts)

		N_0 = int(round(self.N_ge[pos_T0] / self.W, 0))
		N_f = int(round(self.N_ge[pos_Tf] / self.W, 0))

		total_indexs = []

		#Delete all the phonons in boundary subcells
		for i in range(len(self.r)):
			if self.r[i][0] <= self.Lx_subcell: #Subcell with T0 boundary
				total_indexs.append(i)

			elif self.r[i][0] >= (self.N_subcells - 1) * self.Lx_subcell: #Subcell with Tf boundary
				total_indexs.append(i)

		self.r = np.delete(self.r, total_indexs, 0)
		self.v = np.delete(self.v, total_indexs, 0)
		self.E = np.delete(self.E, total_indexs, 0)
		self.v_avg = np.delete(self.v_avg, total_indexs, 0)
		self.w_avg = np.delete(self.w_avg, total_indexs, 0)
		self.C_V = np.delete(self.C_V, total_indexs, 0)
		self.MFP = np.delete(self.MFP, total_indexs, 0)
		self.scattering_time = np.delete(self.scattering_time, total_indexs, 0)

		#Create the new phonons
		self.r = list(self.r)
		self.v = list(self.v)
		self.E = list(self.E)
		self.v_avg = list(self.v_avg)
		self.w_avg = list(self.w_avg)
		self.C_V = list(self.C_V)
		self.MFP = list(self.MFP)
		self.scattering_time = list(self.scattering_time)

		self.create_phonons(N_0, 0, self.T0)
		self.create_phonons(N_f, self.N_subcells - 1, self.Tf)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

	def simulation(self):
		self.init_particles()

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []

		for k in range(self.Nt):
			print(k)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i, self.Lx, self.Ly, self.Lz)

			#interface_scattering()
			#energy_conservation()

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T(1, self.N_subcells - 1) #Calculate energy before scattering

			scattering_events.append(self.scattering())

			E_subcells_new , N_subcells_new = self.calculate_subcell_T(1, self.N_subcells - 1) #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			self.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = self.calculate_subcell_T(1, self.N_subcells - 1) #Calculate final T

			delta_E_final = np.array(E_subcells_final) - np.array(E_subcells)

			delta_energy.append(np.mean(delta_E_final))
			Energy.append(np.sum(E_subcells_final))
			Phonons.append(np.sum(N_subcells_final))
			Temperatures.append(np.mean(self.subcell_Ts[1: self.N_subcells - 1]))

			'''
			if k == self.Nt - 1:
				N_subcells_right, N_subcells_left, Ts_right, Ts_left = self.calculate_subcell_T_2(0, self.N_subcells)

				plt.plot(np.linspace(0, len(self.E), len(self.E)), self.E, ls='', marker='.')
				plt.show()
				
				x = np.linspace(0, len(delta_energy), len(delta_energy))
				avg = np.mean(delta_energy)

				E_min = self.E_ge[self.find_T(self.Tf, self.Ts)] * 1e22

				plt.subplot(1, 2, 1)
				plt.plot(x, delta_energy, ls='-', marker='.', label=r'$E_N=%.2f·10^{-22}$ for $T_f=%.2f$' % (E_min, self.Tf))
				plt.plot(x, np.linspace(avg, avg, len(delta_energy)), ls='--', color='k', label='Avg')
				plt.title(r'$\Delta E$ scattering process')
				plt.legend()

				plt.subplot(1, 2, 2)
				plt.plot(np.linspace(0, self.Nt, self.Nt), scattering_events)
				plt.title('Scattering events in time')
				plt.show()
			'''

		os.chdir(final_arrays_folder)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', self.subcell_Ts)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)

		#return self.subcell_Ts, Energy, Phonons, Temperatures, N_subcells_right, N_subcells_left, Ts_right, Ts_left

	def flux(self, i, r_ant):
		pos = self.find_subcell(i)
		pos_ant = self.find_subcell_flux(i, r_ant)

		
		if pos != pos_ant:
			return self.E[i]

		else:
			return 0

	def simulation_heat_flux(self):
		self.init_particles()

		flux_t = []

		for k in range(self.Nt):
			print(k)

			r_ant = np.zeros((len(self.r), 3))
			r_ant[:,:] = self.r[:,:]

			self.r += self.dt * self.v #Drift

			E_transfered = []

			for i in range(len(self.r)):
				self.check_boundaries(i, self.Lx, self.Ly, self.Lz)
				E_transfered.append(self.flux(i, r_ant)) 

			flux_t.append(np.mean(E_transfered) / (self.Ly * self.Lz))	

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T(1, self.N_subcells - 1) #Calculate energy before scattering

			E_subcells_new , N_subcells_new = self.calculate_subcell_T(1, self.N_subcells - 1) #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			self.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = self.calculate_subcell_T(1, self.N_subcells - 1) #Calculate final Tf

		return flux_t

	def animation(self):
		self.init_particles()

		def update(t, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i, self.Lx, self.Ly, self.Lz)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T(1, self.N_subcells - 1)

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T(1, self.N_subcells - 1)

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T(1, self.N_subcells - 1)

			lines[0].set_data(self.r[:,0], self.r[:,1])
			lines[0].set_3d_properties(self.r[:,2])

			return lines

		# Attaching 3D axis to the figure
		fig = plt.figure()
		ax = p3.Axes3D(fig)


		# Setting the axes properties
		ax.set_xlim3d([0, self.Lx])
		ax.set_xlabel('X')

		ax.set_ylim3d([0, self.Ly])
		ax.set_ylabel('Y')

		ax.set_zlim3d([0, self.Lz])
		ax.set_zlabel('Z')

		lines = []
		lines.append(ax.plot(self.r[:,0], self.r[:,1], self.r[:,2], ls='None', marker='.', label='Phonons')[0])

		ani = FuncAnimation(fig, update, fargs=(lines,), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=100, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper left')
		plt.show()

		return self.r, self.subcell_Ts

	def animation_2(self):
		self.init_particles()

		def update(t, x, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i, self.Lx, self.Ly, self.Lz)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T(1, self.N_subcells - 1)

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T(1, self.N_subcells - 1)

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			#self.energy_conservation(delta_E)

			self.calculate_subcell_T(1, self.N_subcells - 1)

			lines[0].set_data(x, self.subcell_Ts)
			lines[1].set_data(x, diffussive_T(x, self.T0, self.Tf, self.Lx))
			lines[2].set_text('Time step %i of %i' % (k, self.Nt))

			return lines

		# Attaching 3D axis to the figure
		fig, ax = plt.subplots()

		x = np.linspace(0, self.Lx, self.N_subcells)

		lines = [ax.plot(x, self.subcell_Ts, '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, self.T0, self.Tf, self.Lx))[0],
		ax.text(0, self.Tf, '', color='k', fontsize=10)]

		ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=1, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper right')
		plt.show()

		return self.r, self.subcell_Ts

class PhononGas_2(object):
	def __init__(self, Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W):
		self.Lx = float(Lx) #x length of the box
		self.Ly = float(Ly) #y length of the box
		self.Lz = float(Lz) #z lenght of the box
		self.T0 = float(T0) #Temperature of the initial sub-cell (Boundary)
		self.Tf = float(Tf) #Temperature of the last sub-cell (Boundary)
		self.Ti = float(Ti) #Initial temperature of studied subcells
		self.t_MAX = float(t_MAX) #Maximum simulation time
		self.dt = float(dt) #Step size
		self.W = float(W) #Weighting factor

		self.Nt = int(self.t_MAX / self.dt) #Number of simulation steps/iterations

		self.Lx_subcell = float(Lx_subcell) #x length of each subcell
		self.Ly_subcell = float(Ly_subcell) #x length of each subcell
		self.Lz_subcell = float(Lz_subcell) #x length of each subcell

		self.N_subcells_x = int(round(self.Lx / self.Lx_subcell, 0)) #Number of subcells
		self.N_subcells_y = int(round(self.Ly / self.Ly_subcell, 0))
		self.N_subcells_z = int(round(self.Lz / self.Lz_subcell, 0)) 

		self.V_subcell = self.Ly_subcell * self.Lz_subcell * self.Lx_subcell

		self.r = [] #list of the positions of all the phonons
		self.v = [] #list of the velocities of all the phonons

		self.E = []
		self.N = []
		self.w_avg = []
		self.v_avg = []
		self.C_V = []
		self.MFP = []

		self.scattering_time = []

		self.subcell_Ts = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))

		#Load arrays
		os.chdir(array_folder)

		#Silicon
		self.N_si = np.load('N_si.npy')
		self.E_si = np.load('E_si.npy')
		self.w_si = np.load('w_si.npy')
		self.v_si = np.load('v_si.npy')
		self.CV_si = np.load('CV_si.npy')
		self.MFP_si = np.load('MFP_si.npy')
		self.Etot_si = np.load('Etot_si.npy')

		#Germanium
		self.N_ge = np.load('N_ge.npy')
		self.E_ge = np.load('E_ge.npy')
		self.w_ge = np.load('w_ge.npy')
		self.v_ge = np.load('v_ge.npy')
		self.CV_ge = np.load('CV_ge.npy')
		self.MFP_ge = np.load('MFP_ge.npy')
		self.Etot_ge = np.load('Etot_ge.npy')

		#Temperature array
		self.Ts = np.load('T_ge.npy')

		#Account for the different volumes
		self.N_ge *= self.V_subcell 
		self.CV_ge *= self.V_subcell
		self.Etot_ge *= self.V_subcell

		self.N_si *= self.V_subcell 
		self.CV_si *= self.V_subcell
		self.Etot_si *= self.V_subcell

		#Maximum energies
		self.E_max_si = 3.1752e-21
		self.E_max_ge = 1.659e-21

	def find_T(self, value, T): 
		'''
		For a given value of temperature returns the position in the T array
		'''
		
		for i in range(len(T)):
			if T[i] >= value:
				return i

	def create_phonons(self, N, subcell_x, subcell_y, subcell_z, T):
		r = np.zeros((N, 3)) #Array of vector positions
		v = np.zeros((N, 3)) #Array of vector velocities

		rx = np.random.random((N,)) * self.Lx_subcell + subcell_x * self.Lx_subcell
		ry = np.random.random((N,)) * self.Ly_subcell + subcell_y * self.Ly_subcell
		rz = np.random.random((N,)) * self.Lz_subcell + subcell_z * self.Lz_subcell

		pos = self.find_T(T, self.Ts)

		for j in range(N):
			r[j][0] = rx[j]
			r[j][1] = ry[j]
			r[j][2] = rz[j]

			self.E.append(self.E_ge[pos])
			self.v_avg.append(self.v_ge[pos])
			self.w_avg.append(self.w_ge[pos])
			self.C_V.append(self.CV_ge[pos])
			self.MFP.append(self.MFP_ge[pos])
			self.scattering_time.append(0.)

		v_polar = np.random.random((N, 2))

		v[:,0] = (np.sin(v_polar[:,0] * np.pi) * np.cos(v_polar[:,1] * 2 * np.pi)) 
		v[:,1] = (np.sin(v_polar[:,0] * np.pi) * np.sin(v_polar[:,1] * 2 * np.pi)) 
		v[:,2] = np.cos(v_polar[:,0] * np.pi)

		v *= self.v_ge[pos]

		self.r += list(r)
		self.v += list(v)

	def init_particles(self):

		for i in range(self.N_subcells_x):
			for j in range(self.N_subcells_y):
				for k in range(self.N_subcells_z):

					if i == 0:
						T_i = self.T0
						self.subcell_Ts[i][j][k] = self.T0

					elif i == self.N_subcells_x - 1:
						T_i = self.Tf
						self.subcell_Ts[i][j][k] = self.Tf

					else:
						T_i = self.Ti
						self.subcell_Ts[i][j][k] = self.Ti

					pos = self.find_T(T_i, self.Ts)

					N = int(self.N_ge[pos] / self.W)

					self.create_phonons(N, i, j, k, T_i)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

		return self.r, self.v, self.E, self.v_avg, self.w_avg, self.C_V, self.MFP

	def check_boundaries(self, i, Lx, Ly, Lz):

		if self.r[i][0] >= Lx or self.r[i][0] < 0:
			self.v[i][0] *= -1.

			if self.r[i][0] > Lx:
				self.r[i][0] = Lx
			else:
				self.r[i][0] = 0

		if self.r[i][1] > Ly or self.r[i][1] < 0:
			self.v[i][1] *= -1.

			if self.r[i][1] > Ly:
				delta_y = self.r[i][1] - Ly
				self.r[i][1] = self.r[i][1] - 2*delta_y
			else:
				delta_y = -self.r[i][1] 
				self.r[i][1] = delta_y

		if self.r[i][2] > Lz or self.r[i][2] < 0:
			self.v[i][2] *= -1.

			if self.r[i][2] > Lz:
				delta_z = self.r[i][2] - Lz
				self.r[i][2] = self.r[i][2] - 2*delta_z
			else:
				delta_z = -self.r[i][2] 
				self.r[i][2] = delta_z
		
	def match_T(self, value, E, T):
		for i in range(len(E)):
			if E[i] == value:
				return T[i]

			elif E[i] > value: #If we exceed the value, use interpolation
				return T[i] * value /  E[i]

	def calculate_subcell_T(self):

		E_subcells = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))
		N_subcells = np.zeros((self.N_subcells_x, self.N_subcells_y, self.N_subcells_z))

		for i in range(len(self.r)):
			x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
			y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
			z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

			E_subcells[x][y][z] += self.W * self.E[i]
			N_subcells[x][y][z] += self.W

		for i in range(self.N_subcells_x):
			for j in range(self.N_subcells_y):
				for k in range(self.N_subcells_z):

					self.subcell_Ts[i][j][k] = self.match_T(N_subcells[i][j][k], self.N_ge, self.Ts)

		return E_subcells, N_subcells

	def find_subcell(self, i):
		for j in range(1, self.N_subcells - 1):
			if self.r[i][0] >=  j * self.Lx_subcell and self.r[i][0] <= (j + 1) * self.Lx_subcell: #It will be in the j_th subcell
				return j	

	def scattering(self):
		scattering_events = 0

		for i in range(len(self.r)):
			x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
			y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
			z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

			if x < self.Lx_subcell or x > (self.N_subcells_x - 1) * self.Lx_subcell :
				pass #Avoid scattering for phonons in hot and cold boundary cells

			else:
				prob = 1 - np.exp(-self.v_avg[i] * self.scattering_time[i] / self.MFP[i])
				
				dice = random.uniform(0, 1)

				if prob > dice :#Scattering process

					v_polar = np.random.random((1, 2))

					self.v[i][0] = (np.sin(v_polar[:,0] * np.pi) * np.cos(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][1] = (np.sin(v_polar[:,0] * np.pi) * np.sin(v_polar[:,1] * 2 * np.pi)) 
					self.v[i][2] = np.cos(v_polar[:,0] * np.pi)

					current_T = self.subcell_Ts[x][y][z]
					pos = self.find_T(current_T, self.Ts)

					self.v[i] *= self.v_ge[pos]

					self.v_avg[i] = self.v_ge[pos]
					self.w_avg[i] = self.w_ge[pos]
					self.E[i] = self.E_ge[pos]
					self.C_V[i] = self.CV_ge[pos]
					self.MFP[i] = self.MFP_ge[pos]

					self.scattering_time[i] = 0. #Re-init scattering time

					scattering_events += self.W

				else:
					self.scattering_time[i] += self.dt #Account for the scattering time

		return scattering_events

	def energy_conservation(self, delta_E):
		for i in range(self.N_subcells_x):
			for j in range(self.N_subcells_y):
				for k in range(self.N_subcells_z):

					if delta_E[i][j][k] > self.E_max_ge: #Deletion of phonons
						E_sobrant = delta_E[i]

						T = self.subcell_Ts[i][j][k]
						pos_T = self.find_T(T, self.Ts)

						E_phonon_T = self.E_ge[pos_T] #Energy per phonon for this subcell T

						N_phonons = int(round(E_sobrant / (E_phonon_T * self.W), 0)) #Number of phonons to delete

						if N_phonons != 0:

							array_position_phonons_ith_subcell = []
							counter = 0

							for j in range(len(self.r)):

								x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
								y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
								z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

								if counter == N_phonons: 
										break

								if x == i and y == j and z == k : #is in the i_th subcell

									counter += 1
									array_position_phonons_ith_subcell.append(j) #position in self.r array

							self.r = np.delete(self.r, array_position_phonons_ith_subcell, 0)
							self.v = np.delete(self.v, array_position_phonons_ith_subcell, 0)
							self.E = np.delete(self.E, array_position_phonons_ith_subcell, 0)
							self.v_avg = np.delete(self.v_avg, array_position_phonons_ith_subcell, 0)
							self.w_avg = np.delete(self.w_avg, array_position_phonons_ith_subcell, 0)
							self.C_V = np.delete(self.C_V, array_position_phonons_ith_subcell, 0)
							self.MFP = np.delete(self.MFP, array_position_phonons_ith_subcell, 0)
							self.scattering_time = np.delete(self.scattering_time, array_position_phonons_ith_subcell, 0)

					elif -delta_E[i][j][k] > self.E_max_ge: #Production of phonons
						E_sobrant = -delta_E[i][j][k]

						T = self.subcell_Ts[i][j][k]
						pos_T = self.find_T(T, self.Ts)

						E_phonon_T = self.E_ge[pos_T] #Energy per phonon for this subcell T

						N_phonons = int(round(E_sobrant / (E_phonon_T * self.W), 0)) #Number of phonons to create

						if N_phonons != 0:

							self.r = list(self.r)
							self.v = list(self.v)
							self.v_avg = list(self.v_avg)
							self.w_avg = list(self.w_avg)
							self.E = list(self.E)
							self.C_V = list(self.C_V)
							self.MFP = list(self.MFP)
							self.scattering_time = list(self.scattering_time)

							self.create_phonons(N_phonons, i, j, k, T)

							self.r = np.array(self.r)
							self.v = np.array(self.v)
							self.v_avg = np.array(self.v_avg)
							self.w_avg = np.array(self.w_avg)
							self.E = np.array(self.E)
							self.C_V = np.array(self.C_V)
							self.MFP = np.array(self.MFP)
							self.scattering_time = np.array(self.scattering_time)

	def re_init_boundary(self): #Eliminar tots i posar tots nous
		pos_T0 = self.find_T(self.T0, self.Ts)
		pos_Tf = self.find_T(self.Tf, self.Ts)

		N_0 = int(round(self.N_ge[pos_T0] / self.W, 0))
		N_f = int(round(self.N_ge[pos_Tf] / self.W, 0))

		total_indexs = []

		#Delete all the phonons in boundary subcells
		for i in range(len(self.r)):
			if self.r[i][0] <= self.Lx_subcell: #Subcell with T0 boundary
				total_indexs.append(i)

			elif self.r[i][0] >= (self.N_subcells_x - 1) * self.Lx_subcell: #Subcell with Tf boundary
				total_indexs.append(i)

		self.r = np.delete(self.r, total_indexs, 0)
		self.v = np.delete(self.v, total_indexs, 0)
		self.E = np.delete(self.E, total_indexs, 0)
		self.v_avg = np.delete(self.v_avg, total_indexs, 0)
		self.w_avg = np.delete(self.w_avg, total_indexs, 0)
		self.C_V = np.delete(self.C_V, total_indexs, 0)
		self.MFP = np.delete(self.MFP, total_indexs, 0)
		self.scattering_time = np.delete(self.scattering_time, total_indexs, 0)

		#Create the new phonons
		self.r = list(self.r)
		self.v = list(self.v)
		self.E = list(self.E)
		self.v_avg = list(self.v_avg)
		self.w_avg = list(self.w_avg)
		self.C_V = list(self.C_V)
		self.MFP = list(self.MFP)
		self.scattering_time = list(self.scattering_time)

		for j in range(self.N_subcells_y):
			for k in range(self.N_subcells_z):

				self.create_phonons(N_0, 0, j, k, self.T0)
				self.create_phonons(N_f, self.N_subcells_x - 1, j, k, self.Tf)

		self.r = np.array(self.r)
		self.v = np.array(self.v)
		self.E = np.array(self.E)
		self.v_avg = np.array(self.v_avg)
		self.w_avg = np.array(self.w_avg)
		self.C_V = np.array(self.C_V)
		self.MFP = np.array(self.MFP)
		self.scattering_time = np.array(self.scattering_time)

	def simulation(self):
		self.init_particles()

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []

		for k in range(self.Nt):
			self.calculate_subcell_T()
			print(k)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i, self.Lx, self.Ly, self.Lz)

			#interface_scattering()
			#energy_conservation()

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T() #Calculate energy before scattering

			scattering_events.append(self.scattering())

			E_subcells_new , N_subcells_new = self.calculate_subcell_T() #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			self.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = self.calculate_subcell_T() #Calculate final T

			delta_E_final = np.array(E_subcells_final) - np.array(E_subcells)

			delta_energy.append(np.mean(delta_E_final))
			Energy.append(np.sum(E_subcells_final))
			Phonons.append(np.sum(N_subcells_final))
			Temperatures.append(np.mean(self.subcell_Ts))

			'''
			if k == self.Nt - 1:
				N_subcells_right, N_subcells_left, Ts_right, Ts_left = self.calculate_subcell_T_2(0, self.N_subcells)

				plt.plot(np.linspace(0, len(self.E), len(self.E)), self.E, ls='', marker='.')
				plt.show()
				
				x = np.linspace(0, len(delta_energy), len(delta_energy))
				avg = np.mean(delta_energy)

				E_min = self.E_ge[self.find_T(self.Tf, self.Ts)] * 1e22

				plt.subplot(1, 2, 1)
				plt.plot(x, delta_energy, ls='-', marker='.', label=r'$E_N=%.2f·10^{-22}$ for $T_f=%.2f$' % (E_min, self.Tf))
				plt.plot(x, np.linspace(avg, avg, len(delta_energy)), ls='--', color='k', label='Avg')
				plt.title(r'$\Delta E$ scattering process')
				plt.legend()

				plt.subplot(1, 2, 2)
				plt.plot(np.linspace(0, self.Nt, self.Nt), scattering_events)
				plt.title('Scattering events in time')
				plt.show()
			'''

		os.chdir(final_arrays_folder)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', self.subcell_Ts)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)

		#return self.subcell_Ts, Energy, Phonons, Temperatures, N_subcells_right, N_subcells_left, Ts_right, Ts_left

	def animation(self):
		self.init_particles()

		def update(t, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i, self.Lx, self.Ly, self.Lz)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T(1, self.N_subcells - 1)

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T(1, self.N_subcells - 1)

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T(1, self.N_subcells - 1)

			lines[0].set_data(self.r[:,0], self.r[:,1])
			lines[0].set_3d_properties(self.r[:,2])

			return lines

		# Attaching 3D axis to the figure
		fig = plt.figure()
		ax = p3.Axes3D(fig)


		# Setting the axes properties
		ax.set_xlim3d([0, self.Lx])
		ax.set_xlabel('X')

		ax.set_ylim3d([0, self.Ly])
		ax.set_ylabel('Y')

		ax.set_zlim3d([0, self.Lz])
		ax.set_zlabel('Z')

		lines = []
		lines.append(ax.plot(self.r[:,0], self.r[:,1], self.r[:,2], ls='None', marker='.', label='Phonons')[0])

		ani = FuncAnimation(fig, update, fargs=(lines,), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=100, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper left')
		plt.show()

		return self.r, self.subcell_Ts

if __name__ == '__main__':

	#os.chdir(final_arrays_folder)
	#print(balistic_T_2(10, 20))

	#PARAMETERS
	Lx = 480e-9
	Ly = 200e-9
	Lz = 400e-9

	Lx_subcell = 40e-9
	Ly_subcell = 40e-9
	Lz_subcell = 40e-9

	T0 = 150
	Tf = 100
	Ti = 50

	t_MAX = 1000e-12
	dt = 1e-12

	W = 1e5

	#gas = PhononGas_2(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W)
	#gas.simulation()

	os.chdir('/home/alexgimenez/Àlex/Estudis/Python/Termodynamics and statistical physics/Gray Model/Data/Final_arrays')

	E = np.load('Energy.npy')
	T_cells = np.load('Subcell_Ts.npy')


	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))
	
	#plt.plot(x, T_cells[:, int(Ly / 2), int(Lz/2)], ls='-', marker='^')
	#plt.show()

	plt.contourf(T_cells[:, :, 5], cmap='hot')
	plt.colorbar()
	plt.show()