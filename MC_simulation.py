import numpy as np
import math
import random
import os
from scipy import integrate, stats
from numpy import linalg as LA

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

import time
from GrayModelClasses import PhononGas as pg 

current_dir = os.getcwd()

current_milli_time = lambda: int(round(time.time() * 1000))

os.chdir('/home/alexgimenez/Àlex/Estudis/Python/Termodynamics and statistical physics/Gray Model/Data/Arrays')

print('Loading arrays from:', os.getcwd())


#Silicon
N_si = np.load('N_si.npy')
E_si = np.load('E_si.npy')
w_si = np.load('w_si.npy')
v_si = np.load('v_si.npy')
CV_si = np.load('CV_si.npy')
MFP_si = np.load('MFP_si.npy')
Etot_si = np.load('Etot_si.npy')

#Germanium
N_ge = np.load('N_ge.npy')
E_ge = np.load('E_ge.npy')
w_ge = np.load('w_ge.npy')
v_ge = np.load('v_ge.npy')
CV_ge = np.load('CV_ge.npy')
MFP_ge = np.load('MFP_ge.npy')
Etot_ge = np.load('Etot_ge.npy')

Ts = np.load('T_ge.npy')

E_max_si = 3.1752e-21
E_max_ge = 1.659e-21

os.chdir(current_dir)

#######################################
#									  #
#				CLASSES				  #
#									  #
#######################################

def diffussive_T(T, T0, Tf, xf):
	k = (Tf - T0) / xf

	return k * T + T0

def balistic_T(T0, Tf):
	'''
		Computes the steady state temperature for the balistic regime
		from the Boltzmann Law
	'''
	return ((T0**4 + Tf**4)/2)**(0.25)

def balistic_T_2(T0, Tf, exp):
	'''
		Computes the steady state temperature for the balistic regime
		from the Boltzmann Law
	'''
	return ((T0**exp + Tf**exp)/2)**(1/exp)


class Plots(object):
	def __init__(self):
		pass

	def initial_subplots(self, T_0, T_max, n, N, E, w, v, CV, MFP, name):
		#N(T)
		plt.subplot(3, 2, 1)
		plt.plot(np.linspace(T_0, T_max, n), N)
		#plt.title('Nº phonons vs temperature')
		plt.ylabel('Nº phonons')
		plt.xlabel('T (K)')
		
		#E(T)
		plt.subplot(3, 2, 2)
		plt.plot(np.linspace(T_0, T_max, n), E)
		#plt.title('Energy vs temperature')
		plt.ylabel('E per phonon (J)')
		plt.xlabel('T (K)')

		#w_avg
		plt.subplot(3, 2, 3)
		plt.plot(np.linspace(T_0, T_max, n), w)
		#plt.title('Average frequency vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$\omega_{avg} \, (rad/s)$')

		#v_avg
		plt.subplot(3, 2, 4)
		plt.plot(np.linspace(T_0, T_max, n), v)
		#plt.title('Average group velocity vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$v_{avg} \, (m/s)$')

		#C_V
		plt.subplot(3, 2, 5)
		plt.plot(np.linspace(T_0, T_max, n), CV)
		#plt.title('Heat capacity vs temperature')
		plt.ylabel(r'$C_V$ (J/K)')
		plt.xlabel('T (K)')

		#MFP
		plt.subplot(3, 2, 6)
		plt.plot(np.linspace(T_0, T_max, n), np.log(MFP*1e7))
		#plt.title('Mean Free Path vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$\Lambda \, (m)$')

		plt.suptitle('Thermal transport properties for %s' % name)
		plt.show()

class PhononGas(object):
	def __init__(self, L_subcell, Lx, Ly, Lz, T0, Tf, Ti, t_MAX, dt, W):
		self.Lx = float(Lx) #Length of the box
		self.Ly = float(Ly)
		self.Lz = float(Lz)
		self.T0 = float(T0) #Temperature of the initial sub-cell
		self.Tf = float(Tf) #Temperature of the last sub-cell
		self.Ti = float(Ti) #Initial temperature of studied subcells
		self.t_MAX = float(t_MAX) #Maximum simulation time
		self.dt = float(dt) #Step size
		self.W = float(W) #Weighting factor

		self.Nt = int(self.t_MAX / self.dt) #Number of simulation steps/iterations

		self.L_subcell = float(L_subcell) #X length of each subcell

		self.N_subcells = int(self.Lx / self.L_subcell) #Number of subcells

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

	def create_phonons(self, N, subcell, T):
		r = np.zeros((N, 3)) #Array of vector positions
		v = np.zeros((N, 3)) #Array of vector velocities

		rx = np.random.random((N,)) * self.L_subcell + subcell * self.L_subcell
		ry = np.random.random((N,)) * self.Ly
		rz = np.random.random((N,)) * self.Lz

		pos = self.find_T(T, Ts)

		for j in range(N):
			r[j][0] = rx[j]
			r[j][1] = ry[j]
			r[j][2] = rz[j]

			self.E.append(E_ge[pos])
			self.v_avg.append(v_ge[pos])
			self.w_avg.append(w_ge[pos])
			self.C_V.append(CV_ge[pos])
			self.MFP.append(MFP_ge[pos])
			self.scattering_time.append(0.)

		v_polar = np.random.random((N, 2))

		v[:,0] = (np.sin(v_polar[:,0] * np.pi) * np.cos(v_polar[:,1] * 2 * np.pi)) 
		v[:,1] = (np.sin(v_polar[:,0] * np.pi) * np.sin(v_polar[:,1] * 2 * np.pi)) 
		v[:,2] = np.cos(v_polar[:,0] * np.pi)

		v *= v_ge[pos]

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

			pos = self.find_T(T_i, Ts)

			N = int(N_ge[pos] / self.W)

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
				if self.r[j][0] >= i * self.L_subcell and self.r[j][0] < (i + 1) * self.L_subcell:
					E += self.W * self.E[j]
					N += self.W

			if N == 0:
				E_N = 0
			else:
				E_N = E / N

			self.subcell_Ts[i] = (self.match_T(E_N, E_ge, Ts))	
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
				if self.r[j][0] >= i * self.L_subcell and self.r[j][0] <= (i + 1) * self.L_subcell and self.v[j][0] > 0: #dreta
					E_right += self.W * self.E[j]
					N_right += self.W

				elif self.r[j][0] >= i * self.L_subcell and self.r[j][0] <= (i + 1) * self.L_subcell and self.v[j][0] < 0: #esquerra
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

			Ts_right.append(self.match_T(E_N_right, E_ge, Ts))
			Ts_left.append(self.match_T(E_N_left, E_ge, Ts))

			N_subcells_right.append(N_right)
			N_subcells_left.append(N_left)

		return N_subcells_right, N_subcells_left, Ts_right, Ts_left

	def find_subcell(self, i):
		for j in range(1, self.N_subcells - 1):
			if self.r[i][0] >=  j * self.L_subcell and self.r[i][0] <= (j + 1) * self.L_subcell: #It will be in the j_th subcell
				return j	

	def find_subcell_flux(self, i, r):
		for j in range(0, self.N_subcells):
			if r[i][0] >=  j * self.L_subcell and r[i][0] <= (j + 1) * self.L_subcell: #It will be in the j_th subcell
				return j

	def scattering(self):
		scattering_events = 0

		for i in range(len(self.r)):
			if self.r[i][0] < self.L_subcell or self.r[i][0] > (self.N_subcells - 1) * self.L_subcell :
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
					pos = self.find_T(current_T, Ts)

					self.v[i] *= v_ge[pos]

					self.v_avg[i] = v_ge[pos]
					self.w_avg[i] = w_ge[pos]
					self.E[i] = E_ge[pos]
					self.C_V[i] = CV_ge[pos]
					self.MFP[i] = MFP_ge[pos]

					self.scattering_time[i] = 0. #Re-init scattering time

					scattering_events += self.W

				else:
					self.scattering_time[i] += self.dt #Account for the scattering time

		return scattering_events

	def energy_conservation(self, delta_E):
		for i in range(1, self.N_subcells - 1):

			if delta_E[i - 1] > E_max_ge: #Deletion of phonons
				E_sobrant = delta_E[i - 1]

				T = self.subcell_Ts[i]
				pos_T = self.find_T(T, Ts)

				E_phonon_T = E_ge[pos_T] #Energy per phonon for this subcell T

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

			elif -delta_E[i - 1] > E_max_ge: #Production of phonons
				E_sobrant = -delta_E[i - 1]

				T = self.subcell_Ts[i]
				pos_T = self.find_T(T, Ts)

				E_phonon_T = E_ge[pos_T] #Energy per phonon for this subcell T

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
		pos_T0 = self.find_T(self.T0, Ts)
		pos_Tf = self.find_T(self.Tf, Ts)

		N_0 = int(round(N_ge[pos_T0] / self.W, 0))
		N_f = int(round(N_ge[pos_Tf] / self.W, 0))

		total_indexs = []

		#Delete all the phonons in boundary subcells
		for i in range(len(self.r)):
			if self.r[i][0] <= self.L_subcell: #Subcell with T0 boundary
				total_indexs.append(i)

			elif self.r[i][0] >= (self.N_subcells - 1) * self.L_subcell: #Subcell with Tf boundary
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
			Temperatures.append(np.mean(self.subcell_Ts))
			

			if k == self.Nt - 1:
				N_subcells_right, N_subcells_left, Ts_right, Ts_left = self.calculate_subcell_T_2(0, self.N_subcells)

				plt.plot(np.linspace(0, len(self.E), len(self.E)), self.E, ls='', marker='.')
				plt.show()

				
				x = np.linspace(0, len(delta_energy), len(delta_energy))
				avg = np.mean(delta_energy)

				E_min = E_ge[self.find_T(self.Tf, Ts)] * 1e22

				plt.subplot(1, 2, 1)
				plt.plot(x, delta_energy, ls='-', marker='.', label=r'$E_N=%.2f·10^{-22}$ for $T_f=%.2f$' % (E_min, self.Tf))
				plt.plot(x, np.linspace(avg, avg, len(delta_energy)), ls='--', color='k', label='Avg')
				plt.title(r'$\Delta E$ scattering process')
				plt.legend()

				plt.subplot(1, 2, 2)
				plt.plot(np.linspace(0, self.Nt, self.Nt), scattering_events)
				plt.title('Scattering events in time')
				plt.show()
	
		return self.subcell_Ts, Energy, Phonons, Temperatures, N_subcells_right, N_subcells_left, Ts_right, Ts_left

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


#######################################
#									  #
#			   FUNCTIONS			  #
#									  #
#######################################

def plot_subplots():
	plot = Plots()
	plot.initial_subplots(Ts[0], Ts[-1], len(Ts), N_si, E_si, w_si, v_si, CV_si, MFP_si, 'Silicon')
	plot.initial_subplots(Ts[0], Ts[-1], len(Ts), N_ge, E_ge, w_ge, v_ge, CV_ge, MFP_ge, 'Germanium')

def plot_animation():

	gas = PhononGas(L_subcell, Lx, Ly, Lz, T0, Tf, Ti, t_MAX, dt, W)

	gas.animation()

def plot_animation_2():
	gas = PhononGas(L_subcell, Lx, Ly, Lz, T0, Tf, Ti, t_MAX, dt, W)

	gas.animation_2()	

def plots_single_sim():
	gas = pg(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W)

	gas.simulation()

	os.chdir('/home/alexgimenez/Àlex/Estudis/Python/Termodynamics and statistical physics/Gray Model/Data/Final_arrays')

	T_cells = np.load('Subcell_Ts.npy')
	Energy = np.load('Energy.npy')
	Phonons = np.load('Phonons.npy')
	Temperatures = np.load('Temperatures.npy')
	Scattering_events = np.load('Scattering_events.npy')

	N_subcells = int(Lx / Lx_subcell)

	#Temperature plot
	theo_T = ((T0**4 + Tf**4)/2)**(0.25)

	exp_avg_T = 0

	for i in range(1, N_subcells - 1):
		exp_avg_T += T_cells[i]

	exp_avg_T = exp_avg_T / (N_subcells - 2)

	x = np.linspace(0, Lx, N_subcells)

	plt.plot(x, T_cells, '-', marker='.', label='MC simulation T')
	plt.plot(x, np.linspace(theo_T, theo_T, N_subcells), '--', color = 'orange', label='Theoretical steady state T=%.2f' % theo_T)
	plt.plot(x, np.linspace(exp_avg_T, exp_avg_T , N_subcells), '--', color = 'r', label='Experimental steady state T=%.2f' % exp_avg_T)
	plt.plot(x, diffussive_T(x, T0, Tf, Lx), ls = '--', color = 'k', label='Diffussive')
	plt.title('T after %.2f ps' % (t_MAX * 1e12))
	plt.xlabel('Length (m)')
	plt.ylabel('Temperature (K)')
	plt.legend()
	plt.show()

	#Energy plot
	plt.subplot(2,2, 1)
	plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Energy, '-')
	plt.title('Energy of the system')
	plt.ylabel('E (J)')
	plt.xlabel('Time steps')

	#Number of phonons plot
	plt.subplot(2, 2, 2)
	plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Phonons, '-')
	plt.title('Phonons of the system')
	plt.ylabel('# of phonons')
	plt.xlabel('Time steps')

	#Temperature in time plot
	plt.subplot(2, 2, 3)
	plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Temperatures, '-')
	plt.title('Temperature of the system')
	plt.ylabel('T')
	plt.xlabel('Time steps')
	plt.show()

def single_simulation():
	N_subcells = int(Lx / Lx_subcell)

	t0 = current_milli_time()

	gas = PhononGas(Lx_subcell, Lx, Ly, Lz, T0, Tf, Ti, t_MAX, dt, W)

	T_cells, Energy, Phonons, Temperatures,  N_subcells_right, N_subcells_left, Ts_right, Ts_left = gas.simulation()

	tf = (current_milli_time() - t0) / 1000

	#Temperature plot
	theo_T = ((T0**4 + Tf**4)/2)**(0.25)

	exp_avg_T = 0

	for i in range(1, N_subcells - 1):
		exp_avg_T += T_cells[i]

	exp_avg_T = exp_avg_T / (N_subcells - 2)
	
	plt.subplot(1, 2, 1)
	plt.plot(np.linspace(0, N_subcells, N_subcells), Ts_right, '-', color='r', label='T right')
	plt.plot(np.linspace(0, N_subcells, N_subcells), Ts_left, '-', color='b', label='T left')
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(np.linspace(0, N_subcells, N_subcells), N_subcells_right, '--', color='r', label='N right')
	plt.plot(np.linspace(0, N_subcells, N_subcells), N_subcells_left, '--', color='b', label='N left')
	plt.plot(np.linspace(0, N_subcells, N_subcells), np.array(N_subcells_right) + np.array(N_subcells_left), '--', color='k', label='N tot')
	plt.title('tot')
	plt.legend()
	plt.show()

	x = np.linspace(0, Lx, N_subcells)

	plt.plot(x, T_cells, '-', marker='.', label='MC simulation T')
	plt.plot(x, np.linspace(theo_T, theo_T, N_subcells), '--', color = 'orange', label='Theoretical steady state T=%.2f' % theo_T)
	plt.plot(x, np.linspace(exp_avg_T, exp_avg_T , N_subcells), '--', color = 'r', label='Experimental steady state T=%.2f' % exp_avg_T)
	plt.plot(x, diffussive_T(x, T0, Tf, Lx), ls = '--', color = 'k', label='Diffussive')
	plt.title('T after %.2f ps' % (t_MAX * 1e12))
	plt.xlabel('Length (m)')
	plt.ylabel('Temperature (K)')
	plt.legend()
	plt.show()

	#Energy plot
	plt.subplot(2,2, 1)
	plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Energy, '-')
	plt.title('Energy of the system')
	plt.ylabel('E (J)')
	plt.xlabel('Time steps')

	#Number of phonons plot
	plt.subplot(2, 2, 2)
	plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Phonons, '-')
	plt.title('Phonons of the system')
	plt.ylabel('# of phonons')
	plt.xlabel('Time steps')

	#Temperature in time plot
	plt.subplot(2, 2, 3)
	plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Temperatures, '-')
	plt.title('Temperature of the system')
	plt.ylabel('T')
	plt.xlabel('Time steps')
	plt.show()

	print('Time: %.2f' % tf)

def different_t(t_MAXs):

	for t_MAX in t_MAXs:
		N_subcells = int(Lx / L_subcell)

		gas = PhononGas(L_subcell, Lx, Ly, Lz, T0, Tf, Ti, t_MAX, dt, W)

		T_cells, Energy, Phonons = gas.simulation()

		#Temperature plot
		

		exp_avg_T = 0

		for i in range(1, N_subcells - 1):
			exp_avg_T += T_cells[1]

		exp_avg_T = exp_avg_T / (N_subcells - 2)

		plt.plot(np.linspace(0, Lx, N_subcells), T_cells, '-o', label='MC simulation T for t=%.2f ps' % (t_MAX * 1e12))
		#plt.plot(np.linspace(0, Lx, N_subcells), np.linspace(exp_avg_T, exp_avg_T , N_subcells), '--', color = 'r', label='Experimental steady state T=%.2f' % exp_avg_T)
	
	theo_T = ((T0**4 + Tf**4)/2)**(0.25)
	plt.plot(np.linspace(0, Lx, N_subcells), np.linspace(theo_T, theo_T, N_subcells), '--', color = 'k', label='Theoretical steady state T=%.2f' % theo_T)

	plt.title('T vs length')
	plt.xlabel('Length (m)')
	plt.ylabel('Temperature (K)')
	plt.legend()
	plt.show()

def diffetent_W(Ws):
	CPU_time = []

	for W in Ws:

		N_subcells = int(Lx / L_subcell)

		t0 = current_milli_time()

		gas = PhononGas(L_subcell, Lx, Ly, Lz, T0, Tf, Ti, t_MAX, dt, W)

		T_cells, Energy, Phonons = gas.simulation()

		tf = (current_milli_time() - t0) / 1000

		#Temperature plot
		theo_T = ((T0**4 + Tf**4)/2)**(0.25)

		plt.subplot(2, 2, 1)
		plt.plot(np.linspace(0, Lx, N_subcells), T_cells, '-o', label='W=%.i' % W)

		if W == Ws[-1]: 
			plt.plot(np.linspace(0, Lx, N_subcells), np.linspace(theo_T, theo_T, N_subcells), '--', color = 'k', label='Theoretical steady state T=%.2f' % theo_T)
		
		plt.title('Temperature of subcells')
		plt.xlabel('Length (m)')
		plt.ylabel('Temperature (K)')
		plt.legend(loc='lower right')

		#Energy plot
		plt.subplot(2, 2, 2)
		plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Energy, '-', label='W=%.i' % W)
		plt.title('Energy of the system')
		plt.ylabel('E (J)')
		plt.xlabel('Time steps')
		plt.legend(loc='lower right')

		#Number of phonons plot
		plt.subplot(2, 2, 3)
		plt.plot(np.linspace(0, int(t_MAX/dt), int(t_MAX/dt)), Phonons, '-', label='W=%.i' % W)
		plt.title('Phonons of the system')
		plt.ylabel('# of phonons')
		plt.xlabel('Time steps')
		plt.legend(loc='lower right')

		CPU_time.append(tf)

	plt.subplot(2, 2, 4)
	bars = np.arange(len(Ws))
	plt.bar(bars, CPU_time)
	plt.title('CPU time vs W')
	plt.xlabel('W')
	plt.ylabel('CPU time (s)')
	plt.xticks(bars, Ws)

	plt.subplots_adjust(bottom=0.08, top=0.96, wspace=0.22, hspace=0.31 )

	plt.show()

def different_T(dif_T):
	pass

def flux():

	t0 = current_milli_time()

	gas = PhononGas(L_subcell, Lx, Ly, Lz, T0, Tf, Ti, t_MAX, dt, W)

	flux = gas.simulation_heat_flux()

	tf = (current_milli_time() - t0) / 1000

	print('Finished in %.2f' % tf)

	np.save('Flux.npy', flux)

	t = np.linspace(0, len(flux), len(flux))

	plt.plot(t, flux)
	plt.show()

def log_log():
	os.chdir('/home/alexgimenez/Àlex/Estudis/Python/Termodynamics and statistical physics/Gray Model/Data/Arrays')

	E = np.load('Etot_ge.npy')
	T = np.load('T_ge.npy')

	x = np.linspace(1, 501, 5000)

	E_log = np.log(E[90:190])

	x_log = (np.log(x[90:190]))

	print(T[90])

	slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, E_log)

	y_reg = slope * x_log + intercept

	plt.plot(x_log, E_log, ls='', marker='x', label='log log data')
	plt.plot(x_log, 4*x_log, label='4x')
	plt.plot(x_log, y_reg, label='Regre log log slope=%.2f' % slope)
	plt.legend()
	plt.show()

#PARAMETERS
Lx = 500e-9
Ly = 10e-9
Lz = 10e-9

Lx_subcell = 10e-9
Ly_subcell = 10e-9
Lz_subcell = 10e-9

V_subcell = Ly_subcell * Lz_subcell * Lx_subcell

N_ge *= V_subcell #Account for the different volumes
CV_ge *= V_subcell
Etot_ge *= V_subcell
tau_ge = MFP_ge / v_ge #tau = lambda / v_avg

T0 = 200
Tf = 100
Ti = 100

t_MAX = 1000e-12
dt = 10e-12

W = 2000


def find_T(value, T): #For a given value of temperature returns the position in the T array
	for i in range(len(T)):
		if T[i] >= value:
			return i

def match_T(value, E, T): #For a given T value returns the passed array value
	for i in range(len(E)):
		if E[i] == value:
			return T[i]

		elif E[i] > value: #If we exceed the value, use interpolation
			return T[i] * value /  E[i]

def prints():
	print('dt:', Lx_subcell / np.mean(v_ge)) #t = x/v
	print('dt_2', np.amin(tau_ge)) 

	print('\nMFP:', np.mean(MFP_ge[find_T(Tf, Ts) : find_T(T0, Ts)]*1e9), '\tDomain lenght:', Lx*1e9)
	print('tau T0:', tau_ge[find_T(T0, Ts)], '\ttau Tf:', tau_ge[find_T(Tf, Ts)])

	#######################
	print('--------------------------------')
	print('MFP T=10:', MFP_ge[find_T(10, Ts)]*1e9, 'MFP T=20:', MFP_ge[find_T(20, Ts)]*1e9)
	print('tau T=10:', tau_ge[find_T(10, Ts)]*1e9, 'tau T=20:', tau_ge[find_T(20, Ts)]*1e9)
	print('N T=10:', N_ge[find_T(10, Ts)], 'N T=20:', N_ge[find_T(20,Ts)], 'N T=17:', N_ge[find_T(17, Ts)])

	####Manually compute expected steady state T

	N_10 = N_ge[find_T(10, Ts)] #Number of phonons initialized at T=10K
	N_20 = N_ge[find_T(20, Ts)] #Number oh phonons initialized at T=20K

	E_10 = E_ge[find_T(10, Ts)]
	E_20 = E_ge[find_T(20, Ts)]

	'''
	Because of randomly initialization and due the not interaction of the phonons
	the number of phonons of each type in the middle subcells will be 1/2 of the initialized
	for each temperature
	'''

	theo = balistic_T(10, 20)

	avg_E = (N_10 * E_10 + N_20 * E_20) / (N_10 + N_20) #We can simplify the 1/2 factor
	avg_totE  = N_10 * E_10 + N_20 * E_20

	avg_T = match_T(avg_E, E_ge, Ts)

	T2 = match_T(avg_totE, Etot_ge, Ts)

	print('Steady state T from E/N:', avg_T) #El resultat es el mateix que en la simu!
	print('Steady state T from N:', match_T(((N_20 + N_10) / 2), N_ge, Ts))
	print('Steady state T from E:', T2)
	print('Steady state T theo:', theo)

	#When we do the simulation we see that the number of phonons in each subcell is the correct number
	#Something may be wrong with the initial integrals... replacing the integral of w by the same integral in k
	#we get the same result...

#############################################################################################################
#############################################################################################################

#plot_subplots()
prints()

log_log()

print(balistic_T_2(10, 20, 4.56))

#single_simulation()
#plots_single_sim()
#plot_animation()
#plot_animation_2()

t_MAXs = [1e-12, 10e-12, 100e-12, 1000e-12, 2000e-12]
#different_t(t_MAXs)

Ws = [1, 10, 50, 100]
#diffetent_W(Ws)

