import numpy as np
import os
from numpy import linalg as LA
import random
import math

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


############################################
#										   #
#				  Classes			       #
#										   #
############################################

class GrayModel(object):
	def __init__(self, Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux, init_restart, folder):

		if init_restart :

			self.read_restart(current_dir + '/' + folder)

		else:

			self.Lx = float(Lx) #x length of the box
			self.Ly = float(Ly) #y length of the box
			self.Lz = float(Lz) #z lenght of the box
			self.T0 = float(T0) #Temperature of the initial sub-cell (Boundary)
			self.Tf = float(Tf) #Temperature of the last sub-cell (Boundary)
			self.Ti = float(Ti) #Initial temperature of studied subcells
			self.t_MAX = float(t_MAX) #Maximum simulation time
			self.dt = float(dt) #Step size
			self.W = float(W) #Weighting factor

			self.Nt = int(round(self.t_MAX / self.dt, 0)) #Number of simulation steps/iterations

			self.Lx_subcell = float(Lx_subcell) #x length of each subcell
			self.Ly_subcell = float(Ly_subcell) #x length of each subcell
			self.Lz_subcell = float(Lz_subcell) #x length of each subcell

			self.N_subcells_x = int(round(self.Lx / self.Lx_subcell, 0)) #Number of subcells
			self.N_subcells_y = int(round(self.Ly / self.Ly_subcell, 0))
			self.N_subcells_z = int(round(self.Lz / self.Lz_subcell, 0)) 

			self.every_flux = every_flux

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

		self.V_subcell = self.Ly_subcell * self.Lz_subcell * self.Lx_subcell

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

	def diffusive_boundary(self, i): #Not tried yet

		y_wall = False
		z_wall = False

		if self.r[i][0] >= self.Lx or self.r[i][0] < 0:
			self.v[i][0] *= -1.

			if self.r[i][0] >= self.Lx:
				self.r[i][0] = self.Lx - 0.01*self.Lx
			else:
				self.r[i][0] = 0

		if self.r[i][1] >= self.Ly or self.r[i][1] < 0:
			y_wall = True

			if self.r[i][1] > self.Ly or self.r[i][1] < 0:
				self.v[i][1] *= -1.

			if self.r[i][1] > self.Ly:
				delta_y = self.r[i][1] - self.Ly
				self.r[i][1] = self.r[i][1] - 2*delta_y
			else:
				delta_y = -self.r[i][1] 
				self.r[i][1] = delta_y

		if self.r[i][2] >= self.Lz or self.r[i][2] < 0:
			z_wall = True

			if self.r[i][2] > self.Lz:
				delta_z = self.r[i][2] - self.Lz
				self.r[i][2] = self.r[i][2] - 2*delta_z
			else:
				delta_z = -self.r[i][2] 
				self.r[i][2] = delta_z

		if y_wall or z_wall :
			if y_wall and not z_wall:
				x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
				y = int(self.N_subcells_y - 1)
				z = int(self.r[i][2] / self.Lz * self.N_subcells_z)
			if not y_wall and z_wall:
				x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
				y = int(self.r[i][1] / self.Lz * self.N_subcells_z)
				z = int(self.Lz - 1)

			if y_wall and z_wall:
				x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
				y = int(self.Ly - 1)
				z = int(self.Lz - 1)

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

	def check_boundaries(self, i):

		if self.r[i][0] >= self.Lx or self.r[i][0] < 0:
			self.v[i][0] *= -1.

			if self.r[i][0] > self.Lx:
				self.r[i][0] = self.Lx
			else:
				self.r[i][0] = 0

		if self.r[i][1] > self.Ly or self.r[i][1] < 0:
			self.v[i][1] *= -1.

			if self.r[i][1] > self.Ly:
				delta_y = self.r[i][1] - self.Ly
				self.r[i][1] = self.r[i][1] - 2*delta_y
			else:
				delta_y = -self.r[i][1] 
				self.r[i][1] = delta_y

		if self.r[i][2] > self.Lz or self.r[i][2] < 0:
			self.v[i][2] *= -1.

			if self.r[i][2] > self.Lz:
				delta_z = self.r[i][2] - self.Lz
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

					E_N = E_subcells[i][j][k] / N_subcells[i][j][k]

					self.subcell_Ts[i][j][k] = self.match_T(E_N, self.E_ge, self.Ts)

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

			if x < 1 or x > (self.N_subcells_x - 1):
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
						E_sobrant = delta_E[i][j][k]

						T = self.subcell_Ts[i][j][k]
						pos_T = self.find_T(T, self.Ts)

						E_phonon_T = self.E_ge[pos_T] #Energy per phonon for this subcell T

						N_phonons = int(round(E_sobrant / (E_phonon_T * self.W), 0)) #Number of phonons to delete

						if N_phonons != 0:

							array_position_phonons_ith_subcell = []
							counter = 0

							for l in range(len(self.r)):

								x = int(self.r[i][0] / self.Lx * self.N_subcells_x)
								y = int(self.r[i][1] / self.Ly * self.N_subcells_y)
								z = int(self.r[i][2] / self.Lz * self.N_subcells_z)

								if counter == N_phonons: 
										break

								if x == i and y == j and z == k : #is in the i_th subcell

									counter += 1
									array_position_phonons_ith_subcell.append(l) #position in self.r array

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

	def calculate_flux(self, i, r_previous):
		'''
			Calculates the flux in the yz plane in the middle of Lx lenght
		'''

		if self.r[i][0] > self.Lx/2 and r_previous[i][0] < self.Lx/2:

			return self.E[i] * self.W

		elif self.r[i][0] < self.Lx/2 and r_previous[i][0] > self.Lx/2:
			return -self.E[i] * self.W

		else:
			return 0

	def save_restart(self, nt):

		os.chdir(current_dir)

		if not os.path.exists('restart_%i' % nt): os.mkdir('restart_%i' % nt)
		os.chdir('restart_%i' % nt)

		np.save('r.npy', self.r)
		np.save('v.npy', self.v)

		np.save('E.npy', self.E)
		np.save('N.npy', self.N)
		np.save('w_avg.npy', self.w_avg)
		np.save('v_avg.npy', self.v_avg)
		np.save('C_V.npy', self.C_V)
		np.save('MFP.npy', self.MFP)

		np.save('scattering_time.npy', self.scattering_time)

		np.save('subcell_Ts.npy', self.subcell_Ts)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

	def get_parameters(self):
		f = open('parameters_used.txt', 'r')

		i = 0

		for line in f:

			try:

				cols = line.split()

				if len(cols) > 0:
					value = float(cols[1])

				if i == 0:
					self.Lx = value

				elif i == 1:
					self.Ly = value

				elif i == 2:
					self.Lz = value

				elif i == 4:
					self.Lx_subcell = value

				elif i == 5:
					self.Ly_subcell = value

				elif i == 6:
					self.Lz_subcell = value

				elif i == 8:
					self.T0 = value

				elif i == 9:
					self.Tf = value

				elif i == 10:
					self.Ti = value

				elif i == 12:
					self.t_MAX = value

				elif i == 13:
					self.dt = value

				elif i == 15:
					self.W = value

				elif i == 16:
					self.every_flux = value

				i += 1

			except:
				pass

	def read_restart(self, folder):

		os.chdir(folder)

		self.r = np.load('r.npy')
		self.v = np.load('v.npy')

		self.E = np.load('E.npy')
		self.N = np.load('N.npy')
		self.w_avg = np.load('w_avg.npy')
		self.v_avg = np.load('v_avg.npy')
		self.C_V = np.load('C_V.npy')
		self.MFP = np.load('MFP.npy')

		self.scattering_time = np.load('scattering_time.npy')

		self.subcell_Ts = np.load('subcell_Ts.npy')

		self.get_parameters()

		self.Nt = int(round(self.t_MAX / self.dt, 0)) #Number of simulation steps/iterations

		self.N_subcells_x = int(round(self.Lx / self.Lx_subcell, 0)) #Number of subcells
		self.N_subcells_y = int(round(self.Ly / self.Ly_subcell, 0))
		self.N_subcells_z = int(round(self.Lz / self.Lz_subcell, 0)) 

		os.chdir(current_dir)

	def simulation(self, every_restart, folder):
		os.chdir(current_dir)

		self.init_particles()

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []
		cell_temperatures = []
		elapsed_time = []

		flux = []

		t0 = current_time()

		for k in range(self.Nt):
			print(k)

			if k % every_restart == 0:

				#Save configuration actual properties
				self.save_restart(k)

				#Save outputs untill this moment (Inside the restart folder)
				flux_save = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

				np.save('Energy.npy', Energy)
				np.save('Phonons.npy', Phonons)
				np.save('Subcell_Ts.npy', cell_temperatures)
				np.save('Temperatures.npy', Temperatures)
				np.save('Scattering_events.npy', scattering_events)
				np.save('Elapsed_time.npy', elapsed_time)
				np.save('Flux.npy', flux_save)

				os.chdir(current_dir) #Go back to the principal directory

			if k % int(self.every_flux) == 0:

				previous_r = np.copy(self.r) #Save the previous positions to calculate the flux

				self.r += self.dt * self.v #Drift

				flux_k = 0

				for i in range(len(self.r)):
					self.check_boundaries(i)
					#self.diffusive_boundary(i)

					flux_k += self.calculate_flux(i, previous_r)

				flux.append(flux_k)

			else:
				self.r += self.dt * self.v #Drift

				for i in range(len(self.r)):
					self.check_boundaries(i)
					#self.diffusive_boundary(i)

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

			copy_subcells = np.copy(self.subcell_Ts)

			cell_temperatures.append(copy_subcells)

			elapsed_time.append(current_time() - t0)

		if not  os.path.exists(current_dir + '/' + folder): os.mkdir(current_dir + '/' + folder)
		os.chdir(current_dir + '/' + folder)

		flux = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', cell_temperatures)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)
		np.save('Elapsed_time.npy', elapsed_time)
		np.save('Flux.npy', flux)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

	def simulate_from_restart(self, every_restart, folder):

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []
		cell_temperatures = []
		elapsed_time = []

		flux = []

		t0 = current_time()

		for k in range(self.Nt):
			print(k)

			if k % every_restart == 0:

				self.save_restart(k)

				#Save outputs untill this moment (Inside the restart folder)
				flux_save = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

				np.save('Energy.npy', Energy)
				np.save('Phonons.npy', Phonons)
				np.save('Subcell_Ts.npy', cell_temperatures)
				np.save('Temperatures.npy', Temperatures)
				np.save('Scattering_events.npy', scattering_events)
				np.save('Elapsed_time.npy', elapsed_time)
				np.save('Flux.npy', flux_save)

				os.chdir(current_dir) #Go back to the principal directory

			if k % int(self.every_flux) == 0:

				previous_r = np.copy(self.r) #Save the previous positions to calculate the flux

				self.r += self.dt * self.v #Drift

				flux_k = 0

				for i in range(len(self.r)):
					self.check_boundaries(i)
					#self.diffusive_boundary(i)

					flux_k += self.calculate_flux(i, previous_r)

				flux.append(flux_k)

			else:
				self.r += self.dt * self.v #Drift

				for i in range(len(self.r)):
					self.check_boundaries(i)
					#self.diffusive_boundary(i)

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

			copy_subcells = np.copy(self.subcell_Ts)

			cell_temperatures.append(copy_subcells)

			elapsed_time.append(current_time() - t0)

		if not  os.path.exists(current_dir + '/' + folder): os.mkdir(current_dir + '/' + folder)
		os.chdir(current_dir + '/' + folder)

		flux = np.array(flux) / (self.Ly * self.Lz * self.dt * self.every_flux)

		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Subcell_Ts.npy', cell_temperatures)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)
		np.save('Elapsed_time.npy', elapsed_time)
		np.save('Flux.npy', flux)

		f = open('parameters_used.txt', 'w')

		f.write('Lx: ' + str(self.Lx) + '\n')
		f.write('Ly: ' + str(self.Ly) + '\n')
		f.write('Lz: ' + str(self.Lz) + '\n\n')

		f.write('Lx_subcell: ' + str(self.Lx_subcell) + '\n')
		f.write('Ly_subcell: ' + str(self.Ly_subcell) + '\n')
		f.write('Lz_subcell: ' + str(self.Lz_subcell) + '\n\n')

		f.write('T0: ' + str(self.T0) + '\n')
		f.write('Tf: ' + str(self.Tf) + '\n')
		f.write('Ti: ' + str(self.Ti) + '\n\n')

		f.write('t_MAX: ' + str(self.t_MAX) + '\n')
		f.write('dt: ' + str(self.dt) + '\n\n')

		f.write('W: ' + str(self.W) + '\n')
		f.write('Every_flux: ' + str(self.every_flux))

		f.close()

	def animation(self):
		self.init_particles()

		def update(t, x, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i)
				#self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T()

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T()

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T()

			lines[0].set_data(x, self.subcell_Ts[:, int(Ly / 2), int(Lz/2)])
			lines[1].set_data(x, diffussive_T(x, self.T0, self.Tf, self.Lx))
			lines[2].set_data(x, np.linspace(balistic_T(T0, Tf), balistic_T(T0, Tf), len(x)))
			lines[3].set_text('Time step %i of %i' % (k, self.Nt))

			return lines

		# Attaching 3D axis to the figure
		fig, ax = plt.subplots()

		x = np.linspace(0, self.Lx, int(round(self.Lx/self.Lx_subcell, 0)))

		lines = [ax.plot(x, self.subcell_Ts[:, int(Ly / 2), int(Lz/2)], '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, self.T0, self.Tf, self.Lx), label='Diffusive')[0],
		ax.plot(x, np.linspace(balistic_T(T0, Tf), balistic_T(T0, Tf), len(x)), ls='--', color='k', label='Ballistic')[0], ax.text(0, self.Tf, '', color='k', fontsize=10)]

		ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=1, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper right')
		plt.show()

		return self.r, self.subcell_Ts

	def animation_from_restart(self, folder):
		
		self.read_restart(current_dir + '/' + folder)

		def update(t, x, lines):
			k = int(t / self.dt)

			self.r += self.dt * self.v #Drift

			for i in range(len(self.r)):
				self.check_boundaries(i)
				#self.diffusive_boundary(i)

			self.re_init_boundary()

			E_subcells, N_subcells = self.calculate_subcell_T()

			self.scattering()

			E_subcells_new , N_subcells_new = self.calculate_subcell_T()

			delta_E = np.array(E_subcells_new) - np.array(E_subcells)

			self.energy_conservation(delta_E)

			self.calculate_subcell_T()

			lines[0].set_data(x, self.subcell_Ts[:, int(Ly / 2), int(Lz/2)])
			lines[1].set_data(x, diffussive_T(x, self.T0, self.Tf, self.Lx))
			lines[2].set_data(x, np.linspace(balistic_T(T0, Tf), balistic_T(T0, Tf), len(x)))
			lines[3].set_text('Time step %i of %i' % (k, self.Nt))

			return lines

		# Attaching 3D axis to the figure
		fig, ax = plt.subplots()

		x = np.linspace(0, self.Lx, int(round(self.Lx/self.Lx_subcell, 0)))

		lines = [ax.plot(x, self.subcell_Ts[:, int(Ly / 2), int(Lz/2)], '-o', color='r', label='Temperature')[0], ax.plot(x, diffussive_T(x, self.T0, self.Tf, self.Lx), label='Diffusive')[0],
		ax.plot(x, np.linspace(balistic_T(T0, Tf), balistic_T(T0, Tf), len(x)), ls='--', color='k', label='Ballistic')[0], ax.text(0, self.Tf, '', color='k', fontsize=10)]

		ani = FuncAnimation(fig, update, fargs=(x, lines), frames=np.linspace(0, self.t_MAX-self.dt, self.Nt),
		                    blit=True, interval=1, repeat=False)
		#ani.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
		plt.legend(loc = 'upper right')
		plt.show()

		return self.r, self.subcell_Ts

if __name__ == '__main__':

	def find_T(value, T): 
		for i in range(len(T)):
			if T[i] >= value:
				return i

	def match_T(value, E, T):
		for i in range(len(E)):
			if E[i] == value:
				return T[i]

			elif E[i] > value: #If we exceed the value, use interpolation
				return T[i] * value /  E[i]

	os.chdir(array_folder)

	#PARAMETERS
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

	folder_outputs = 'PROVA'

	MFP = np.load('MFP_ge.npy')
	N = np.load('N_ge.npy') *Lx*Ly*Lz
	v_avg = np.load('v_ge.npy')
	Ts = np.load('T_ge.npy')

	print('Max_tau:', np.max(MFP[find_T(Tf, Ts): find_T(T0, Ts)] / v_avg[find_T(Tf, Ts): find_T(T0, Ts)]))
	print('MFP max:', MFP[find_T(Tf, Ts)], ' MFP_avg: ', np.mean(MFP[find_T(Tf, Ts): find_T(T0, Ts)]))
	print('v_avg max:', v_avg[find_T(Tf, Ts)])
	print('N: ', np.mean(N[find_T(Tf, Ts): find_T(T0, Ts)]))

	gas = GrayModel(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W, every_flux, init_restart, folder_restart)

	#gas.simulation(every_restart, folder_outputs)
	gas.animation()

	'''
	E = np.load('Energy.npy')
	N = np.load('Phonons.npy')
	T_cells = np.load('Subcell_Ts.npy')
	scattering_events = np.load('Scattering_events.npy')
	temperatures = np.load('Temperatures.npy')

	print(len(T_cells))

	#Subplots
	plt.subplot(2, 2, 1)
	plt.plot(np.linspace(0, len(E), len(E)), E)
	plt.title('E')

	plt.subplot(2, 2, 2)
	plt.plot(np.linspace(0, len(N), len(N)), N)
	plt.title('N')

	plt.subplot(2, 2, 3)
	plt.plot(np.linspace(0, len(temperatures), len(temperatures)), temperatures)
	plt.title('T')
	
	plt.subplot(2, 2, 4)
	plt.plot(np.linspace(0, len(scattering_events), len(scattering_events)), scattering_events)
	plt.title('Scattering events')

	plt.show()

	#T plot
	x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

	plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], ls='-', color='b', marker='^')
	plt.plot(x, T_cells[500][:, int(Ly / 2), int(Lz/2)], ls='-', color='r', marker='^')

	plt.plot(x, np.linspace(balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), balistic_T(float(T_cells[-1][0]), float(T_cells[-1][-1])), len(x)), ls='--', color='k')
	plt.plot(x, diffussive_T(x, float(T_cells[-1][0]), float(T_cells[-1][-1]), Lx), color='k')

	plt.show()

	#plt.contourf(T_cells[:, :, 0], cmap='hot')
	#plt.colorbar()
	#plt.show()
	'''
	
