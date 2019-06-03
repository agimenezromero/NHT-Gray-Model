import numpy as np
import math
import os
from scipy import integrate, stats
import matplotlib.pyplot as plt

##########################################################
#														 #
#					Silicium parameters					 #
#														 #	
##########################################################

#Constants
v_si_LA = 9.01e3
v_si_TA = 5.23e3 
v_si_LO = 0
v_si_TO = -2.57e3

c_si_LA = -2e-7
c_si_TA = -2.26e-7
c_si_LO = -1.6e-7
c_si_TO = 1.12e-7

omega_0_si_LA = 0
omega_0_si_TA = 0
omega_0_si_LO = 9.88e13
omega_0_si_TO = 10.2e13

k_max_si = 1.157e10 / 2

#Maximum frequencies for silicon
w_max_si_LA = 7.747e13
w_max_si_TA = 3.026e13
w_max_si_LO = omega_0_si_LO
w_max_si_TO = omega_0_si_TO

#Minimum frequencies for silicon
w_min_si_LA = 0
w_min_si_TA = 0
w_min_si_LO = 7.738e13
w_min_si_TO = 8.726e13

k_bulk_si = 139

vs_si = [v_si_TA, v_si_LA, v_si_LO, v_si_TO]
cs_si = [c_si_TA, c_si_LA, c_si_LO, c_si_TO]
maximum_freqs_si = [w_max_si_TA, w_max_si_LA, w_max_si_LO, w_max_si_TO]
minimum_freqs_si = [w_min_si_TA, w_min_si_LA, w_min_si_LO, w_min_si_TO]
omegas_0_si = [omega_0_si_TA, omega_0_si_LA, omega_0_si_LO, omega_0_si_TO]

##########################################################
#														 #
#					Germanium parameters				 #
#														 #	
##########################################################

#Constants
v_ge_LA = 5.3e3
v_ge_TA = 2.26e3 
v_ge_LO = -0.99e3
v_ge_TO = -0.18e3

c_ge_LA = -1.2e-7
c_ge_TA = -0.82e-7
c_ge_LO = -0.48e-7
c_ge_TO = 0

omega_0_ge_LA = 0
omega_0_ge_TA = 0
omega_0_ge_LO = 5.7e13
omega_0_ge_TO = 5.5e13

k_max_ge = 1.1105e10 / 2

#Maximum frequencies for germanium
w_max_ge_LA = 4.406e13
w_max_ge_TA = 1.498e13
w_max_ge_LO = omega_0_ge_LO
w_max_ge_TO = omega_0_ge_TO

#Minimum frequencies for germanium
w_min_ge_LA = 0
w_min_ge_TA = 0
w_min_ge_LO = 4.009e13
w_min_ge_TO = 5.3e13

k_bulk_ge = 58

vs_ge = [v_ge_TA, v_ge_LA, v_ge_LO, v_ge_TO]
cs_ge = [c_ge_TA, c_ge_LA, c_ge_LO, c_ge_TO]
maximum_freqs_ge = [w_max_ge_TA, w_max_ge_LA, w_max_ge_LO, w_max_ge_TO]
minimum_freqs_ge = [w_min_ge_TA, w_min_ge_LA, w_min_ge_LO, w_min_ge_TO]
omegas_0_ge = [omega_0_ge_TA, omega_0_ge_LA, omega_0_ge_LO, omega_0_ge_TO]

##########################################################
#														 #
#					General parameters				 	 #
#														 #	
##########################################################

hbar = 1.05457e-34
k_B = 1.38064e-23

k_max_array = [k_max_si, k_max_ge]

class ThermalProperties(object):
	def __init__(self, T_0, T_max, n, w_max_array, v_array, c_array, omega_0_array, k_bulk, name):
		self.Ns = []
		self.Es = []
		self.ws = []
		self.vs = []
		self.CVs = []
		self.MFPs = []
		self.E_tot = []

		self.T_0 = T_0
		self.T_max = T_max
		self.n = n

		self.Ts = np.linspace(self.T_0, self.T_max, self.n)

		self.w_max_array = w_max_array
		self.v_array = v_array
		self.c_array = c_array
		self.omega_0_array = omega_0_array
		self.k_bulk = k_bulk

		self.name = name

	def N(self, T):

		def f_N(w, T, v, c, omega_0):

			num = (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2 
			denom = 4*c**2 * (np.exp(hbar*w / (k_B*T)) - 1)*(np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))

			return num / denom

		N = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			N += integrate.quad(f_N, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				N *= 2

		return N  / (2*np.pi**2) 

	def E(self, T):

		def f_E(w, T, v, c, omega_0):

			num = hbar * w * (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2
			denom = 4*c**2 * (np.exp(hbar*w / (k_B*T)) - 1)*(np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))

			return num / denom

		E = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			E += integrate.quad(f_E, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				E *= 2

		return E / (2*np.pi**2)

	def v_avg(self, T):

		def f_v(w, T, v, c, omega_0):

			num = (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2 
			denom = 4*c**2 * (np.exp(hbar*w / (k_B*T)) - 1)

			return num / denom

		x = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			x += integrate.quad(f_v, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				x *= 2

		return x / (2*np.pi**2) 

	def C_V(self, T):

		def f_C_V(w, T, v, c, omega_0):

			num = (-v + np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))**2 * (hbar * w)**2 * np.exp(hbar * w / (k_B * T))
			denom = 4*c**2 * k_B * T**2 * (np.exp(hbar*w / (k_B*T)) - 1)**2 * (np.sqrt(abs(v**2 + 4 * c * (w-omega_0))))

			return num / denom

		CV = 0
		for i in range(2): #Sum for all considered polarizations
			w_max = self.w_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			CV += integrate.quad(f_C_V, 1, w_max, args = (T, v_i, c_i, omega_0_i))[0]
			
			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				CV *= 2

		return CV / (2*np.pi**2)

	def fill_arrays(self):
		for T in self.Ts:
			N_T = self.N(T)
			E_T = self.E(T)
			v = self.v_avg(T)
			CV_T = self.C_V(T)

			self.Ns.append(N_T) #N per unit volume
			self.E_tot.append(E_T) #E per unit volume
			self.Es.append(E_T / N_T) #E per unit volume per phonon
			self.ws.append(E_T / (hbar * N_T)) #w_avg
			self.vs.append(v / N_T) #v_avg
			self.CVs.append(CV_T) #Cv per unit volume
			self.MFPs.append(3 * N_T * self.k_bulk / (v * CV_T)) #MFP

		return self.Ns, self.Es, self.ws, self.vs, self.CVs, self.MFPs, self.E_tot, self.Ts

	def plot_properties(self):
		#N(T)
		plt.subplot(3, 2, 1)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.Ns)
		#plt.title('Nº phonons vs temperature')
		plt.ylabel('Nº phonons')
		plt.xlabel('T (K)')
		
		#E(T)
		plt.subplot(3, 2, 2)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.Es)
		#plt.title('Energy vs temperature')
		plt.ylabel('E (J) per phonon')
		plt.xlabel('T (K)')

		#w_avg
		plt.subplot(3, 2, 3)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.ws)
		#plt.title('Average frequency vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$\omega_{avg} \, (rad/s)$')

		#v_avg
		plt.subplot(3, 2, 4)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.vs)
		#plt.title('Average group velocity vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$v_{avg} \, (m/s)$')

		#C_V
		plt.subplot(3, 2, 5)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.CVs)
		#plt.title('Heat capacity vs temperature')
		plt.ylabel(r'$C_V$ (J/K)')
		plt.xlabel('T (K)')

		#MFP
		plt.subplot(3, 2, 6)
		plt.plot(np.linspace(self.T_0, self.T_max, self.T_max - self.T_0), self.MFPs)
		#plt.title('Mean Free Path vs Temperature')
		plt.xlabel('T(K)')
		plt.ylabel(r'$\Lambda \, (m)$')

		plt.suptitle('Thermal transport properties for %s' % self.name)
		plt.show()


if __name__ == '__main__':

	os.chdir('/home/alexgimenez/Àlex/Estudis/Python/Termodynamics and statistical physics/Gray Model/Data/Arrays')

	print(os.getcwd())

	E = np.load('E_ge.npy')
	MFP = np.load('MFP_ge.npy')
	v = np.load('v_ge.npy')
	T = np.load('T_ge.npy')
	N_si = np.load('N_si.npy')

	tau = MFP / v

	print(MFP[190]*1e9)
	print(tau[190]*1e9)
	print('N_si at T=%.2f k:' % T[2990], N_si[2990]*1e-29 , '·10^5')

	x = np.linspace(1, 501, 5000)

	E_log = np.log(E[90:190])

	x_log = (np.log(x[90:190]))

	slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, E_log)

	y_reg = slope * x_log + intercept

	'''
	plt.plot(x_log, E_log, ls='', marker='x', label='log log data')
	plt.plot(x_log, 4*x_log, label='4x')
	plt.plot(x_log, y_reg, label='Regre log log slope=%.2f' % slope)
	plt.legend()
	plt.show()
	'''

	'''
	def omega_pos(k):

		return np.sqrt((1 + np.sqrt(1 - 0.88 * np.sin(k/2)**2)))

	def omega_neg(k):
		return np.sqrt((1 - np.sqrt(1 - 0.88 * np.sin(k/2)**2)))

	k = np.linspace(-8*np.pi/2, 8*np.pi/2, 1000)

	plt.figure(figsize=(8, 6))

	plt.plot(k, omega_pos(k), label='Optical branch')
	plt.plot(k, omega_neg(k), label='Acoustical branch')

	plt.plot(np.linspace(-np.pi, -np.pi, 100), np.linspace(0, 0.81, 100), ls='--', color='k')
	plt.plot(np.linspace(np.pi, np.pi, 100), np.linspace(0, 0.81, 100), ls='--', color='k')

	plt.text(-2.3, 0.9, r'Brillouin Zone')
	plt.annotate(s='', xy=(-3,0.85), xytext=(3,0.85), arrowprops=dict(arrowstyle='<->'))

	locs, labels = plt.xticks()

	new_labels = [r'$-3\pi/a$', r'$-2\pi/a$', r'$-\pi/a$', '0', r'$\pi/a$', r'$2\pi/a$', r'$3\pi/a$']
	new_locs = np.linspace(-3*np.pi, 3*np.pi, 7)

	plt.legend(loc='upper left')
	plt.title(r'Dispersion relation $\omega(k)$')
	plt.ylabel(r'Frequency ($\omega$)')
	plt.xlabel(r'Wave vector ($k$)')

	plt.ylim(0, 1.8)

	plt.xticks(new_locs, new_labels)
	plt.show()
	'''

