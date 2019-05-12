import numpy as np
import math
import os
from scipy import integrate
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

k_max_si = 1.157e10

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

vs_si = [v_si_LA, v_si_TA, v_si_LO, v_si_TO]
cs_si = [c_si_LA, c_si_TA, c_si_LO, c_si_TO]
maximum_freqs_si = [w_max_si_LA, w_max_si_TA, w_max_si_LO, w_max_si_TO]
minimum_freqs_si = [w_min_si_LA, w_min_si_TA, w_min_si_LO, w_min_si_TO]
omegas_0_si = [omega_0_si_LA, omega_0_si_TA, omega_0_si_LO, omega_0_si_TO]

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

k_max_ge = 1.1105e10

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

vs_ge = [v_ge_LA, v_ge_TA, v_ge_LO, v_ge_TO]
cs_ge = [c_ge_LA, c_ge_TA, c_ge_LO, c_ge_TO]
maximum_freqs_ge = [w_max_ge_LA, w_max_ge_TA, w_max_ge_LO, w_max_ge_TO]
minimum_freqs_ge = [w_min_ge_LA, w_min_ge_TA, w_min_ge_LO, w_min_ge_TO]
omegas_0_ge = [omega_0_ge_LA, omega_0_ge_TA, omega_0_ge_LO, omega_0_ge_TO]

##########################################################
#														 #
#					General parameters				 	 #
#														 #	
##########################################################

hbar = 1.05457e-34
k_B = 1.38064e-23

V = 1 #1e-24

k_max_array = [k_max_si, k_max_ge]

#FUNCTIONS

def k(v, c, w, omega_0):
	if c !=0:
		b = v/c
		inverse_c = 1/c

		return (-b + np.sqrt(abs(b**2 + 4*inverse_c*(w - omega_0))))/2
	else:
		return (w-omega_0) / v

def vg(v, c, w, omega_0):
	return v + 2 * c * k(v, c, w, omega_0)

def w(omega_0, v, c, k):
	return omega_0 + v * k + c * k**2


#PLOTS
def w_vs_K(k_max, v_array, c_array, omega_0_array, name_material):

	for i in range(4):
		k = np.linspace(0, k_max, 1000)

		v_i = v_array[i]
		c_i = c_array[i]
		omega_0_i = omega_0_array[i]

		if i == 0: name = 'LA'
		elif i == 1: name = 'TA'
		elif i == 2: name = 'LO'
		else: name = 'TO'

		plt.plot(k, w(omega_0_i, v_i, c_i, k), label = name)

	plt.title('Dispersion curve of %s in [100] direction' % name_material)
	plt.xlabel(r'$K(m^{-1})$')
	plt.ylabel(r'$\omega(rad/s)$')
	plt.grid(True)
	plt.legend(loc = 'center left')
	plt.show()

def vg_vs_w(v_array, c_array, omega_0_array, w_max_array, w_min_array, name_material):

	for i in range(4):
		v_i = v_array[i]
		c_i = c_array[i]
		omega_0_i = omega_0_array[i]

		w = np.linspace(w_min_array[i], w_max_array[i], 100000)

		if i == 0: 
			name = 'LA'
		elif i == 1: 
			name = 'TA'
		elif i == 2: 
			name = 'LO'
		else: 
			name = 'TO'

		#Preguntar xavi pq he de fer-ho aixi pq doni be
		if i == 3: #TO
			plt.plot(w, vg(v_i, c_i, w, omega_0_i), label = name)
		else:
			plt.plot(w, -vg(v_i, c_i, w, omega_0_i), label = name)


	plt.title('Frequency vs group velocity for %s' % name_material)
	plt.ylabel(r'$v_g(m/s)$')
	plt.xlabel(r'$\omega(rad/s)$')
	plt.grid(True)
	plt.legend(loc = 'upper right')
	plt.show()


#Classe
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

			N += integrate.quad(f_N, 0, w_max, args = (T, v_i, c_i, omega_0_i))[0]

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

			E += integrate.quad(f_E, 0, w_max, args = (T, v_i, c_i, omega_0_i))[0]

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

			x += integrate.quad(f_v, 0, w_max, args = (T, v_i, c_i, omega_0_i))[0]

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

			CV += integrate.quad(f_C_V, 0, w_max, args = (T, v_i, c_i, omega_0_i))[0]
			
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
			self.MFPs.append(3 * N_T * self.k_bulk / (v * CV_T)) #MFP per unit volume

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

class ThermalProperties_K(object):
	def __init__(self, T_0, T_max, n, V, w_max_array, v_array, c_array, omega_0_array, k_bulk, name):
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

		self.k_max_array = k_max_array
		self.v_array = v_array
		self.c_array = c_array
		self.omega_0_array = omega_0_array
		self.k_bulk = k_bulk

		self.V = V
		self.name = name

	def N(self, T):

		def f_N(k, T, v, c, omega_0):

			num = k**2
			denom = np.exp(hbar/(k_B * T) * (omega_0 + v*k + c*k**2)) - 1

			return num / denom

		N = 0
		for i in range(2): #Sum for all considered polarizations
			k_max = self.k_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			N += integrate.quad(f_N, 0, k_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				N *= 2

		return N * self.V / (2*np.pi**2) 

	def E(self, T):
		def f_E(k, T, v, c, omega_0):

			num = hbar * (omega_0 + v*k + c*k**2) * k**2
			denom = np.exp(hbar/(k_B * T) * (omega_0 + v*k + c*k**2)) - 1

			return num / denom

		E = 0
		for i in range(2): #Sum for all considered polarizations
			k_max = self.k_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			E += integrate.quad(f_E, 0, k_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				E *= 2

		return E * self.V / (2*np.pi**2) 

	def v_avg(self, T):

		def f_v(k, T, v, c, omega_0):

			num = (v + 2*c*k) * k**2  
			denom = np.exp(hbar / (k_B * T) * (omega_0 + v*k + c*k**2)) -1

			return num / denom

		x = 0
		for i in range(2): #Sum for all considered polarizations
			k_max = self.k_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			x += integrate.quad(f_v, 0, k_max, args = (T, v_i, c_i, omega_0_i))[0]

			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				x *= 2

		return x * self.V / (2*np.pi**2) 

	def C_V(self, T):

		def f_C_V(k, T, v, c, omega_0):

			num = hbar**2 * (omega_0 + v*k + c*k**2)**2 * np.exp(hbar / (k_B * T) * (omega_0 + v*k + c*k**2))
			denom = k_B * T**2 * (np.exp(hbar / (k_B * T) * (omega_0 + v*k + c*k**2)) - 1)**2

			return num / denom

		CV = 0
		for i in range(2): #Sum for all considered polarizations
			k_max = self.k_max_array[i]
			v_i = self.v_array[i]
			c_i = self.c_array[i]
			omega_0_i = self.omega_0_array[i]

			CV += integrate.quad(f_C_V, 0, k_max, args = (T, v_i, c_i, omega_0_i))[0]
			
			if i == 0: #Degeneracy of each polarization (2 TA and 1 LA)
				CV *= 2

		return CV * self.V / (2*np.pi**2)

	def fill_arrays(self):
		for T in self.Ts:
			N_T = self.N(T)
			E_T = self.E(T)
			v = self.v_avg(T)
			CV_T = self.C_V(T)

			self.Ns.append(N_T) #N per unit volume
			self.E_tot.append(E_T) #E per unit volume
			self.Es.append(E_T / N_T) #E per phonon per unit volume
			self.ws.append(E_T / (hbar * N_T)) #w_avg
			self.vs.append(v / N_T) #v_avg
			self.CVs.append(CV_T) #Cv per unit volume
			self.MFPs.append(3 * N_T * self.k_bulk / (v * CV_T))

		return self.Ns, self.Es, self.ws, self.vs, self.CVs, self.MFPs, self.E_tot, self.Ts

current_dir = os.getcwd()
array_diffussive_folder = current_dir + '/Arrays_diffussive'
array_balistic_folder = current_dir + '/Arrays_balistic'
array_folder_general = current_dir + '/Arrays_general/V'
array_folder_k = current_dir + '/Arrays_k'
array_folder = current_dir + '/Data/Arrays'

#w_vs_K(k_max_si, vs_si, cs_si, omegas_0_si, 'Silicon')
#w_vs_K(k_max_ge, vs_ge, cs_ge, omegas_0_ge, 'Germanium')

#vg_vs_w(vs_si, cs_si, omegas_0_si, maximum_freqs_si, minimum_freqs_si, 'Silicon')
#vg_vs_w(vs_ge, cs_ge, omegas_0_ge, maximum_freqs_ge, minimum_freqs_ge, 'Germanium')

def fill_arrays():
	
	#Silicon
	properties = ThermalProperties(1, 501, 5000, V, maximum_freqs_si, vs_si, cs_si, omegas_0_si, k_bulk_si, 'Silicon')
	N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()
 
	np.save('N_si', N)
	np.save('E_si', E)
	np.save('w_si', w)
	np.save('v_si', v)
	np.save('CV_si', CV)
	np.save('MFP_si', MFP)
	np.save('Etot_si', E_tot)
	np.save('T_si', T)

	#Germanium
	properties = ThermalProperties(1, 501, 5000, V, maximum_freqs_ge, vs_ge, cs_ge, omegas_0_ge, k_bulk_ge, 'Germanium')
	N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()

	np.save('N_ge', N)
	np.save('E_ge', E)
	np.save('w_ge', w)
	np.save('v_ge', v)
	np.save('CV_ge', CV)
	np.save('MFP_ge', MFP)
	np.save('Etot_ge', E_tot)
	np.save('T_ge', T)

def fill_arrays_K():
	
	#Silicon
	properties = ThermalProperties_K(1, 501, 5000, V, k_max_array, vs_si, cs_si, omegas_0_si, k_bulk_si, 'Silicon')
	N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()
 
	np.save('N_si_k', N)
	np.save('E_si_k', E)
	np.save('w_si_k', w)
	np.save('v_si_k', v)
	np.save('CV_si_k', CV)
	np.save('MFP_si_k', MFP)
	np.save('Etot_si_k', E_tot)
	np.save('T_si_k', T)

	#Germanium
	properties = ThermalProperties_K(1, 501, 5000, V, k_max_array, vs_ge, cs_ge, omegas_0_ge, k_bulk_ge, 'Germanium')
	N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()

	np.save('N_ge_k', N)
	np.save('E_ge_k', E)
	np.save('w_ge_k', w)
	np.save('v_ge_k', v)
	np.save('CV_ge_k', CV)
	np.save('MFP_ge_k', MFP)
	np.save('Etot_ge_k', E_tot)
	np.save('T_ge_k', T)

#os.chdir(array_folder_general)
#fill_arrays()
#print('Finished!')

#os.chdir(array_folder_k)
#fill_arrays_K()

def find_T(value, T): #For a given value of temperature returns the position in the T array
	for i in range(len(T)):
		if T[i] >= value:
			return i


'''
properties = ThermalProperties(1, 1000, 999, V, maximum_freqs_ge, vs_ge, cs_ge, omegas_0_ge, k_bulk_ge, 'Germanium')
N, E, w, v, CV, MFP, E_tot, Ts = properties.fill_arrays()

def match_T(value, E, T):
	for i in range(len(E)):
		if E[i] == value:
			return T[i]

		elif E[i] > value: #If we exceed the value, use interpolation
			return T[i] * value /  E[i]

def find_T(value, T): #For a given value of temperature returns the position in the T array
	for i in range(len(T)):
		if T[i] >= value:
			return i

x = np.linspace(np.log(Ts[0]), np.log(Ts[-1]), 999)

print(E[19], E_tot[19])

T1 = match_T(8.72e-22, E, Ts)
T2 = match_T(3.50e-19, E_tot, Ts)

print(T1, T2)

print(find_T(20, Ts))

plt.plot(np.log(Ts), np.log(E), '-', label=r'E_N=$log(T)$')
plt.plot(np.log(Ts), np.log(E_tot), '-', label=r'E=$log(T)$')
plt.plot(x, 5*x + np.log(E[0]) , '--', label=r'$5T$')
plt.plot(x, 4*x + np.log(E[0]) , '--', label=r'$4T$')
plt.plot(x, 3*x + np.log(E[0]) , '--', label=r'$3T$')
plt.plot(x, 2*x + np.log(E[0]) , '--', label=r'$2T$')
plt.plot(x, x + np.log(E[0]) , '--', label=r'$T$')
plt.legend()
#plt.show()
'''

'''
os.chdir(array_folder_k)
N_ge_k = np.load('N_ge_k.npy') * 1e-24
E_ge_k = np.load('E_ge_k.npy')
w_ge_k = np.load('w_ge_k.npy')
v_ge_k = np.load('v_ge_k.npy')
CV_ge_k = np.load('CV_ge_k.npy')  #Cv per phonon per unit volume
MFP_ge_k = np.load('MFP_ge_k.npy') 
Etot_ge_k = np.load('Etot_ge_k.npy') * 1e-24

Ts = np.load('T_ge_k.npy')

os.chdir(array_folder_general)
N_ge = np.load('N_ge.npy') * 1e-24
E_ge = np.load('E_ge.npy') 
w_ge = np.load('w_ge.npy')
v_ge = np.load('v_ge.npy')
CV_ge = np.load('CV_ge.npy') #Cv per phonon per unit volume
MFP_ge = np.load('MFP_ge.npy')
Etot_ge = np.load('Etot_ge.npy') * 1e-24

print('N_k:', N_ge_k[find_T(20, Ts)])
print('N:', N_ge[find_T(20, Ts)])

print('---------------------------------------------')

print('E_k:', E_ge_k[find_T(20,Ts)])
print('E:', E_ge[find_T(20,Ts)])

print('---------------------------------------------')

print('w_k:', w_ge_k[find_T(20,Ts)])
print('w:', w_ge[find_T(20,Ts)])

print('---------------------------------------------')

print('v_k:', v_ge_k[find_T(20,Ts)])
print('v:', v_ge[find_T(20,Ts)])

print('---------------------------------------------')

print('Cv_k:', CV_ge_k[find_T(20,Ts)])
print('Cv:', CV_ge[find_T(20,Ts)])

print('---------------------------------------------')

print('MFP_k:', MFP_ge_k[find_T(20,Ts)])
print('MFP:', MFP_ge[find_T(20,Ts)])
'''

