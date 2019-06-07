import sys, re
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QDate, Qt
from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import matplotlib.colors
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patches as mpatches
from scipy import stats

import time
import os
from shutil import copyfile

from CreateArrays import *
from GrayModelClasses import *

current_dir = os.getcwd()
data_folder = current_dir + '/Data'
array_folder = data_folder + '/Arrays'
final_arrays_folder = data_folder + '/Final_arrays'

if not os.path.exists(data_folder): os.mkdir(data_folder)
if not os.path.exists(array_folder): os.mkdir(array_folder)
if not os.path.exists(final_arrays_folder): os.mkdir(final_arrays_folder)


class Window(QMainWindow): 
	def __init__(self):
		QMainWindow.__init__(self)

		os.chdir(data_folder)
		uic.loadUi('GrayModel.ui', self)

		self.showMaximized()

		#Manage Files
		self.manage_files = ManageFiles()

		self.load.clicked.connect(self.obrir_load_folder)

		self.save_energy_evolution.clicked.connect(self.obrir_save_files_energy)
		self.save_phonons_evolution.clicked.connect(self.obrir_save_files_phonons)
		self.save_T_evolution.clicked.connect(self.obrir_save_files_temperature)
		self.save_scattering_evolution.clicked.connect(self.obrir_save_files_scattering)
		self.save_subplots_evolution.clicked.connect(self.obrir_save_files_time_evolution_subplots)

		self.save_T_steady.clicked.connect(self.obrir_save_files_steady_state_T)

		#Save arrays
		self.save_arrays.clicked.connect(self.obrir_save_arrays)

		#Perform simultions
		self.simulate.clicked.connect(self.obrir_simulate)
		self.T_animation.clicked.connect(self.obrir_T_animation)

		#Make plots
		self.make_evolution_plots.clicked.connect(self.obrir_make_evolution_plots)
		self.make_steady_state_plots.clicked.connect(self.obrir_make_steady_state_plots)

		#Plots
		self.energy_evolution_plot.clicked.connect(self.plot_energy_evolution)
		self.phonons_evolution_plot.clicked.connect(self.plot_phonons_evolution)
		self.T_evolution_plot.clicked.connect(self.plot_temperature_evolution)
		self.scattering_evolution_plot.clicked.connect(self.plot_scattering)
		self.time_evolution_subplots.clicked.connect(self.plot_time_evolution_subplots)

		self.steady_T_plot.clicked.connect(self.plot_steady_state_T)

	#Manage files
	def obrir_save_files_energy(self):
		self.manage_files.saveFileDialog('Energy_evolution')

	def obrir_save_files_phonons(self):
		self.manage_files.saveFileDialog('Phonons_evolution')

	def obrir_save_files_temperature(self):
		self.manage_files.saveFileDialog('Temperature_evolution')

	def obrir_save_files_scattering(self):
		self.manage_files.saveFileDialog('Scattering_events')

	def obrir_save_files_time_evolution_subplots(self):
		self.manage_files.saveFileDialog('Time_evolution_subplots')

	def obrir_save_files_steady_state_T(self):
		self.manage_files.saveFileDialog('Steady_state_T')

	def obrir_load_folder(self):
		self.manage_files.openFolderDialog(self.folder_loaded)

	#Save arrays
	def obrir_save_arrays(self):
		os.chdir(array_folder)

		init_T = self.init_T.value()
		final_T = self.final_T.value()
		n = self.n.value()

		self.state.setText('Calculating Si...')

		#Silicon
		properties = ThermalProperties(init_T, final_T, n, maximum_freqs_si, vs_si, cs_si, omegas_0_si, k_bulk_si, 'Silicon')
		N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()
	 
		np.save('N_si', N)
		np.save('E_si', E)
		np.save('w_si', w)
		np.save('v_si', v)
		np.save('CV_si', CV)
		np.save('MFP_si', MFP)
		np.save('Etot_si', E_tot)
		np.save('T_si', T)

		self.state.setText('Calculating Ge...')

		#Germanium
		properties = ThermalProperties(init_T, final_T, n, maximum_freqs_ge, vs_ge, cs_ge, omegas_0_ge, k_bulk_ge, 'Germanium')
		N, E, w, v, CV, MFP, E_tot, T = properties.fill_arrays()

		np.save('N_ge', N)
		np.save('E_ge', E)
		np.save('w_ge', w)
		np.save('v_ge', v)
		np.save('CV_ge', CV)
		np.save('MFP_ge', MFP)
		np.save('Etot_ge', E_tot)
		np.save('T_ge', T)

		QMessageBox.information(self, 'Information', 'Arrays calculated and saved successfully')

		self.state.setText('Finished')

		os.chdir(current_dir)


	#Perform simulations
	def obrir_simulate(self):
		'''
			Convert all lengths in nanometers (typical length scale)
			Convert all times in picoseconds (typical time scale for dt, max_time in nanoseconds)
		'''

		Lx = float(self.Lx.value()) * 1e-9
		Ly = float(self.Ly.value()) * 1e-9
		Lz = float(self.Lz.value()) * 1e-9

		Lx_subcell = float(self.Lx_subcell.value()) * 1e-9
		Ly_subcell = float(self.Ly_subcell.value()) * 1e-9
		Lz_subcell = float(self.Lz_subcell.value()) * 1e-9

		T0 = float(self.T0.value())
		Tf = float(self.Tf.value())
		Ti = float(self.Ti.value())

		t_MAX = float(self.t.value()) * 1e-12
		dt = float(self.dt.value()) * 1e-12
		W = self.W.value()

		gray_model = PhononGas(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W)

		gray_model.init_particles()

		Energy = []
		Phonons = []
		Temperatures = []
		delta_energy = []
		scattering_events = []

		for k in range(gray_model.Nt):
			
			self.progressBar_simulation.setValue(int((k / gray_model.Nt) * 100))

			gray_model.r += gray_model.dt * gray_model.v #Drift

			for i in range(len(gray_model.r)):
				gray_model.check_boundaries(i, gray_model.Lx, gray_model.Ly, gray_model.Lz)

			#interface_scattering()
			#energy_conservation()

			gray_model.re_init_boundary()

			E_subcells, N_subcells = gray_model.calculate_subcell_T(1, gray_model.N_subcells - 1) #Calculate energy before scattering

			scattering_events.append(gray_model.scattering())

			E_subcells_new , N_subcells_new = gray_model.calculate_subcell_T(1, gray_model.N_subcells - 1) #Calculate energy after scattering

			delta_E = np.array(E_subcells_new) - np.array(E_subcells) #Account for loss or gain of energy

			gray_model.energy_conservation(delta_E) #Impose energy conservation

			E_subcells_final, N_subcells_final = gray_model.calculate_subcell_T(1, gray_model.N_subcells - 1) #Calculate final T

			delta_E_final = np.array(E_subcells_final) - np.array(E_subcells)

			delta_energy.append(np.mean(delta_E_final))
			Energy.append(np.sum(E_subcells_final))
			Phonons.append(np.sum(N_subcells_final))
			Temperatures.append(np.mean(gray_model.subcell_Ts[1:gray_model.N_subcells - 1]))

		os.chdir(final_arrays_folder)

		np.save('Subcell_Ts.npy', gray_model.subcell_Ts)
		np.save('Energy.npy', Energy)
		np.save('Phonons.npy', Phonons)
		np.save('Temperatures.npy', Temperatures)
		np.save('Scattering_events.npy', scattering_events)

		QMessageBox.information(self, 'Information', 'Simulation finished!')

		self.progressBar_simulation.setValue(0)

	def obrir_T_animation(self):
		'''
			Convert all lengths in nanometers (typical length scale)
			Convert all times in picoseconds (typical time scale for dt, max_time in nanoseconds)
		'''

		Lx = float(self.Lx.value()) * 1e-9
		Ly = float(self.Ly.value()) * 1e-9
		Lz = float(self.Lz.value()) * 1e-9

		Lx_subcell = float(self.Lx_subcell.value()) * 1e-9
		Ly_subcell = float(self.Ly_subcell.value()) * 1e-9
		Lz_subcell = float(self.Lz_subcell.value()) * 1e-9

		T0 = self.T0.value()
		Tf = self.Tf.value()
		Ti = self.Ti.value()

		t_MAX = float(self.t.value()) * 1e-12
		dt = float(self.dt.value()) * 1e-12
		W = self.W.value()

		gray_model = PhononGas(Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W)

		gray_model.animation_2()


	#Make plots
	def get_parameters(self, filename):
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

	def obrir_make_evolution_plots(self):

		if self.choose_folder.isChecked() and self.folder_loaded.text() == '':
			QMessageBox.warning(self, 'Warning!', 'Folder not loaded!')

		else:

			if self.choose_folder.isChecked() and self.folder_loaded.text() != '':
				os.chdir(self.folder_loaded.text())

			else:

				os.chdir(final_arrays_folder)

			if os.path.exists('Energy.npy') and os.path.exists('Phonons.npy') and os.path.exists('Temperatures.npy') and os.path.exists('Scattering_events.npy'):
				E = np.load('Energy.npy')
				N = np.load('Phonons.npy')
				T = np.load('Temperatures.npy')
				scattering_events = np.load('Scattering_events.npy')

				#With this we can plot the time if we want as xlabel rather than frames
				Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W = self.get_parameters('parameters_used.txt')

				os.chdir(data_folder)

				#Energy plot
				plt.plot(np.linspace(0, len(E), len(E)), E, '-', color='b')
				plt.title('Energy of the system')
				plt.ylabel('E (J)')
				plt.xlabel('Time steps')

				plt.savefig('Energy_evolution.png')
				plt.gcf().clear()

				#Phonons plot
				plt.plot(np.linspace(0, len(N), len(N)), N, '-', color='b')
				plt.title('Phonons of the system')
				plt.ylabel('# of phonons')
				plt.xlabel('Time steps')

				plt.savefig('Phonons_evolution.png')
				plt.gcf().clear()

				#Temperature plot
				plt.plot(np.linspace(0, len(T), len(T)), T, '-', color='b')
				plt.title('Temperature of the system')
				plt.ylabel('Temperature')
				plt.xlabel('Time steps')

				plt.savefig('Temperature_evolution.png')
				plt.gcf().clear()

				#Scattering plot
				plt.plot(np.linspace(0, len(scattering_events), len(scattering_events)), scattering_events, '-', color='b')
				plt.title('Scattering events in time')
				plt.xlabel('Time steps')
				plt.ylabel('# scattering events')
				
				plt.savefig('Scattering_events.png')
				plt.gcf().clear()

				#Subplots
				plt.subplot(2,2, 1)
				plt.plot(np.linspace(0, len(E), len(E)), E, '-', color='b', label='Energy')
				plt.ylabel('E (J)')
				plt.xlabel('Time steps')
				plt.legend()

				plt.subplot(2, 2, 2)
				plt.plot(np.linspace(0, len(N), len(N)), N, '-', color='g', label='# phonons')
				plt.ylabel('# of phonons')
				plt.xlabel('Time steps')
				plt.legend()

				plt.subplot(2, 2, 3)
				plt.plot(np.linspace(0, len(T), len(T)), T, '-', color='r', label='Temperature')
				plt.ylabel('Temperature')
				plt.xlabel('Time steps')
				plt.legend()

				plt.subplot(2, 2, 4)
				plt.plot(np.linspace(0, len(scattering_events), len(scattering_events)), scattering_events, '-', color='c', label='Scattering events')
				plt.xlabel('Time steps')
				plt.ylabel('# scattering events')
				plt.legend()

				plt.subplots_adjust(left=0.12, bottom=0.10, right=0.96, top=0.94, wspace=0.38, hspace=0.26)

				plt.savefig('Time_evolution_subplots.png')	

				QMessageBox.information(self, 'Information', 'Plots done successfully!')		

			else:
				QMessageBox.warning(self, 'Warning!', 'You must simulate the system first!')

	def obrir_make_steady_state_plots(self):

		if self.choose_folder.isChecked() and self.folder_loaded.text() == '':
			QMessageBox.warning(self, 'Warning!', 'Folder not loaded!')

		else:

			if self.choose_folder.isChecked() and self.folder_loaded.text() != '':
				os.chdir(self.folder_loaded.text())

			else:

				os.chdir(final_arrays_folder)

			Lx, Ly, Lz, Lx_subcell, Ly_subcell, Lz_subcell, T0, Tf, Ti, t_MAX, dt, W = self.get_parameters('parameters_used.txt')

			if os.path.exists('Subcell_Ts.npy'):
				T_cells = list(np.load('Subcell_Ts.npy'))

				x = np.linspace(0, Lx, int(round(Lx/Lx_subcell, 0)))

				#Temperature plot
				theo_T = ((T0**4 + Tf**4)/2)**(0.25)

				exp_avg_T = np.mean(T_cells[-1][1 : len(T_cells) - 1, int(Ly / 2), int(Lz/2)])

				plt.plot(x, T_cells[-1][:, int(Ly / 2), int(Lz/2)], '-', marker='.', label='MC simulation T')
				
				if self.balistic.isChecked():
					plt.plot(x, np.linspace(theo_T, theo_T, len(T_cells[-1])), '--', color = 'orange', label='Balistic steady state T=%.2f' % theo_T)
				
				if self.average.isChecked():
					plt.plot(x, np.linspace(exp_avg_T, exp_avg_T , len(T_cells[-1])), '--', color = 'r', label='Experimental steady state avg T=%.2f' % exp_avg_T)

				if self.diffussive.isChecked():
					plt.plot(x, diffussive_T(x, T0, Tf, Lx), ls = '--', color = 'k', label='Diffussive steady state')
				
				plt.title('T after %.2f ps' % (t_MAX * 1e12))
				plt.xlabel('Length (m)')
				plt.ylabel('Temperature (K)')
				plt.legend()

				os.chdir(data_folder)

				plt.savefig('Steady_state_T.png')
				plt.gcf().clear()

				QMessageBox.information(self, 'Information', 'Plots done successfully!')

	#Plots
	def plot_energy_evolution(self):
		self.imageLabel.clear()
		filename = 'Energy_evolution.png'

		os.chdir(data_folder)

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must make the plot first!')

	def plot_phonons_evolution(self):
		self.imageLabel.clear()
		filename = 'Phonons_evolution.png'

		os.chdir(data_folder)

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must make the plot first!')

	def plot_temperature_evolution(self):
		self.imageLabel.clear()
		filename = 'Temperature_evolution.png'

		os.chdir(data_folder)

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must make the plot first!')

	def plot_scattering(self):
		self.imageLabel.clear()
		filename = 'Scattering_events.png'

		os.chdir(data_folder)

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must make the plot first!')

	def plot_time_evolution_subplots(self):
		self.imageLabel.clear()
		filename = 'Time_evolution_subplots.png'

		os.chdir(data_folder)

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must make the plot first!')

	def plot_steady_state_T(self):
		self.imageLabel.clear()
		filename = 'Steady_state_T.png'

		os.chdir(data_folder)

		if os.path.exists(filename):
			image = QImage(filename)

			self.imageLabel.setPixmap(QPixmap.fromImage(image))
		else:
			QMessageBox.warning(self, 'Warning!', 'You must make the plot first!')


	#close event
	def closeEvent(self, event):
		result = QMessageBox.question(self, 'Leaving...','Do you want to exit?', QMessageBox.Yes | QMessageBox.No)
		if result == QMessageBox.Yes:
			
			os.chdir(data_folder)

			if os.path.exists('Energy_evolution.png'): os.remove('Energy_evolution.png')
			if os.path.exists('Phonons_evolution.png'): os.remove('Phonons_evolution.png')
			if os.path.exists('Temperature_evolution.png'): os.remove('Temperature_evolution.png')
			if os.path.exists('Scattering_events.png'): os.remove('Scattering_events.png')
			if os.path.exists('Time_evolution_subplots.png'): os.remove('Time_evolution_subplots.png')

			if os.path.exists('Steady_state_T.png'): os.remove('Steady_state_T.png')

			event.accept()
			
		else:
			event.ignore()

class ManageFiles(QFileDialog):
	def __init__(self):
		QFileDialog.__init__(self)

		self.title = 'Save files'
		self.left = 10
		self.top = 10
		self.width = 640
		self.height = 400 

		self.initUI()

	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

	def saveFileDialog(self, name):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog

		fileName, _ = QFileDialog.getSaveFileName(self, 'Save files') 

		if fileName:
			os.chdir(data_folder)
			if os.path.exists('%s.png' % name): copyfile('%s.png' % name, fileName + '.png')
			else: QMessageBox.warning(self, 'Warning!', 'The plot doesn\'t exist!') 

	def openFileNameDialog(self, name):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog

		fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)

		if fileName:
			name.setText(fileName)

	def openFolderDialog(self, name):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog

		fileName = str(QFileDialog.getExistingDirectory(self, 'Select folder', options=options))

		if fileName:
			name.setText(fileName)


app = QApplication(sys.argv)
_window=Window()
_window.show()
app.exec_()