#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import lmfit
import ilmfit # interactive plotting of lmfit models

fname = 'data/A344/raw/A344_GdN_FC_050_Oe_160908.rso.raw'
raw_data_fc = np.genfromtxt(fname,
                        delimiter = ',', names = True, skip_header = 30)

# filter columns with no data
used_columns = [c for c in raw_data_fc.dtype.names if ~np.all(np.isnan(raw_data_fc[c]))]
raw_data_fc = raw_data_fc[used_columns]

# calculate how many lines correspond to each rso scan
n_pts = np.nonzero(np.diff(raw_data_fc['Time']))[0][0] + 1
print(f'Number of points per scan: {n_pts}')

# reshape array so that first index corresponds to a whole scan
raw_data_fc = raw_data_fc.reshape(-1, n_pts)

position = raw_data_fc['Position_cm']
voltage = raw_data_fc['Long_Voltage']
voltage_fit = raw_data_fc['Long_Reg_Fit']
start_temperature = raw_data_fc['Start_Temperature_K'][:,0]
field = raw_data_fc['Field_Oe'][:,0]

# interactive plot to switch between scans in the raw data
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Position (cm)')
ax1.set_ylabel('Long voltage (V)')

# make space for slider
fig1.subplots_adjust(bottom = 0.25)

# create a slider which can be used to change index of shown scan
slider_ax1 = fig1.add_axes([0.15, 0.1, 0.75, 0.03])
scan_slider1 = Slider(ax = slider_ax1, label = 'Scan #', valmin = 0, valmax = len(raw_data_fc) - 1, valstep = 1)
raw_line1, = ax1.plot(position[0], voltage[0], label = 'Raw data', marker = '.')
fit_line1, = ax1.plot(position[0], voltage_fit[0], label = 'MPMS Fit', marker = '.')
ax1.set_title(f'Scan {0}, T = {start_temperature[0]:.1f} K, H = {field[0]:.1f} Oe')
def update_1(n):
    # update the data of each line to show the i
    n = int(n)
    raw_line1.set_data(position[n], voltage[n])
    fit_line1.set_data(position[n], voltage_fit[n])
    raw_line1.axes.set_title(f'Scan {n}, T = {start_temperature[n]:.1f} K, H = {field[n]:.1f} Oe')
    raw_line1.axes.relim()
    raw_line1.axes.autoscale_view()
scan_slider1.on_changed(update_1)
ax1.legend()

# Get the scaling factor used by the MPMS
# Ignore possible divisions by zero
with np.errstate(divide = 'ignore', invalid = 'ignore'):
    scale_factor = raw_data_fc['Long_Scaled_Response']/raw_data_fc['Long_Voltage']
    scale_factor = np.nanmean(scale_factor, axis = 1).reshape(-1, 1)
fig2, ax2 = plt.subplots()
ax2.plot(scale_factor, marker = '.')
ax2.set_xlabel('Scan number')
ax2.set_ylabel('Scaling factor')

#
position = raw_data_fc['Position_cm']
scaled_voltage = raw_data_fc['Long_Scaled_Response']
voltage_fit = raw_data_fc['Long_Reg_Fit']
scaled_fit = voltage_fit * scale_factor
start_temperature = raw_data_fc['Start_Temperature_K'][:,0]
field = raw_data_fc['Field_Oe'][:,0]

# interactive plot to switch between scans in the raw data
fig3, ax3 = plt.subplots()
ax3.set_xlabel('Position (cm)')
ax3.set_ylabel('Scaled voltage (V)')

# make space for slider
fig3.subplots_adjust(bottom = 0.25)

# create a slider which can be used to change index of shown scan
slider_ax2 = fig3.add_axes([0.15, 0.1, 0.75, 0.03])
scan_slider2 = Slider(ax = slider_ax2, label = 'Scan #', valmin = 0, valmax = len(raw_data_fc) - 1, valstep = 1)
raw_line2, = ax3.plot(position[0], scaled_voltage[0], label = 'Raw data', marker = '.')
fit_line2, = ax3.plot(position[0], scaled_fit[0], label = 'MPMS Fit', marker = '.')
ax3.set_title(f'Scan {0}, T = {start_temperature[0]:.1f} K, H = {field[0]:.1f} Oe')
def update_2(n):
    # update the data of each line to show the i
    n = int(n)
    raw_line2.set_data(position[n], scaled_voltage[n])
    fit_line2.set_data(position[n], scaled_fit[n])
    raw_line2.axes.set_title(f'Scan {n}, T = {start_temperature[n]:.1f} K, H = {field[n]:.1f} Oe')
    raw_line2.axes.relim()
    raw_line2.axes.autoscale_view()
scan_slider2.on_changed(update_2)
ax3.legend()

# Equation from https://www.qdusa.com/siteDocs/appNotes/1014-213.pdf
# modified so x4 is positive center offset
def rso_response(pos, x1 = 0, x2 = 0, x3 = 1e-4, x4 = 2.5):
    R = 0.97
    L = 1.519
    v_i = np.linspace(0, 1, len(pos))                                          
    X = R**2 + (pos - x4)**2                                                    
    Y = R**2 + (L + (pos - x4))**2                                              
    Z = R**2 + (-L + (pos - x4))**2                                            
    return x1 + x2*v_i + x3*(2*X**(-3/2) - Y**(-3/2) - Z**(-3/2))

# making lmfit model from rso_response which can be used to fit data
rso_model = lmfit.Model(rso_response)

# peak to peak magnitude used to define bounds for x1, x2, x3
vp = raw_data_fc['Long_Scaled_Response'][0].ptp()
scan_fitter = ilmfit.Ilmfit(rso_model,
                   x = raw_data_fc['Position_cm'][0],
                   y = raw_data_fc['Long_Scaled_Response'][0],
                   bounds = {'x1': (-vp, vp), 'x2': (-vp, vp), 'x3': (-vp, vp), 'x4': (-1, 6)})

# fit scan with specified initial parameters
scan_fitter.fit(x1 = 0, x2 = 0, x3 = 1e-5, x4 = 1)
scan_fitter.res.params

# Check what parameters the MPMS fit resulted in
vp = scaled_fit[0].ptp()
mpms_fitter = ilmfit.Ilmfit(rso_model,
                   x = position[0],
                   y = scaled_fit[0],
                   bounds = {'x1': (-vp, vp), 'x2': (-vp, vp), 'x3': (-vp, vp), 'x4': (-1, 6)})

mpms_fitter.fit(x1 = 0, x2 = 0, x3 = 0, x4 = 2.5)
mpms_fitter.res.params

# calculate x3-emu conversion factor using the mpms fit parameters and
# originally saved .dat file
fc_data = np.genfromtxt('data/A344/dat/A344_GdN_FC_050_Oe_160908.rso.dat',
                        delimiter = ',', names = True, skip_header = 30)
emu_conversion_factor = fc_data['Long_Moment_emu'][0]/mpms_fitter.res.params['x3'].value
print(f'x3 to emu conversion factor: {emu_conversion_factor}')

# Example of scan with two clear dipole contributions with similar magnitudes
x, y = position[47], scaled_voltage[47]
vp = y.ptp()
# can create composite models by adding together; need to have unique prefixes
rso_model_n = lmfit.Model(rso_response, prefix = 'd1_') + lmfit.Model(rso_response, prefix = 'd2_')
rso_model_n.set_param_hint('d2_x1', vary = False)
rso_model_n.set_param_hint('d2_x2', vary = False)
rso_model_n.set_param_hint('d1_x4', min = 0, max = 5)
rso_model_n.set_param_hint('d2_x4', min = 0, max = 5)
rso_model_n.set_param_hint('d1_moment', expr = '1.09589*d1_x3')
rso_model_n.set_param_hint('d2_moment', expr = '1.09589*d2_x3')
bounds = {k:(-vp, vp) for k in ['d1_x1', 'd1_x2', 'd1_x3', 'd2_x1', 'd2_x2', 'd2_x3']}
scan_fitter_2 = ilmfit.Ilmfit(rso_model_n, x = x, y = y, bounds = bounds)

scan_fitter_2.fit(d1_x4 = 1, d2_x4 = 2.5)
scan_fitter_2.res.params

scan_fitter.res.params

# fitting two dipole moments to each scan
fits = []
last_params = scan_fitter_2.res.params
last_params['d1_x4'].set(min = 0.97, max = 1.01)
last_params['d2_x4'].set(min = 2.49, max = 2.56)
for i in range(len(raw_data_fc)):
    res = rso_model_n.fit(scaled_voltage[i], pos = position[i], params = last_params)
    fits.append(res)
    last_params = res.params

ilmfit.plot_res(fits[100])

# subtracting high temperature offset dipole from each scan data
fits2 = []
model = lmfit.Model(rso_response, prefix = 'd1_')
model.set_param_hint('moment', expr = '1.09589*d1_x3')
last_params = model.make_params()
last_params['d1_x4'].set(min = 2.52, max = 2.56)
scan_fitter.fit(x1 = 1)
for i in range(len(raw_data_fc)):
    pos = position[i]
    voltage_c = scaled_voltage[i] - scan_fitter.res.eval(pos = pos)
    res = model.fit(voltage_c, pos = pos, params = last_params)
    fits2.append(res)
    last_params = res.params

# overview of fitting parameters for each scan
fig4, axes = plt.subplots(4, 2)
for axi, p in zip(axes.T.flatten(), fits[0].params.keys()):
    axi.set_ylabel(p)
    axi.plot([f.params[p].value for f in fits])
fig4.tight_layout()

fig5, (ax5, ax5b) = plt.subplots(2, 1, sharex = True)
fig5.set_size_inches(6, 6)
temperature = fc_data['Temperature_K']
moment_orig = fc_data['Long_Moment_emu']
moment_refit = np.array([f.params['d2_moment'] for f in fits])
moment_refit2 = np.array([f.params['d1_moment'] for f in fits2])
ax5.plot(temperature, moment_orig, marker = '.', label = 'Original')
ax5.plot(temperature, moment_refit, marker = '.', label = 'Refitted')
ax5.plot(temperature, moment_refit2, marker = '.', label = 'Refitted 2')
ax5b.plot(temperature, moment_orig - moment_refit, color = 'C1', marker = '.', label = r'$\Delta$M')
ax5b.plot(temperature, moment_orig - moment_refit2, color = 'C2', marker = '.', label = r'$\Delta$M2')
ax5b.set_xlabel('Temperature (K)')
ax5.set_ylabel('Moment (emu)')
ax5b.set_ylabel(r'$\Delta$M (emu)')
ax5.legend()
ax5b.legend()
fig5.tight_layout()

# check linearity of inverse susceptibility
fig6, ax6 = plt.subplots(1, 1)
temperature = fc_data['Temperature_K']
applied_field = fc_data['Field_Oe'][0]
moment_refit = np.array([f.params['d2_moment'] for f in fits])
moment_refit2 = np.array([f.params['d1_moment'] for f in fits2])
ax6.plot(temperature, applied_field/moment_refit, marker = '.', label = 'Refitted')
ax6.plot(temperature, applied_field/moment_refit2, marker = '.', label = 'Refitted 2')
ax6.set_xlabel('Temperature (K)')
ax6.set_ylabel('Inverse suscepbility (Oe/emu)')
ax6.legend()
fig6.tight_layout()

# curie weiss with correction for constant moment offset
k_B = 1.380649e-16 # erg/K
mu_b = 9.274009994e-21 # erg/G, emu
# gd ion
g = 2
J = 7/2
mu_eff = mu_b * g * np.sqrt(J*(J + 1))

idx = temperature>90
inv_susc = applied_field/(moment_refit + moment_refit2)*2
m, c = np.polyfit(temperature[idx], inv_susc[idx], 1)
print(f'T_C = {-c/m:.2f} K')
print(f'N_Gd = {3*k_B/(mu_eff**2 * m)}')

plt.show()
