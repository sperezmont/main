#!/home/sergio/apps/anaconda3/envs/yelmo_tools/bin/python3
"""
    Author: Sergio Pérez Montero\n
    Date: 29.03.2022\n

    Aim: Script for plotting a variable (scalar) of different experiments versus parameter value \n
        or the value of a variable as a function of two parameters \n

"""
# Libraries
import os

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib import rc

import cmocean as cmo

import yelmo_functions as yf

# Variables
locdata = '/home/sergio/entra/yelmo_vers/v1.75/yelmox/output/ismip6/bmb-taud_lim_yelmo-v1.75/'
plotpath = '/home/sergio/entra/proyects/ABUMIP/proyects/abumip_yelmo-v1.75/bmb-taudl_32KM/' 
plot_name = 'heatmap-SLR_bmb-taudl_32KM.png'

# size must be len(xaxis_names) * len(yaxis_names), put them as you want to see them in the plot (as an array)
exp_names = [['abum_fcmp_fmb0.0_taudl30e5', 'abum_pmp_fmb0.0_taudl30e5', 'abum_nmp_fmb0.0_taudl30e5'],
                ['abum_fcmp_fmb0.0_taudl30e4', 'abum_pmp_fmb0.0_taudl30e4', 'abum_nmp_fmb0.0_taudl30e4'],
                ['abum_fcmp_fmb0.0_taudl30e3', 'abum_pmp_fmb0.0_taudl30e3', 'abum_nmp_fmb0.0_taudl30e3'],
                ['abum_fcmp_fmb0.0_taudl30e2', 'abum_pmp_fmb0.0_taudl30e2', 'abum_nmp_fmb0.0_taudl30e2']]


#exp_names = [['abum-marine_eps0.5_dtt5.0', 'abum-marine_eps1.0_dtt5.0', 'abum-marine_eps2.0_dtt5.0', 'abum-marine_eps3.0_dtt5.0'],
#              ['abum-marine_eps0.5_dtt4.0', 'abum-marine_eps1.0_dtt4.0', 'abum-marine_eps2.0_dtt4.0','abum-marine_eps3.0_dtt4.0'],
#              ['abum-marine_eps0.5_dtt3.0', 'abum-marine_eps1.0_dtt3.0', 'abum-marine_eps2.0_dtt3.0', 'abum-marine_eps3.0_dtt3.0'],
#              ['abum-marine_eps0.5_dtt2.0', 'abum-marine_eps1.0_dtt2.0', 'abum-marine_eps2.0_dtt2.0', 'abum-marine_eps3.0_dtt2.0'],
#              ['abum-marine_eps0.5_dtt1.0', 'abum-marine_eps1.0_dtt1.0', 'abum-marine_eps2.0_dtt1.0', 'abum-marine_eps3.0_dtt1.0'],
#              ['abum-marine_eps0.5_dtt0.5', 'abum-marine_eps1.0_dtt0.5', 'abum-marine_eps2.0_dtt0.5', 'abum-marine_eps3.0_dtt0.5']] 

var2load = 'V_sle'
var2plot = 'SLR' # 'SLR'
var2plot_units = 'm SLE'
time_index = -1
yelmo_fname = 'yelmo1D.nc' # default

xaxis_label = 'ytopo.bmb_gl_method'
xaxis_names = ['FCMP', 'PMP', 'NMP']    # as a string
yaxis_label = 'ydyn.taud_lim'
yaxis_names = ['3e3', '3e4', '3e5 (default)', '3e6']    # as a string

fontsize = 20
nrounds = 2
lab_colors = 'k'
lab_offset, lab_offset_minor = 0.5, 1
LatexON = True
figsize = (10, 8)

# Load Variables
data_array = np.empty(np.shape(exp_names))
exp_names.reverse()

for i in range(0, len(xaxis_names)):
    for j in range(0, len(yaxis_names)):
        if os.path.exists(locdata+exp_names[j][i]+'/yelmo_killed.nc'):
            data_array[j, i] = np.NaN
        elif var2load != var2plot:
            if var2plot == 'SLR':
                try:
                    dataij = yf.LoadYelmo(locdata+exp_names[j][i], var2load, time=[0, time_index], yelmo_file=yelmo_fname)
                    data_array[j,i] = dataij[0] - dataij[time_index]
                except:
                    data_array[j, i] = np.NaN
        else:
            try:
                data_array[j,i] = yf.LoadYelmo(locdata+exp_names[j][i], var2load, time=time_index, yelmo_file=yelmo_fname)
            except:
                data_array[j,i] = np.NaN

# Plot

if LatexON:
    # Activating LaTeX font
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    rc('text', usetex=True)

vmin, vmax = np.nanmin(data_array), np.nanmax(data_array)

if var2plot in ['SLR']:
    if var2plot == 'SLR':
        colormap = 'cmo.deep'
else:
    colormap = 'jet'

fig, ax = plt.subplots(figsize=figsize)

im = plt.imshow(np.flip(data_array, axis=0), cmap=colormap, vmin=vmin, vmax=vmax, extent=[0, len(xaxis_names), 0, len(yaxis_names)])
cb = plt.colorbar(im)

cb.ax.tick_params(labelsize=fontsize)
cb.set_label(label=r''+var2plot+' ('+ var2plot_units +')', size=fontsize)

ax.set_xticks(np.arange(lab_offset, len(xaxis_names)+lab_offset), labels=xaxis_names, fontsize=fontsize)
ax.set_yticks(np.arange(lab_offset, len(yaxis_names)+lab_offset), labels=yaxis_names, fontsize=fontsize)

ax.set_xticks(np.arange(0, len(xaxis_names)+lab_offset_minor), minor=True)
ax.set_yticks(np.arange(0, len(yaxis_names)+lab_offset_minor), minor=True)

ax.set_xlabel(xaxis_label, fontsize=fontsize)
ax.set_ylabel(yaxis_label, fontsize=fontsize)

for i in range(len(xaxis_names)):
    for j in range(len(yaxis_names)):
        text = ax.text(i+lab_offset, j+lab_offset, round(data_array[j, i], nrounds), color=lab_colors, fontsize=fontsize, ha="center", va="center")

ax.grid(which='minor',color='k', linestyle='-', linewidth=2)
plt.tight_layout()
plt.savefig(plotpath + plot_name)

