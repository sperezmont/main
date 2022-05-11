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
locdata = '/home/sergio/entra/yelmo_vers/v1.75/yelmox/output/ismip6/bmb-sliding-fmb_yelmo-v1.75/'
plotpath = '/home/sergio/entra/proyects/ABUMIP/proyects/abumip_yelmo-v1.75/bmb-sliding-fmb_32KM/' 

# size must be len(xaxis_names) * len(yaxis_names), put them as you want to see them in the plot (as an array)
exp_names = [['abuc_pmp-m3q0.5f0.0', 'abuc_pmp-m3q0.5f1.0', 'abuc_pmp-m3q0.5f10.0'],
                ['abuc_fcmp-m3q0.5f0.0', 'abuc_fcmp-m3q0.5f1.0', 'abuc_fcmp-m3q0.5f10.0'],
                ['abuc_nmp-m3q0.5f0.0', 'abuc_nmp-m3q0.5f1.0', 'abuc_nmp-m3q0.5f10.0']]

#exp_names = [['abum-marine_eps0.5_dtt5.0', 'abum-marine_eps1.0_dtt5.0', 'abum-marine_eps2.0_dtt5.0', 'abum-marine_eps3.0_dtt5.0'],
#              ['abum-marine_eps0.5_dtt4.0', 'abum-marine_eps1.0_dtt4.0', 'abum-marine_eps2.0_dtt4.0','abum-marine_eps3.0_dtt4.0'],
#              ['abum-marine_eps0.5_dtt3.0', 'abum-marine_eps1.0_dtt3.0', 'abum-marine_eps2.0_dtt3.0', 'abum-marine_eps3.0_dtt3.0'],
#              ['abum-marine_eps0.5_dtt2.0', 'abum-marine_eps1.0_dtt2.0', 'abum-marine_eps2.0_dtt2.0', 'abum-marine_eps3.0_dtt2.0'],
#              ['abum-marine_eps0.5_dtt1.0', 'abum-marine_eps1.0_dtt1.0', 'abum-marine_eps2.0_dtt1.0', 'abum-marine_eps3.0_dtt1.0'],
#              ['abum-marine_eps0.5_dtt0.5', 'abum-marine_eps1.0_dtt0.5', 'abum-marine_eps2.0_dtt0.5', 'abum-marine_eps3.0_dtt0.5']] 

var2load = 'V_sle'
var2plot = 'V' # 'SLR' 'V'
varlims = []
plot_name = 'heatmap-{}_bmb-fmb-abuc_32KM.png'.format(var2plot)
var2plot_units = 'm SLE'
time_index = -1
yelmo_fname = 'yelmo1D.nc' # default

# y1
yaxis_label = 'Basal melting method at the GL'
yaxis_loc = [0.5, 1.5, 2.5]
yaxis_names = ['nmp', 'fcmp', 'pmp']    # as a string
ylimits_cell = np.arange(1, 4, 1)

# y2
y2axis_on = False
y2axis_label = 'Frontal mass balance scale'
y2axis_loc = [0.5, 1.5, 2.5]
y2axis_names = ['10.0', '1.0', '0.0']    # as a string
y2limits_cell = np.arange(0.5, 4.5, 1)

# x1
xaxis_label = 'Frontal melting scale'
xaxis_loc = [0.5, 1.5, 2.5]
xaxis_names =  ['0.0', '1.0', '10.0']   # as a string
xlimits_cell = np.arange(1, 4, 1)

# x2
x2axis_on = False
x2axis_label = 'Frontal melting scale' #(Regularized Coulomb = C, Power law = P)'
x2axis_loc = [1.5, 4.5, 7.5]
x2axis_names = ['10.0', '1.0', '0.0']
x2limits_cell = []

fontsize = 25
nrounds = 2
lab_colors = 8*['k'] + ['w']
lab_offset, lab_offset_minor = 0.25, 0.5
xoffset, yoffset = 2, 1
LatexON = True
figsize = (7, 7)
pad=0.009 
shrink=0.8
text = True


# Load Variables
data_array = np.empty(np.shape(exp_names))

if os.path.isdir(plotpath) == False:
    os.mkdir(plotpath)

for i in range(0, int(len(exp_names[0][:]))):
    for j in range(0, int(len(exp_names))):
        if os.path.exists(locdata+exp_names[j][i]+'/yelmo_killed.nc'):
            data_array[j, i] = np.NaN
        elif var2load != var2plot:
            if var2plot == 'SLR':
                try:
                    dataij = yf.LoadYelmo(locdata+exp_names[j][i], var2load, time=[0, time_index], yelmo_file=yelmo_fname)
                    data_array[j,i] = dataij[0] - dataij[time_index]
                except:
                    data_array[j, i] = np.NaN
            elif var2plot == 'V':
                try:
                    data_array[j, i] = yf.LoadYelmo(locdata+exp_names[j][i], var2load, time=[time_index], yelmo_file=yelmo_fname)
                except:
                    data_array[j, i] = np.NaN
        else:
            try:
                data_array[j,i] = yf.LoadYelmo(locdata+exp_names[j][i], var2load, time=time_index, yelmo_file=yelmo_fname)
            except:
                data_array[j,i] = np.NaN


mean = np.nanmean(data_array)
std = np.std(data_array)
print(mean, std)
# Plot
if LatexON:
    # Activating LaTeX font
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    rc('text', usetex=True)

if varlims == []:
    vmin, vmax = np.nanmin(data_array), np.nanmax(data_array)
    print(vmin, vmax)
else:
    vmin, vmax = varlims

if var2plot in ['SLR', 'V']:
    if var2plot == 'SLR':
        colormap = 'cmo.amp'#'cmo.thermal'
    elif var2plot == 'V':
        colormap = 'nipy_spectral'
else:
    colormap = 'jet'

fig, ax = plt.subplots(figsize=figsize)

ax.set_title('mean = '+str(round(mean,4))+', std = '+str(round(std,4)), fontsize=fontsize)

im = plt.imshow(data_array, cmap=colormap, vmin=vmin, vmax=vmax, origin='upper',
                 extent=[0, len(xaxis_names), 0, len(yaxis_names)])
cb = plt.colorbar(im, pad=pad, shrink=shrink)

cb.ax.tick_params(labelsize=fontsize)
cb.set_label(label=r''+var2plot+' ('+ var2plot_units +')', size=fontsize)

# y1
ax.set_yticks(yaxis_loc, fontsize=fontsize)
ax.set_yticklabels(yaxis_names, fontsize=fontsize)
ax.set_ylabel(yaxis_label, fontsize=fontsize)
for i in ylimits_cell:
    plt.axhline(y=i, color='k', linestyle='-', linewidth=2)

# y2
if y2axis_on:
    ax2 = ax.secondary_yaxis('right') # ax.twiny()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y2axis_loc)
    ax2.set_yticklabels(y2axis_names, fontsize=fontsize)
    ax2.set_ylabel(y2axis_label, fontsize=fontsize)
    for i in y2limits_cell:
        plt.axhline(y=i, color='k', linestyle='dashed', linewidth=1)
# x1
ax.set_xticks(xaxis_loc, fontsize=fontsize)
ax.set_xticklabels(xaxis_names, fontsize=fontsize)
ax.set_xlabel(xaxis_label, fontsize=fontsize)
for i in xlimits_cell:
    plt.axvline(x=i, color='k', linestyle='-', linewidth=2)

# x2
if x2axis_on:
    ax2 = ax.secondary_xaxis('top') # ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x2axis_loc)
    ax2.set_xticklabels(x2axis_names, fontsize=fontsize)
    ax2.set_xlabel(x2axis_label, fontsize=fontsize)
    for i in x2limits_cell:
        plt.axvline(x=i, color='k', linestyle='dotted', linewidth=1)

if text:
    k = 0
    for y_index, y in enumerate(exp_names):
        for x_index, x in enumerate(exp_names[0][:]):
            label = round(np.flip(data_array, axis=0)[y_index, x_index],2)
            text_x = x_index + lab_offset_minor
            text_y = y_index + lab_offset_minor
            ax.text(text_x, text_y, label, color=lab_colors[k], ha='center', va='center', fontsize=fontsize)
            k = k + 1


plt.tight_layout()
plt.savefig(plotpath + plot_name)

