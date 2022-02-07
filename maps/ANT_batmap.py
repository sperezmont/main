'''
    Author: Sergio Pérez Montero
    Date: 2022.01.21
    Aim: Plot Antarctica bathymetry from observational data
'''

#####################
import math
import numpy as np
from numpy import ma

import netCDF4 as nc

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patheffects as PathEffects

import cmocean.cm as cmo
from cmaps.bat_cmap import bat_mapCentered as bat_map
####################
# Parameters
locdata = '/home/sergio/entra/proyectos/d01/sources/ANT-16KM/'
locplot = '/home/sergio/entra/proyectos/d01/figures/'
file_name = 'ANT-16KM_TOPO-BedMachine.nc'  # 'ANT-16KM_TOPO-RTOPO-2.0.1.nc'
plot_name = 'batmap_ANT-16KM_TOPO-BedMachine'  # 'batmap_ANT-16KM_RTOPO-2.0.1'
plot_format = 'pdf'  # 'eps' # pdf is recommended
resolution = 100  # 1, 10, 100 and so on, 1 generates an image of ~235 Mb!
bat_cmap = bat_map

# Load
data = nc.Dataset(locdata + file_name)

lat = data.variables['lat2D'][:]
lon = data.variables['lon2D'][:]

z_bed = data.variables['z_bed'][:]
mask = data.variables['mask'][:]

# Some calculations
bat = z_bed

# Activating LaTeX font
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
rc('text', usetex=True)

# Custom cmaps


def list2list(pylist, slice_len):
    '''
    Transform pylist with 1 dimension to pylist_converted with dimensions (len(pylist), slice_len)
    '''
    pylist_converted = []
    k = 0
    for i in range(int(len(pylist)/slice_len)):
        temp_list = []
        for j in np.arange(k, k + slice_len, 1):
            temp_list.append(pylist[j])
        pylist_converted.append(temp_list)
        k = k + slice_len
    return pylist_converted


def Ccmap(C):
    '''
    Transforms C (numpy array, dimC = (longitude of colors, 3)) into a colormap
    '''
    cm = mpl.colors.ListedColormap(C/255)
    return cm


C = np.array(list2list(bat_cmap, 3))
cm_bat = Ccmap(C)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Bathymetry
norm = mpl.colors.TwoSlopeNorm(vmin=-6000., vcenter=0, vmax=2000.)
im = ax.contourf(bat, np.arange(-6000, 2000 +
                                resolution, resolution), cmap=cm_bat, norm=norm, extend='both')
ax.contour(bat, np.arange(-6000, 2000+resolution, resolution),
           cmap=cm_bat, norm=norm, extend='both')
ax.contour(mask, [0], colors='k')

cbaxes = cbaxes = fig.add_axes([0.15, 0.15, 0.48, 0.025])
cb = fig.colorbar(im, ax=ax, cax=cbaxes,
                  orientation='horizontal', extendrect=True)
cb.set_ticks([-6000, -4000, -2000, 0, 1000, 2000])
cb.set_ticklabels([-6000, -4000, -2000, 0, 1000, 2000])
cb.ax.tick_params(labelsize=16, color='white')
plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='white')
cb.set_label(label=r'\textbf{Bathymetry (m)}', size=20,
             color='white', labelpad=-65, x=0.25)

ax.set_xticks([])
ax.set_yticks([])

# Coordinates
lat = ma.abs(lat)
clat = ax.contour(lat, [60, 70, 80, 85], colors='w',
                  linewidths=0.8, linestyles='dotted')
manual_locations = [(190+150, 190-150), (190+98, 190-98),
                    (190+68, 190-68), (190+35, 190-35)]
ax.clabel(clat, fmt=r'\textbf{%i}'+r'\textbf{$\bf{^o}$S}', colors='red',
          fontsize=14, manual=manual_locations)

lon[lon < 0] = lon[lon < 0] + 360  # only positive values
lon[int(len(lon)/2):, int(len(lon)/2)-1] = np.NaN  # dealing with discontinuity
clon = ax.contour(lon, [0, 90, 180, 270], colors='w',
                  linewidths=0.8, linestyles='dotted')
manual_locations = [(190, 355), (355, 190), (190, 50), (25, 190)]
ax.clabel(clon, fmt=r'\textbf{%i}'+r'\textbf{$\bf{^o}$E}', colors='red',
          fontsize=14, manual=manual_locations)

# South Pole
circle = plt.Circle((190, 190), 1, color='r', zorder=10)
ax.add_patch(circle)
ax.text(190, 190, 'South Pole', horizontalalignment='left',
        verticalalignment='bottom', color='k', fontsize=12)

# Important sites
txt = ax.text(130, 227, r'\textbf{\textit{Filchner-Ronne}}'+'\n'+r'\textbf{\textit{Ice Shelf}}', horizontalalignment='center',
              verticalalignment='center', color='navy', fontsize=12)
txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
txt = ax.text(180, 133, r'\textbf{\textit{Ross}}'+'\n'+r'\textbf{\textit{Ice Shelf}}', horizontalalignment='center',
              verticalalignment='center', color='navy', fontsize=12)
txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
txt = ax.text(88, 185, r'\textbf{\textit{Pine}}'+'\n'+r'\textbf{\textit{Island}}', horizontalalignment='center',
              verticalalignment='center', color='navy', fontsize=12)
txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])

# Main regions
txt = ax.text(245, 245, r'\textbf{EAIS}', horizontalalignment='center',
              verticalalignment='center', color='k', fontsize=24)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
txt = ax.text(120, 170, r'\textbf{WAIS}', horizontalalignment='center',
              verticalalignment='center', color='k', fontsize=24)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
txt = ax.text(65, 250, r'\textbf{APIS}', horizontalalignment='center',
              verticalalignment='center', color='k', fontsize=24, rotation=-30)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

# Some important basins
txt = ax.text(252, 88, r'\textbf{Wilkes}'+'\n'+r'\textbf{Subglacial Basin}', horizontalalignment='center',
              verticalalignment='center', color='k', fontsize=16, fontweight='bold')
txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
txt = ax.text(300, 138, r'\textbf{Aurora}'+'\n'+r'\textbf{Subglacial Basin}', horizontalalignment='center',
              verticalalignment='center', color='k', fontsize=16)
txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
txt = ax.text(173, 270, r'\textbf{Recovery}'+'\n'+r'\textbf{Subglacial Basin}', horizontalalignment='center',
              verticalalignment='center', color='k', fontsize=16)
txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

# Main seas
txt = ax.text(315, 335, r'\textbf{\textit{Southern Ocean}}', horizontalalignment='center',
              verticalalignment='center', color='lightgrey', fontsize=24, rotation=-40)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
txt = ax.text(100, 295, r'\textbf{\textit{Weddell Sea}}', horizontalalignment='center',
              verticalalignment='center', color='lightgrey', fontsize=12)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
txt = ax.text(180, 85, r'\textbf{\textit{Ross Sea}}', horizontalalignment='center',
              verticalalignment='center', color='lightgrey', fontsize=12)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
txt = ax.text(55, 137, r'\textbf{\textit{Amundsen Sea}}', horizontalalignment='center',
              verticalalignment='center', color='lightgrey', fontsize=12)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
txt = ax.text(38, 200, r'\textbf{\textit{Bellinghausen Sea}}', horizontalalignment='center',
              verticalalignment='center', color='lightgrey', fontsize=12)
txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

# plt.show()
plt.savefig(locplot + plot_name + '.' + plot_format, format=plot_format)
