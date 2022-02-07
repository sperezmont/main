'''
    Author: Sergio Pérez Montero
    Date: 2022.01.25
    Aim: Plot Antarctica's surface mass balance map from observational data
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
from cmaps.ocean_cmap import ocean_dark
####################
# Parameters
locdata = '/home/sergio/entra/proyectos/d01/sources/ANT-16KM/'
locplot = '/home/sergio/entra/proyectos/d01/figures/'
file_name = 'ANT-16KM_RACMO23-ERAINT-HYBRID_1981-2010.nc'
plot_name = 'smbmap_ANT-16KM_RACMO23-ERAINT-HYBRID_1981-2010'
plot_format = 'pdf'  # 'eps' # pdf is recommended
resolution = 100  # 1, 10, 100 and so on, 1 generates an image of ~235 Mb!
cmap = 'cmo.balance'

ocean_file = 'ANT-16KM_TOPO-BedMachine.nc'  # geometry + mask file

# Load
data = nc.Dataset(locdata + file_name)

smb = data.variables['smb'][:]

data2 = nc.Dataset(locdata + ocean_file)
ocean = data2.variables['z_bed'][:]
mask = data2.variables['mask'][:]
z_srf = data2.variables['z_srf'][:]

lat = data2.variables['lat2D'][:]
lon = data2.variables['lon2D'][:]

# Some calculations
smb = ma.sum(smb*30, axis=0)
smb = ma.masked_where(mask == 0, smb)

z_srf = ma.masked_where(mask == 0, z_srf)

ocean = ma.masked_where(mask != 0, ocean)

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


def mk_cmap(nportions, values, ncolors, mode='linear', deg=2):
    '''
    Generates a color map splitting the colorbar into 'nportions' with colors='values' with a lenght of ncolors \n
    >>> nportions = len(values) - 1 \n
    >>> ncolors > nportions \n
    >>> values = [[red, green, blue],...] in percentage units \n
    >>> mode, gradation mode, default is 'linear', 'power' \n
    >>> default power mode is 2 --> ax**2 + bx + c \n
    '''
    import numpy as np
    import matplotlib as mpl

    if nportions > len(values)-1:
        raise ValueError('nportions must be equal to len(values) - 1')

    if ncolors < nportions:
        raise ValueError('ncolors must be >= nportions')

    ncol_port = int(ncolors/nportions)
    cmap = []
    for i in range(nportions):
        jcmap = []
        for x in range(ncol_port):
            vals = []
            for j in range(3):
                ini, fin = values[i][j], values[i+1][j]
                if ini == fin:
                    vals.append(np.array(ini))
                elif mode == 'linear':
                    vals.append((fin - ini) * x / ncol_port + ini)
                elif mode == 'power':
                    xpoints = np.arange(0, ncol_port*(1+1/deg), ncol_port/deg)
                    ypoints = np.arange(
                        ini, fin + (fin-ini)/(len(xpoints)-1), (fin-ini)/(len(xpoints)-1))
                    coef = np.polyfit(xpoints, ypoints, deg)
                    term = 0
                    for p in range(0, deg+1):
                        term = term + coef[p]*x**(deg-p)
                    vals.append(term)
            jcmap.append(vals)
        cmap.append(jcmap)
    cmap = np.array(cmap).reshape(nportions*ncol_port, 3)
    cmap = cmap * 255/100  # convert to rgb values
    cmap = mpl.colors.ListedColormap(cmap/255)  # transform to cmap
    return cmap


C = np.array(list2list(ocean_dark, 3))
cm_ocean = Ccmap(C)

cmap = mk_cmap(6, [[0, 20, 40], [0, 30, 60], [0, 40, 90], [
               85, 80, 80], [85, 15, 0], [80, 5, 0], [10, 0, 0]], 255)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# Ocean
ax.contourf(ocean, np.arange(np.min(ocean),
                             np.max(ocean)+100, 100), cmap=cm_ocean)
ax.contour(ocean, np.arange(np.min(ocean),
                            np.max(ocean)+100, 100), cmap=cm_ocean)

# SMB
levels = np.arange(ma.min(smb), ma.max(smb) + resolution, resolution)
norm = mpl.colors.TwoSlopeNorm(vmin=ma.min(smb), vcenter=0, vmax=ma.max(smb))

im = ax.contourf(smb, levels, cmap=cmap, extend='both', norm=norm)
ax.contour(smb, levels, cmap=cmap, extend='both', norm=norm)

cbaxes = cbaxes = fig.add_axes([0.16, 0.15, 0.48, 0.025])
cb = fig.colorbar(im, ax=ax, cax=cbaxes,
                  orientation='horizontal', extendrect=False)
cb.set_ticks([-8, 0, 1000, 2000, 3000, 4000, 5000])
cb.set_ticklabels([0, 1000, 2000, 3000, 4000, 5000])
cb.ax.tick_params(labelsize=16, color='w')
plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='w')
cb.set_label(label=r'\textbf{SMB (kg m$^{-2}$yr$^{-1}$)}', size=20,
             color='w', labelpad=-65, x=0.28)

ax.set_xticks([])
ax.set_yticks([])

# z_srf
ax.contour(mask, [0], colors='k')
cz = ax.contour(z_srf, [500, 1000, 2000, 3000, 4000],
                colors='k', linewidths=0.8, linestyles='--')
manual_locations = [(323, 243), (308, 227), (305, 220), (283, 205), (250, 200)]
ax.clabel(cz, fmt='%i'+r' m', colors='k',
          fontsize=10, manual=manual_locations)

# Coordinates
lat = ma.abs(lat)
clat = ax.contour(lat, [60, 70, 80, 85], colors='grey',
                  linewidths=0.8, linestyles='dotted')
manual_locations = [(190+150, 190-150), (190+98, 190-98),
                    (190+68, 190-68), (190+35, 190-35)]
ax.clabel(clat, fmt=r'\textbf{%i}'+r'\textbf{$\bf{^o}$S}', colors='red',
          fontsize=14, manual=manual_locations)

lon[lon < 0] = lon[lon < 0] + 360  # only positive values
lon[int(len(lon)/2):, int(len(lon)/2)-1] = np.NaN  # dealing with discontinuity
clon = ax.contour(lon, [0, 90, 180, 270], colors='grey',
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
#txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
txt = ax.text(180, 133, r'\textbf{\textit{Ross}}'+'\n'+r'\textbf{\textit{Ice Shelf}}', horizontalalignment='center',
              verticalalignment='center', color='navy', fontsize=12)
#txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
txt = ax.text(88, 185, r'\textbf{\textit{Pine}}'+'\n'+r'\textbf{\textit{Island}}', horizontalalignment='center',
              verticalalignment='center', color='navy', fontsize=12)
#txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])

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
