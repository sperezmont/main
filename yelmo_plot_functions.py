#!/home/sergio/apps/anaconda3/envs/yelmo_tools/bin/python3
"""
Author: Sergio Pérez Montero\n
Date: 22.12.2021\n

Aim: Library for plotting Yelmo results\n

Considerations about data dimensions:\n
>>> 1D : (nexperiments, ntimes)\n
>>> 2D : (nexperiments, ny, nx)\n

"""

################################################
import math
import numpy as np
from numpy import ma

import netCDF4 as nc

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import cmocean.cm as cmo

#################################################

# Activating LaTeX font
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
rc('text', usetex=True)


def LatexFormatter(string):
    '''
    Change strings to something LaTeX can handle
    '''
    bad_c = np.array(['_'])
    for i in range(len(string)):
        if string[i] == bad_c:
            if string[i-1] != '\\':
                string = string.replace(
                    string[i], '\\'+str(bad_c[bad_c == string[i]][0]))

    return string

# 1D variables


def Plot1D(data, name, units, time_units, plotpath, shades=[],  labels=['ABUC', 'ABUK', 'ABUM'], color=['blue', 'red', 'orange'], linestyles=['solid', 'solid', 'solid'], markers=[None, None, None], linewidths=[2, 2, 2], file_name='plot1D.png', fontsize=20):
    ''' Plots the time series of one 1D variable \n
        data.shape = nexps, ntimes 
    '''
    nexps, ntimes = np.shape(data)
    fig, ax = plt.subplots(figsize=(10, 8))
    alpha = [1, 0.7, 0.5]
    shadecolor = ['lightblue', 'lightcoral', 'yellow']

    if shades != []:
        for j in range(3):
            ax.fill_between(np.arange(0, ntimes, round(ntimes/51)), shades[j, 0, 0, :],
                            shades[j, 1, 1, :], alpha=alpha[j], color=shadecolor[j], edgecolor=shadecolor[j])
    for j in range(nexps):
        label = LatexFormatter(labels[j])
        ax.plot(data[j, :], color=color[j], linestyle=linestyles[j],
                linewidth=linewidths[j], label=r''+label)
        ax.text(450, 0.98*data[j, -1], r''+str(round(data[j, -1], 1)) + ' ' + units,
                color=color[j], fontsize=0.8*fontsize, horizontalalignment='center')
        ax.set_xlim([0, 500])
        ax.grid(linestyle='--', alpha=0.5)
        ax.set_xlabel(r'Time (' + time_units + ')', fontsize=fontsize)
        ax.set_ylabel(r''+name + ' (' + units + ')', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=0.8*fontsize)
        ax.tick_params(axis='y', labelsize=0.8*fontsize)
    ax.legend(fontsize=0.8*fontsize)
    plt.tight_layout()
    plt.savefig(plotpath + file_name)


def comPlot1D(data1, data2, name1, units1, name2, units2, time_units, xticks, xtickslab, ylimits, plotpath, shades=[], text=False, labels=['ABUC', 'ABUK', 'ABUM'], color=['blue', 'red', 'orange'], linestyles=['solid', 'solid', 'solid'], markers=[None, None, None], linewidths=[2, 2, 2], file_name='cplot1D.png', fontsize=20):
    ''' Plots the time series of two 1D variables \n
        data1.shape = nexps, ntimes 
    '''
    nexps, ntimes = np.shape(data1)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(18, 8))
    ax, data, names, units = [ax1, ax2], [
        data1, data2], [name1, name2], [units1, units2]

    alpha = [1, 0.7, 0.5]
    shadecolor = ['lightblue', 'lightcoral', 'yellow']
    for i in range(2):
        if shades != []:
            for j in range(3):
                stepping = (ntimes-1)/(len(shades[j,i,0,:])-1)
                ax[i].fill_between(np.arange(0, ntimes -1 + stepping, stepping), shades[j, i, 0, :],
                                   shades[j, i, 1, :], alpha=alpha[j], color=shadecolor[j], edgecolor=shadecolor[j])
            ax[i].set_xlim([0, len(shades[j,i,0,:])])
        else:
            ax[i].set_xlim([0, ntimes])
        for j in range(nexps):
            label = LatexFormatter(labels[j])
            ax[i].plot(data[i][j, :], color=color[j],
                       linestyle=linestyles[j], marker=markers[j], markersize=linewidths[j], linewidth=linewidths[j], label=r''+label)
            if text == True:
                ax[i].text(450, 0.98*data[i][j, -1], r''+str(round(data[i][j, -1], 1)) + ' ' +
                           units[i], color=color[j], fontsize=fontsize, horizontalalignment='center')

        if ylimits[i] != []:
            ax[i].set_ylim(ylimits[i])
            
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xtickslab)
        ax[i].grid(linestyle='--', alpha=0.5)
        ax[i].set_xlabel(r'Time (' + time_units + ')', fontsize=fontsize)
        ax[i].set_ylabel(r''+names[i] + ' (' + units[i] + ')',
                         fontsize=fontsize)
        ax[i].tick_params(axis='x', labelsize=0.8*fontsize)
        ax[i].tick_params(axis='y', labelsize=0.8*fontsize)
    
    if nexps > 3:
        ax2.legend(bbox_to_anchor=(1,0.5), fontsize=0.8*fontsize, loc='center left')
    else:
        ax2.legend(fontsize=0.8*fontsize)
    
    plt.tight_layout()
    plt.savefig(plotpath + file_name)

# 2D variables


def Map2D(data, x, y, bar_name, exp_names, levels, contours, contours_levels, cmap='cmo.ice_r', log_scale=False, fig_size=[], fontsize=20, SHOW=False, base=10, ltresh=0.1, lscale=1, subs=[10], plotpath=[], file_name='map2D.png', set_ax='On'):
    ''' Plot 2D data from Yelmo in n panels \n
        data.shape = (n, :, :) where n is the number of experiments
    '''
    nexps, leny, lenx = np.shape(data)
    axes = []

    if log_scale:
        vmin, vmax = levels[0], levels[-1]
        locator = matplotlib.ticker.SymmetricalLogLocator(
            base=base, linthresh=ltresh, subs=subs)
        locator.tick_values(vmin=vmin, vmax=vmax)
        norm = matplotlib.colors.SymLogNorm(
            base=base, linthresh=ltresh, linscale=lscale, vmin=vmin, vmax=vmax)
    if fig_size == []:
        ncols = min(3, nexps)
        nrows = max(1, math.ceil(nexps/ncols))
    else:
        nrows, ncols = fig_size
    fig = plt.figure(figsize=(7*ncols, 8*nrows))

    for i in range(nexps):
        ax = fig.add_subplot(nrows, ncols, i+1)
        title = LatexFormatter(exp_names[i])
        ax.set_title(r''+title, fontsize=fontsize)
        ax.grid(linestyle='--')
        ax.set_xticks(np.arange(-2500, 2500+1000, 1000))
        ax.set_yticks(np.arange(-2500, 2500+1000, 1000))
        ax.axis(set_ax)

        if log_scale:
            im = ax.contourf(
                x, y, data[i, :, :], cmap=cmap, locator=locator, norm=norm)
        else:
            im = ax.contourf(x, y, data[i, :, :], levels, cmap=cmap)

        if contours != []:
            ax.contour(x, y, contours[i, :, :], contours_levels,
                       colors='k', linewidths=2)
        if nrows > 1:
            if i in np.arange(0, nrows+2*ncols, ncols):
                ax.set_ylabel(r'yc (km)', fontsize=fontsize)
            else:
                ax.set_yticklabels([])
            if i in np.arange(nexps-ncols, nexps, 1):
                ax.set_xlabel(r'xc (km)', fontsize=fontsize)
            else:
                ax.set_xticklabels([])
        elif nrows == 1:
            if i > 0:
                ax.set_yticklabels([])

        ax.tick_params(axis='x', labelsize=0.8*fontsize)
        ax.tick_params(axis='y', labelsize=0.8*fontsize)
        axes.append(ax)

    if nrows == 1:
        pad = 0.15
    else:
        pad = 0.1

    if set_ax == 'Off':
        pad = 0

    if (nrows == 1) & (ncols == 1):
        shrink = 1
    else:
        shrink = 0.6

    if log_scale:
        cb = fig.colorbar(im, ax=axes, pad=pad, shrink=shrink,
                          ticks=locator, orientation='horizontal')
    else:
        cb = fig.colorbar(im, ax=axes, pad=pad, shrink=shrink,
                          orientation='horizontal')

    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(label=r''+bar_name, size=fontsize)
    
    if SHOW:
        plt.show()

    if plotpath != []:
        plt.savefig(plotpath + file_name)

def com_contMap2D(data1, data2, x, y, exp_names, levels, color1, color2, linewidths, fig_size=[], fontsize=20, SHOW=False, plotpath=[], file_name='com_contMap2D.png', set_ax='On'):
    ''' Plot 2 arrays in contours in n panels \n
        data.shape = (n, :, :) where n is the number of experiments
    '''
    nexps, leny, lenx = np.shape(data1) # size(data1) = size(data2)
    axes = []

    if fig_size == []:
        ncols = min(3, nexps)
        nrows = max(1, math.ceil(nexps/ncols))
    else:
        nrows, ncols = fig_size
    fig = plt.figure(figsize=(7*ncols, 8*nrows))

    for i in range(nexps):
        ax = fig.add_subplot(nrows, ncols, i+1)
        title = LatexFormatter(exp_names[i])
        ax.set_title(r''+title, fontsize=fontsize)
        ax.grid(linestyle='--')
        ax.set_xticks(np.arange(-2500, 2500+1000, 1000))
        ax.set_yticks(np.arange(-2500, 2500+1000, 1000))
        ax.axis(set_ax)
        
        if levels == []:
            ax.contour(x, y, data1[i, :, :], colors=color1, linewidths=linewidths)
            ax.contour(x, y, data2[i, :, :], colors=color2, linewidths=linewidths)
        else:
            ax.contour(x, y, data1[i, :, :], levels, colors=color1, linewidths=linewidths)
            ax.contour(x, y, data2[i, :, :], levels, colors=color2, linewidths=linewidths)

        if nrows > 1:
            if i in np.arange(0, nrows+2*ncols, ncols):
                ax.set_ylabel(r'yc (km)', fontsize=fontsize)
            else:
                ax.set_yticklabels([])
            if i in np.arange(nexps-ncols, nexps, 1):
                ax.set_xlabel(r'xc (km)', fontsize=fontsize)
            else:
                ax.set_xticklabels([])
        elif nrows == 1:
            if i > 0:
                ax.set_yticklabels([])

        ax.tick_params(axis='x', labelsize=0.8*fontsize)
        ax.tick_params(axis='y', labelsize=0.8*fontsize)
        axes.append(ax)

    if nrows == 1:
        pad = 0.15
    else:
        pad = 0.1

    if set_ax == 'Off':
        pad = 0

    if (nrows == 1) & (ncols == 1):
        shrink = 1
    else:
        shrink = 0.6
    
    if SHOW:
        plt.show()

    if plotpath != []:
        plt.savefig(plotpath + file_name)

# OLD FUNCTIONS
def Plot2D(data, x, y, bar_name, exp_names, levels, plotpath, cmap='cmo.ice_r', file_name='file_2Dplot.png', contours=[], contours_levels=[], fig_size=[], fontsize=20):
    ''' Plot 2D data from Yelmo in n panels \n
        data.shape = (n, :, :) where n is the number of experiments \n
        For logaritmic scale use logPlot2D 
    '''
    nexps, leny, lenx = np.shape(data)
    axes = []

    if fig_size == []:
        ncols = min(3, nexps)
        nrows = max(1, math.ceil(nexps/ncols))
    else:
        nrows, ncols = fig_size

    fig = plt.figure(figsize=(7*ncols, 8*nrows))
    for i in range(nexps):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_title(exp_names[i], fontsize=fontsize)
        ax.grid(linestyle='--')

        im = ax.contourf(x, y, data[i, :, :], levels, cmap=cmap)
        if contours != []:
            ax.contour(x, y, contours[i, :, :], contours_levels,
                       colors='k', linewidths=2)
        if nrows > 1:
            if i in np.arange(0, nrows+2*ncols, ncols):
                ax.set_ylabel(r'yc (km)', fontsize=fontsize)
            else:
                ax.set_yticklabels([])
            if i in np.arange(nexps-ncols, nexps, 1):
                ax.set_xlabel(r'xc (km)', fontsize=fontsize)
            else:
                ax.set_xticklabels([])
        elif nrows == 1:
            if i > 0:
                ax.set_yticklabels([])

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        axes.append(ax)

    if nrows == 1:
        pad = 0.15
    else:
        pad = 0.1

    cb = fig.colorbar(im, ax=axes, pad=pad, shrink=0.6,
                      orientation='horizontal')
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(label=bar_name, size=fontsize)

    plt.savefig(plotpath + file_name)


def logPlot2D(data, x, y, bar_name, exp_names, plotpath, vmin=0, vmax=1e4, cmap='cmo.ice_r', file_name='file_2Dplot.png', contours=[], contours_levels=[], fig_size=[], fontsize=20, base=10, ltresh=0.1, lscale=1, subs=[10]):
    ''' Plot 2D data from Yelmo in n panels in log-scale \n
        data.shape = (n, :, :) where n is the number of experiments
    '''
    locator = matplotlib.ticker.SymmetricalLogLocator(
        base=base, linthresh=ltresh, subs=subs)
    locator.tick_values(vmin=vmin, vmax=vmax)
    norm = matplotlib.colors.SymLogNorm(
        base=base, linthresh=ltresh, linscale=lscale, vmin=vmin, vmax=vmax)
    extend = 'neither'

    nexps, leny, lenx = np.shape(data)
    axes = []

    if fig_size == []:
        ncols = min(3, nexps)
        nrows = max(1, math.ceil(nexps/ncols))
    else:
        nrows, ncols = fig_size

    fig = plt.figure(figsize=(7*ncols, 8*nrows))
    for i in range(nexps):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_title(exp_names[i], fontsize=fontsize)
        ax.grid(linestyle='--')

        im = ax.contourf(x, y, data[i, :, :], cmap=cmap,
                         locator=locator, norm=norm, extend=extend)
        if contours != []:
            ax.contour(x, y, contours[i, :, :], contours_levels,
                       colors='k', linewidths=2)

        if nrows > 1:
            if i in np.arange(0, nrows+2*ncols, ncols):
                ax.set_ylabel(r'yc (km)', fontsize=fontsize)
            else:
                ax.set_yticklabels([])
            if i in np.arange(nexps-ncols, nexps, 1):
                ax.set_xlabel(r'xc (km)', fontsize=fontsize)
            else:
                ax.set_xticklabels([])
        else:
            if i > 0:
                ax.set_yticklabels([])

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        axes.append(ax)

    if nrows == 1:
        pad = 0.15
    else:
        pad = 0.1

    cb = fig.colorbar(im, ax=axes, pad=pad, shrink=0.6, ticks=locator,
                      orientation='horizontal')
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(label=bar_name, size=fontsize)

    plt.savefig(plotpath + file_name)
