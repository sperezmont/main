#!/home/sergio/apps/anaconda3/envs/yelmo_tools/bin/python3
"""
Author: Sergio Pérez Montero\n
Date: 13.01.2022\n

Aim: Library for making gifs with Yelmo results\n

Considerations about data dimensions:\n
>>> 3D : (nexperiments, ntimes, ny, nx)\n
"""

################################################
import os
import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

import cmocean.cm as cmo

import imageio

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


def Map2DI(data, time, x, y, bar_name, exp_names, levels, contours, con_levels, cmap='cmo.ice_r', colors=None, log_scale=False, fig_size=[], fontsize=20, SHOW=False, base=10, ltresh=0.1, lscale=1, subs=[10], gamma=0, plotpath=[], file_name='map2D.png', set_ax='On', vis='2D'):
    ''' Plot 2D data from Yelmo in n panels '''
    ''' data.shape = (n, :, :) where n is the number of experiments '''
    nexps, lenx, leny = np.shape(data)
    axes = []

    if log_scale:
        vmin, vmax = levels[0], levels[-1]
        locator = mpl.ticker.SymmetricalLogLocator(base=base, linthresh=ltresh, subs=subs)
        locator.tick_values(vmin=vmin, vmax=vmax)
        norm = mpl.colors.SymLogNorm(
            base=base, linthresh=ltresh, linscale=lscale, vmin=vmin, vmax=vmax)
    if fig_size == []:
        ncols = min(3, nexps)
        nrows = max(1, math.ceil(nexps/ncols))
    else:
        nrows, ncols = fig_size

    fig = plt.figure(figsize=(7*ncols, 8*nrows))
    fig.suptitle(r'Time = ' + str((time)) + ' yrs', fontsize=fontsize)
    if vis == '3D':
        X, Y = np.meshgrid(x, y)

    for i in range(nexps):
        if vis == '3D':
            ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
            ax.view_init(45, 240)
            ax.set_zlim([np.nanmin(levels), 1.5*np.nanmax(levels)])
            ax.set(xlim=[-1800, 1800], ylim=[-1800, 1800])
        else:
            ax = fig.add_subplot(nrows, ncols, i+1)
        title = LatexFormatter(exp_names[i])
        ax.set_title(r''+title, fontsize=fontsize)
        ax.grid(linestyle='--')
        ax.axis(set_ax)

        if vis == '3D':
            if log_scale:
                print('Not implemented yet')
                # im = ax.plot_surface(
                #    X, Y, data[i, :, :], cmap=cmap, locator=locator, norm=norm)
            else:
                im = ax.plot_surface(
                    X, Y, data[i, :, :], alpha=0.8, cmap=cmap, cstride=3, rstride=3, antialiased=False, vmin=np.nanmin(levels), vmax=np.nanmax(levels))
        else:
            if log_scale:
                im = ax.contourf(
                    x, y, data[i, :, :], cmap=cmap, locator=locator, norm=norm)
            if log_scale == None:
                im = ax.contourf(x, y, data[i, :, :], levels, colors=colors)
            else:
                im = ax.contourf(x, y, data[i, :, :], levels, cmap=cmap)

        if contours != []:
            if vis == '3D':
                ax.contour(X, Y, contours[i, :, :],
                           con_levels, zdir='z', offset=0, colors='k')
            else:
                ax.contour(x, y, contours[i, :, :], con_levels, colors='k')

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

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def map2gif(x, y, data, bar_name, exp_names, times, levels, contours=[], con_levels=[], cmap='cmo.ice_r', colors=None, log_scale=False, fig_size=[], plotpath=[], file_name='map2gif.gif', FPS=0.6, fontsize=20, base=10, ltresh=0.1, lscale=1, subs=[10], gamma=0, set_ax='On', vis='2D'):
    ''' Generate a gif from nexps experiments and one variable '''

    if plotpath == []:
        outname = os.getcwd() + '/' + file_name
    else:
        outname = plotpath + file_name

    if contours != []:
        imageio.mimsave(outname, [Map2DI(data[:, i, :, :], times[i], x, y, bar_name, exp_names, levels, contours[:, i, :, :], con_levels, cmap, colors, log_scale, fig_size, fontsize, base=base, ltresh=ltresh, lscale=lscale, subs=subs, gamma=gamma, set_ax=set_ax, vis=vis)
                              for i in range(len(times))], fps=FPS)
    else:
        imageio.mimsave(outname, [Map2DI(data[:, i, :, :], times[i], x, y, bar_name, exp_names, levels, [], con_levels, cmap, colors, log_scale, fig_size, fontsize, base=base, ltresh=ltresh, lscale=lscale, subs=subs, gamma=gamma, set_ax=set_ax, vis=vis)
                              for i in range(len(times))], fps=FPS)
    plt.close()
