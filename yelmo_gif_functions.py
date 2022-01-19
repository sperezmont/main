"""
Author: Sergio Pérez Montero\n
Date: 13.01.2022\n

Aim: Library for making gifs with Yelmo results\n

Considerations about data dimensions:\n
>>> 3D : (nexperiments, ntimes, ny, nx)\n

This script has two ways to create a gif: map2gif (IMAGEIO lib) and mkGif (GIF lib)\n
Both functions have their own associated function to create the frames Map2DI (map2gif) and mkMap (mkGif)\n
Both map functions are similar\n
map2gif performs faster\n

"""

################################################
import os
import math
from numpy import ma
import numpy as np

import netCDF4 as nc

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.path as mpath

import cmocean.cm as cmo

import imageio
import gif

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


# IMAGEIO library


def Map2DI(data, time, x, y, bar_name, exp_names, levels, contours, con_levels, cmap='cmo.ice_r', log_scale=False, fig_size=[], fontsize=20, SHOW=False, base=10, ltresh=0.1, lscale=1, subs=[10], plotpath=[], file_name='map2D.png', set_ax='On'):
    ''' Plot 2D data from Yelmo in n panels '''
    ''' data.shape = (n, :, :) where n is the number of experiments '''
    nexps, lenx, leny = np.shape(data)
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
    fig.suptitle(r'Time = ' + str((time)*10) + ' yrs', fontsize=2*fontsize)

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

    if log_scale:
        cb = fig.colorbar(im, ax=axes, pad=pad, shrink=0.6,
                          ticks=locator, orientation='horizontal')
    else:
        cb = fig.colorbar(im, ax=axes, pad=pad, shrink=0.6,
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


def map2gif(x, y, data, bar_name, exp_names, times, levels, contours=[], con_levels=[], cmap='cmo.ice_r', log_scale=False, fig_size=[], plotpath=[], file_name='map2gif.gif', FPS=0.6, fontsize=20, base=10, ltresh=0.1, lscale=1, subs=[10], set_ax='On'):
    ''' Generate a gif from nexps experiments and one variable '''

    if plotpath == []:
        outname = os.getcwd() + '/' + file_name
    else:
        outname = plotpath + file_name

    imageio.mimsave(outname, [Map2DI(data[:, i, :, :], times[i], x, y, bar_name, exp_names, levels, contours[:, i, :, :], con_levels, cmap, log_scale, fig_size, fontsize, base=base, ltresh=ltresh, lscale=lscale, subs=subs, set_ax=set_ax)
                              for i in range(len(times))], fps=FPS)
    plt.close()


# GIF library
@gif.frame
def mkMap(data, time, x, y, bar_name, exp_names, levels, contours, con_levels, cmap='cmo.ice_r', log_scale=False, fig_size=[], fontsize=20, base=10, ltresh=0.1, lscale=1, subs=[10]):
    ''' Plot 2D data from Yelmo in n panels and save the image '''
    ''' data.shape = (n, :, :) where n is the number of experiments '''
    nexps, lenx, leny = np.shape(data)
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
    fig.suptitle(r'Time = ' + str((time)*10) +
                 ' yrs', fontsize=2*fontsize)

    for i in range(nexps):
        ax = fig.add_subplot(nrows, ncols, i+1)
        title = LatexFormatter(exp_names[i])
        ax.set_title(title, fontsize=fontsize)
        ax.grid(linestyle='--')

        if log_scale:
            im = ax.contourf(
                x, y, data[i, :, :], cmap=cmap, locator=locator, norm=norm)
        else:
            im = ax.contourf(x, y, data[i, :, :], levels, cmap=cmap)

        if contours != []:
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

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        axes.append(ax)

    if nrows == 1:
        pad = 0.15
    else:
        pad = 0.1

    if log_scale:
        cb = fig.colorbar(im, ax=axes, pad=pad, shrink=0.6,
                          ticks=locator, orientation='horizontal')
    else:
        cb = fig.colorbar(im, ax=axes, pad=pad, shrink=0.6,
                          orientation='horizontal')

    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(label=r''+bar_name, size=fontsize)


def mkGif(x, y, data, bar_name, exp_names, times, levels, contours=[], con_levels=[], cmap='cmo.ice_r', log_scale=False, fig_size=[], plotpath=[], file_name='mkGif.gif', FPS=0.6, fontsize=20, base=10, ltresh=0.1, lscale=1, subs=[10]):
    ''' Generate a gif from nexps experiments and one variable '''
    ''' It uses GIF library but its slower than map2gif '''
    if plotpath == []:
        outname = os.getcwd() + '/' + file_name
    else:
        outname = plotpath + file_name

    frames = []
    for i in range(len(times)):
        frame = mkMap(data[:, i, :, :], times[i], x, y, bar_name, exp_names, levels, contours[:, i, :, :],
                      con_levels, cmap, log_scale, fig_size, fontsize, base=base, ltresh=ltresh, lscale=lscale, subs=subs)
        frames.append(frame)

    gif.save(frames, outname, duration=0.5, unit="s", between="frames")
    plt.close()
