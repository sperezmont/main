#!/home/sergio/apps/anaconda3/envs/yelmo_tools/bin/python3
#  Libraries
from pyexpat.errors import XML_ERROR_SUSPEND_PE
from tkinter import HORIZONTAL
from matplotlib import units
import yelmo_functions as yf
import yelmo_plot_functions as ypf
import yelmo_gif_functions as ygf

import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import netCDF4 as nc

# Variables
# GENERIC
locplot = '/home/sergio/entra/proyects/ABUMIP/proyects/abumip_yelmo-v1.75/' 
locsources = '/home/sergio/entra/ice_data/sources/ABUMIP_shades/'
locbasins = '/home/sergio/entra/ice_data/Antarctica/ANT-32KM/ANT-32KM_BASINS-nasa.nc'

# Switches (set to 1 what you want to pplot)
sVAF = 0
sHCHANGE = 0
sZSRF = 0
sUXY = 0
sGL = 1
sGIFS, szgif, shgif, sugif, sGLBMBGIF, sTAUDgif, sTAUBgif, sdifTAUgif, sMB = 0, 0, 1, 0, 0, 0, 0, 0, 0
sGIFS3D, zgif3D, hgif3D = 0, 0, 0   # only one at a time

# Experiment
locdata = '/home/sergio/entra/yelmo_vers/v1.75/yelmox/output/ismip6/bmb-fmb_yelmo-v1.75/' 
experiments = ['abum_fcmp_fmb0.0', 'abum_pmp_fmb0.0', 'abum_nmp_fmb0.0',
               'abum_fcmp_fmb1.0', 'abum_pmp_fmb1.0', 'abum_nmp_fmb1.0',
               'abum_fcmp_fmb10.0', 'abum_pmp_fmb10.0', 'abum_nmp_fmb10.0'] 
control_run = None  # 'abuc_01'  # set to None if needed
out_fldr = '/bmb-fmb_32KM/' # '/abuk_02-marine_32KM/' # '/abumip_01_32KM/'

# PLOTTING
plot_name = 'yelmo_abum_bmb-fmb-32KM.png' 
shades1D = [0, 0, 1]    # abuc, abuk, abum | from Sun et al., 2020
color = 4*['orange', 'navy', 'green']
linestyles = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted', 'dotted', 'dashdot', 'dashdot', 'dashdot']
markers = 12*[None]
linewidths = [2, 2, 2, 3, 3, 3, 4, 4, 4, 2, 2, 2]#[2, 2, 2, 4, 4, 4, 2, 2, 2]#, 8, 4, 4, 4, 4]
fig_size = [3, 3]  # nrows, ncols
fnt_size1D, fnt_size2D = 28, 30  # 28, 35  # fontsize

xtickslab1D = [0, 100, 200, 300, 400, 500] # [0, 5000, 10000, 15000, 20000, 25000, 30000] # 
units1D = 'yr'
vaf_lim = [[35, 63],[-4, 20]]#[[20, 65],[-2.5, 35]] # VAF fig limits

set_ax = 'Off'  # Do you want to draw axis?

# Levels
taud_lvls, taub_lvls, diftau_lvls = np.arange(0, 300 + 10, 10), np.arange(0, 300 + 10, 10), np.arange(-20, 20 + 1, 1)
mb_lvls = np.arange(-400, 5+50, 50)

# GIFs
FPS = 1.5

# YELMO simulation
res = 32
region_names = ['WAIS', 'EAIS', 'PAIS']
basin_names = ['1, Filchner-Ronne RSB', '2, Riiser-Larsen, Stancomb, Brunt', '3, Fimbul', '4, Baudouin', '5, Shirase, Holmes', '6, Amery', '7, Shackleton, West', '8, Totten (ASB)', '9, Cook, Ninnis, Mertz (WSB)',
               '10, Rennick (WSB)', '11, Drygalski (WSB)', '12, Ross', '13, Getz', '14, Pine Island, Thwaites (AMS)', '15, Abbot', '16, Wilkins, Stange, Bach, George VI', '17', '18, Larsen C', '19']

# SCRIPT
# -- Directories
if os.path.isdir(locplot) == False:
    print('Plot main directory does not exist!')
if os.path.isdir(locdata) == False:
    print('Data main directory does not exist!')

if os.path.isdir(locplot+out_fldr) == False:
    os.mkdir(locplot+out_fldr)
if os.path.isdir(locplot+out_fldr+'/contributions/') == False:
    os.mkdir(locplot+out_fldr+'/contributions')
    print('/contributions/ made and ready')

# -- Calculations
# ---- Initialization
n = len(experiments)

time1D = yf.LoadYelmo1D(experiments[0], 'time', locdata)
time2D = yf.Load1DYelmo2D(experiments[0], 'time', locdata)
 
xc = yf.Load1DYelmo2D(experiments[0], 'xc', locdata)
yc = yf.Load1DYelmo2D(experiments[0], 'yc', locdata)

lent, lent2D, lenx, leny = len(time1D), len(time2D), len(xc), len(yc)
xticks1D = list(np.arange(0, lent-1 + (lent-1)/(len(xtickslab1D)-1), (lent-1)/(len(xtickslab1D)-1)))

gif_name, gif3_name = plot_name[:-4]+'.gif', plot_name[:-3]+'3D.gif'

# Check if too many experiments and plots
if n > 3:
    if sGIFS == 1:
        if szgif + shgif +sugif > 1:
            print('Too many gifs, it consumes a lot \n Setting all of them to 0 ... \n Change these values and rerun')
            sGIFS = 0
    if sGIFS3D == 1:
        if zgif3D + hgif3D > 1:
            print('Too many 3D gifs, it consumes a lot \n Setting all of them to 0 ... \n Change these values and rerun')
            sGIFS3D = 0


# Initialize arrays
if sVAF == 1:
    print('*** VAF and SLR ***')
    shades = np.empty((3, 2, 2, 51))
    VAFdata, dVAFdata = np.empty((n, lent)), np.empty((n, lent))
    if any(shades1D) != 0:
        kindlist, varlist, abumip_exps = ['min', 'max'], [
            'VAF', 'SLR'], ['ABUC', 'ABUK', 'ABUM']
        data_shades = nc.Dataset(
            locsources + '/ABUMIP_vaf-slr-shades.nc')
    for i in range(3):
        if shades1D[i] != 0:
            for j in range(len(varlist)):
                for k in range(len(kindlist)):
                    shades[i, j, k, :] = data_shades.variables[kindlist[k] +
                                                               varlist[j]+' '+abumip_exps[i]][:]
        else:
            shades[i, :, :, :] = None
if sHCHANGE == 1:
    print('*** H_change ***')
    H_change, Hreg_change, Hbas_change = ma.empty(
        (n, lenx, leny)), np.empty((n, 3)), np.empty((n, 2, 19))
    hmask_bed = ma.empty((n, lenx, leny))
if sZSRF == 1:
    print('*** z_srf ***')
    z_srf = ma.empty((n, lenx, leny))
    zmask_bed = ma.empty((n, lenx, leny))
if sUXY == 1:
    print('*** uxy_s ***')
    uxy_s = ma.empty((n, lenx, leny))
    umask_bed = ma.empty((n, lenx, leny))
if sGL == 1:
    print('*** GL at the beginning and at the end ***')
    gl = ma.empty((n, 2, lenx, leny)) # at the beginning and at the end

if sGIFS == 1:
    print('*** making gifs ... (time consuming) ***')
    zmask_bed_gif = ma.empty((n, len(time2D), lenx, leny))
    if szgif == 1:
        z_srf_gif = ma.empty((n, len(time2D), lenx, leny))
    if shgif == 1:
        H_grnd_gif = ma.empty((n, len(time2D), lenx, leny))
    if sugif == 1:
        uxy_s_gif = ma.empty((n, len(time2D), lenx, leny))
    if sGLBMBGIF == 1:
        gl_gif = ma.empty((n, len(time2D), lenx, leny))
        bmb_gif = ma.empty((n, len(time2D), lenx, leny))
    if sTAUDgif == 1:
        taud_gif = ma.empty((n, len(time2D), lenx, leny))
    if sTAUBgif == 1:
        taub_gif = ma.empty((n, len(time2D), lenx, leny))
    if sdifTAUgif == 1:
        diftau_gif = ma.empty((n, len(time2D), lenx, leny))
    if sMB == 1:
        MB_gif = ma.empty((n, len(time2D), lenx, leny))

if sGIFS3D == 1:
    print('*** making gifs 3D ... (very time consuming) ***')
    zmask_bed_gif3 = ma.empty((n, len(time2D), lenx, leny))
    if zgif3D == 1:
        z_srf_gif3 = ma.empty((n, len(time2D), lenx, leny))
    if hgif3D == 1:
        H_grnd_gif3 = ma.empty((n, len(time2D), lenx, leny))


# ---- Calcs
for i in range(n):
    if sVAF == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            VAFdata[i,:], dVAFdata[i,:] = np.NaN, np.NaN
            continue
        datan = yf.LoadYelmo1D(experiments[i], 'V_sl', locdata)
        if control_run != None:
            ref = yf.LoadYelmo1D(control_run, 'V_sl', locdata)
            drift = yf.Drift(ref)
            datan = datan - drift
        vaf = yf.SLE(datan*1e6)
        dvaf = yf.SLR(datan*1e6)
        VAFdata[i, :], dVAFdata[i, :] = vaf, dvaf

    if sHCHANGE == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            H_change[i, :, :], hmask_bed[i, :, :] = np.NaN, np.NaN
            continue
        H_grnd, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'H_grnd', locdata, time=[0, -1])
        basins, xc, yc = yf.LoadYelmo2D(experiments[i], 'basins', locdata)
        maskbed, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'mask_bed', locdata, time=0)

        if control_run != None:
            ref = yf.LoadYelmo3D(control_run, 'H_grnd', locdata, time=[0, -1])
            drift = yf.Drift(ref)
            H_grnd = H_grnd - drift
        H_change[i, :, :], Hreg_change[i, :], Hbas_change[i,
                                                          :, :] = yf.Hchange(H_grnd, basins=basins, resolution=res, basins_nasa=locbasins)
        hmask_bed[i, :, :] = maskbed
        H_change[i, :, :] = ma.masked_where(
            hmask_bed[i, :, :] == 0, H_change[i, :, :])

        if i == 0:
            print('--> Each contribution is stored with plots')
        with open(locplot + out_fldr + '/contributions/' + 'Hreg_change-' + experiments[i] + '.txt', 'w') as f:
            f.write('#### ' + experiments[i] + ' ####' + '\n')
            f.write('\n')
            f.write('** Contribution to sea-level rise by each region **' + '\n')
            f.write('AIS --> '+ str(np.sum(Hreg_change[i, :])) +'   m SLE' + '\n')
            for j in range(len(region_names)):
                f.write(region_names[j] + ' --> ' +
                        str(Hreg_change[i, j])+'    m SLE' + '\n')
            f.write('\n')
            f.write('** Contribution to sea-level rise by each basin **' + '\n')
            for k in range(len(basin_names)):
                f.write(basin_names[k] + '  --> ' +
                        str(Hbas_change[i, 0, k])+' m SLE' + '\n')

    if sZSRF == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            z_srf[i, :, :], zmask_bed[i, :, :] = np.NaN, np.NaN
            continue
        zsrf, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'z_srf', locdata, time=-1)
        maskbed, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'mask_bed', locdata, time=-1)

        if control_run != None:
            ref = yf.LoadYelmo3D(control_run, 'z_srf', locdata, time=-1)
            drift = yf.Drift(ref)
            zsrf = zsrf - drift

        zsrf = ma.masked_where(maskbed == 0, zsrf)
        z_srf[i, :, :], zmask_bed[i, :, :] = zsrf/1000, maskbed

    if sUXY == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            uxy_s[i, :, :], umask_bed[i, :, :] = np.NaN, np.NaN
            continue
        uxys, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'uxy_s', locdata, time=-1)
        maskbed, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'mask_bed', locdata, time=-1)

        if control_run != None:
            ref = yf.LoadYelmo3D(control_run, 'uxy_s', locdata, time=-1)
            drift = yf.Drift(ref)
            uxys = uxys - drift

        uxys = ma.masked_where(maskbed == 0, uxys)
        uxy_s[i, :, :], umask_bed[i, :, :] = uxys, maskbed
    if sGL == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            gl[i, :, :, :] = np.NaN
            continue
        gl[i, :, :, :], xc, yc = yf.LoadYelmo3D(experiments[i], 'mask_bed', locdata, time=[1, -1])
        
    if sGIFS == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            zmask_bed_gif[i, :, :, :] = np.NaN
        else:
            maskbedgif, xc, yc = yf.LoadYelmo3D(experiments[i], 'mask_bed', locdata)
            zmask_bed_gif[i, :, :, :] = maskbedgif

        if szgif == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                z_srf_gif[i, :, :, :] = np.NaN
                continue
            z_gif, xc, yc = yf.LoadYelmo3D(
                experiments[i], 'z_srf', locdata)

            if control_run != None:
                ref = yf.LoadYelmo3D(control_run, 'z_srf',
                                     locdata)
                drift = yf.Drift(ref)
                z_gif = z_gif - drift

            z_gif = ma.masked_where(maskbedgif == 0, z_gif)
            z_srf_gif[i, :, :, :] = z_gif/1000
        if shgif == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                H_grnd_gif[i, :, :, :] = np.NaN
                continue
            Hgif, xc, yc = yf.LoadYelmo3D(experiments[i], 'H_grnd',  locdata)

            if control_run != None:
                ref = yf.LoadYelmo3D(control_run, 'H_grnd', locdata, time=-1)
                drift = yf.Drift(ref)
                Hgif = Hgif - drift

            Hgif = ma.masked_where(maskbedgif == 0, Hgif)
            H_grnd_gif[i, :, :, :] = Hgif
        if sugif == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                uxy_s_gif[i, :, :, :] = np.NaN
                continue
            u_gif, xc, yc = yf.LoadYelmo3D(experiments[i], 'uxy_s', locdata)

            if control_run != None:
                ref = yf.LoadYelmo3D(control_run, 'uxy_s', locdata, time=-1)
                drift = yf.Drift(ref)
                u_gif = u_gif - drift

            u_gif = ma.masked_where(maskbedgif == 0, u_gif)
            uxy_s_gif[i, :, :, :] = u_gif
        if sGLBMBGIF == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                gl_gif[i, :, :, :], bmb_gif[i, :, :, :] = np.NaN, np.NaN
                continue
            gl_gif[i, :, :, :], xc, yc = yf.LoadYelmo3D(experiments[i], 'dist_grline', locdata)
            bmb_gif[i, :, :, :], xc, yc = yf.LoadYelmo3D(experiments[i], 'bmb', locdata)
        if sTAUDgif == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                taud_gif[i, :, :, :] = np.NaN
                continue
            taud_gif[i, :, :, :], xc, yc = yf.LoadYelmo3D(experiments[i], 'taud', locdata)
            taud_gif[i, :, :, :] = taud_gif[i, :, :, :]/1000
        if sTAUBgif == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                taub_gif[i, :, :, :] = np.NaN
                continue
            taub_gif[i, :, :, :], xc, yc = yf.LoadYelmo3D(experiments[i], 'taub', locdata)
            taub_gif[i, :, :, :] = taub_gif[i, :, :, :]/1000
        if sdifTAUgif == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                diftau_gif[i, :, :, :] = np.NaN
                continue
            taud, xc, yc = yf.LoadYelmo3D(experiments[i], 'taud', locdata)
            taud = ma.masked_where(maskbedgif == 0, taud)
            taub, xc, yc = yf.LoadYelmo3D(experiments[i], 'taub', locdata)
            taub = ma.masked_where(maskbedgif == 0, taub)
            diftau_gif[i, :, :, :] = taud/1000 - taub/1000
        if sMB == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                MB_gif[i, :, :, :] = np.NaN
                continue
            smb, xc, yc = yf.LoadYelmo3D(experiments[i], 'smb', locdata)
            bmb_g, xc, yc = yf.LoadYelmo3D(experiments[i], 'bmb_grnd', locdata)
            bmb_s, xc, yc = yf.LoadYelmo3D(experiments[i], 'bmb_shlf', locdata)
            calv, xc, yc = yf.LoadYelmo3D(experiments[i], 'calv', locdata)

            MB_gif[i, :, :, :] = smb + bmb_g + bmb_s + calv

    if sGIFS3D == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            zmask_bed_gif3[i, :, :, :] = np.NaN
        else:
            maskbedgif3, xc, yc = yf.LoadYelmo3D(experiments[i], 'mask_bed', locdata)
            zmask_bed_gif3[i, :, :, :] = maskbedgif3
        
        if zgif3D == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                z_srf_gif3[i, :, :, :] = np.NaN
                continue

            z_gif3, xc, yc = yf.LoadYelmo3D(
                    experiments[i], 'z_srf', locdata)

            if control_run != None:
                ref = yf.LoadYelmo3D(control_run, 'z_srf', locdata)
                drift = yf.Drift(ref)
                z_gif3 = z_gif3 - drift

            z_gif3 = ma.masked_where(maskbedgif3 == 0, z_gif3)
            z_srf_gif3[i, :, :, :] = z_gif3/1000
        if hgif3D == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                H_grnd_gif3[i, :, :, :] = np.NaN
                continue

            Hgif3, xc, yc = yf.LoadYelmo3D(experiments[i], 'H_grnd',  locdata)

            if control_run != None:
                ref = yf.LoadYelmo3D(control_run, 'H_grnd', locdata, time=-1)
                drift = yf.Drift(ref)
                Hgif3 = Hgif3 - drift

            Hgif3 = ma.masked_where(maskbedgif3 == 0, Hgif3)
            H_grnd_gif3[i, :, :, :] = Hgif3

# -- Plots
if sVAF == 1:
    ypf.comPlot1D(VAFdata, dVAFdata, r'VAF', r'm SLE', r'$\Delta$VAF', r'm SLE', units1D, xticks1D, xtickslab1D, vaf_lim, locplot+out_fldr, shades, text=False,
                  labels=experiments, color=color, linestyles=linestyles, markers=markers, linewidths=linewidths, file_name='vaf-'+plot_name, fontsize=fnt_size1D)
if sHCHANGE == 1:
    ypf.Map2D(H_change, xc, yc, r'Grounded Ice thickness Relative change', experiments, np.arange(0, 1.1, 0.1),
              contours=hmask_bed, contours_levels=[1, 4], cmap='cmo.tempo', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Hchange-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)
if sZSRF == 1:
    ypf.Map2D(z_srf, xc, yc, r'Ice surface elevation (km)', experiments, np.arange(0, 4.5+0.1, 0.1),
              contours=zmask_bed, contours_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)
if sUXY == 1:
    ypf.Map2D(uxy_s, xc, yc, r'Ice surface velocity (m/yr)', experiments, [0, 1e4],
              contours=zmask_bed, contours_levels=[1, 4], cmap='cmo.solar_r', log_scale=True, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='uxys-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)
if sGL == 1:
    ypf.com_contMap2D(gl[:, 0, :, :] == 4.0, gl[:, 1, :, :] == 4.0, xc, yc, experiments, [], 'b', 'GL at initial stage', 'r', 'GL at final stage', linewidths=0.5, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='GL-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)


if sGIFS == 1:
    if szgif == 1:
        ygf.map2gif(xc, yc, z_srf_gif, r'Ice surface elevation (km)',
                    experiments, time2D, np.arange(0, 4.5+0.1, 0.1), contours=zmask_bed_gif, con_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
        # ygf.mkGif(xc, yc, z_srf_gif, r'Ice surface elevation (km)',
        #          experiments, times2plot, np.arange(0, 4.5+0.1, 0.1), contours=zmask_bed_gif, con_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+gif_name)
    if shgif == 1:
        ygf.map2gif(xc, yc, H_grnd_gif, r'Grounded ice thickness (m)',
                    experiments, time2D, np.arange(0, 4500+500, 500), contours=zmask_bed_gif, con_levels=[1, 4], cmap='cmo.ice_r', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Hgrnd-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sugif == 1:
        ygf.map2gif(xc, yc, uxy_s_gif, r'Ice surface velocity (m/a)',
                    experiments, time2D, [0, 1e4], contours=zmask_bed_gif, con_levels=[1, 4], cmap='cmo.solar_r', log_scale=True, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='uxys-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sGLBMBGIF == 1:
        bmb_min, bmb_max = ma.min(bmb_gif), ma.max(bmb_gif)
        maxval = max(np.abs(bmb_min), np.abs(bmb_max))
        step_bmb = 0.1*maxval
        bmblevels = np.arange(bmb_min, bmb_max+step_bmb, step_bmb)
        ygf.map2gif(xc, yc, bmb_gif, r'Total basal mass balance (m/a)',
                    experiments, time2D, bmblevels, contours=[], con_levels=[0], cmap='cmo.phase', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='GLbmb-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sTAUDgif == 1:
        ygf.map2gif(xc, yc, taud_gif, r'Driving stress ($10^3$ Pa)',
                    experiments, time2D, taud_lvls, [], [], cmap='cmo.solar_r', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='taud-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sTAUBgif == 1:
        ygf.map2gif(xc, yc, taub_gif, r'Basal stress ($10^3$ Pa)',
                    experiments, time2D, taub_lvls, [], [], cmap='cmo.solar_r', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='taub-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sdifTAUgif == 1:
        ygf.map2gif(xc, yc, diftau_gif, r'Driving stress - Basal stress ($10^3$ Pa)',
                    experiments, time2D, diftau_lvls, [], [], cmap='cmo.balance', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='diftau-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sMB == 1:
        ygf.map2gif(xc, yc, MB_gif, r'Total mass balance (m/a)',
                    experiments, time2D, mb_lvls, contours=[], con_levels=[0], cmap='cmo.thermal', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='MB-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)

if sGIFS3D == 1:
    if zgif3D == 1:
        ygf.map2gif(xc, yc, z_srf_gif3, r'Ice surface elevation (km)',
                    experiments, time2D, np.arange(0, 4.5+0.1, 0.1), contours=zmask_bed_gif3, con_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+gif3_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax, vis='3D')
    if hgif3D == 1:
        ygf.map2gif(xc, yc, H_grnd_gif3, r'Grounded ice thickness (m)',
                    experiments, time2D, np.arange(0, 4500+500, 500), contours=zmask_bed_gif3, con_levels=[1, 4], cmap='cmo.ice_r', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Hgrnd-'+gif3_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax, vis='3D')
