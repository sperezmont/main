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
sParPlot = 0   # plots HCHANGE as a function of parameter values, it needs par_labels = [[xaxis], [yaxis]]
sZSRF = 1
sUXY = 0
sGL = 0
sgrnGL = 0
sRMSEwrpd = 0 # Plots the differences and RMSE wrt pd
sGIFS, szgif, shgif, sugif, sdugif, sGLBMBGIF, sTAUDgif, sTAUBgif, sdifTAUgif, sMB = 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
sGIFS3D, zgif3D, hgif3D = 0, 0, 0   # only one at a time

# Experiment
locdata = '/home/sergio/entra/yelmo_vers/v1.75/yelmox/output/ismip6/bmb-sliding-fmb_yelmo-v1.75/' 
experiments = ['abum_pmp-m3q0.5f0.0', 'abum_pmp-m3q0.5f1.0', 'abum_pmp-m3q0.5f10.0','abum_fcmp-m3q0.5f0.0', 'abum_fcmp-m3q0.5f1.0', 'abum_fcmp-m3q0.5f10.0','abum_nmp-m3q0.5f0.0', 'abum_nmp-m3q0.5f1.0', 'abum_nmp-m3q0.5f10.0']
exp_names = 3*['PMP'] + 3*['FCMP'] + 3*['NMP']
control_run = None  # 'abuc_01'  # set to None if needed
out_fldr = '/bmb-sliding-fmb_32KM/' # '/abuk_02-marine_32KM/' # '/abucip_01_32KM/'
plot_name = 'yelmo_abumip-32KM.png' 

# PLOTTING
shades1D = [0, 0, 1]    # abuc, abuk, abum | from Sun et al., 2020
color = 3*['k']+3*['b']+3*['r']#3*['r', 'b', 'k']
linestyles = 9*['']#3*['solid', 'dashed', 'dotted']#3*['solid'] + 3*['dashed'] + 3*['dotted']
markers = 32*[None]
linewidths = 3*[3,3,3]#[5, 3, 2, 3, 3, 2, 3, 3, 2]
fig1D = (18, 5)
fig2D = []
fig_size = [3, 3]  # nrows, ncols
fnt_size1D, fnt_size2D = 28, 35  # 28, 35  # fontsize

xtickslab1D = [0, 100, 200, 300, 400, 500] # [0, 5000, 10000, 15000, 20000, 25000, 30000]
units1D = 'yr'
vaf_lim =[[38, 58],[-1, 15]] # VAF fig limits

set_ax = 'Off'  # Do you want to draw axis?
cbar_orientation = 'horizontal'

# Levels
taud_lvls, taub_lvls, diftau_lvls = np.arange(0, 300 + 2, 2), np.arange(0, 300 + 2, 2), np.arange(-18, 18 + 1, 1)
mb_lvls = np.arange(-400, 5+50, 50)

par_labels = [['$f_c$ = 0.0', '$f_c$ = 1.0', '$f_c$ = 10.0'], ['PMP', 'FCMP', 'NMP']]

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

# -- Labeling
if exp_names == []:
    exp_labels = experiments
else:
    exp_labels = exp_names

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
if (sHCHANGE == 1) or (sParPlot == 1):
    print('*** H_change ***')
    H_change, Hreg_change, Hbas_change = ma.empty(
        (n, lenx, leny)), np.empty((n, 3)), np.empty((n, 2, 19))
    hmask_bed = ma.empty((n, 2, lenx, leny))
    dduxy_s = ma.empty((n, lenx, leny))
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
if sgrnGL == 1:
    print('*** Initial and final grounded cells GL ***')
    grndgl = ma.empty((n, 2, lenx, leny))
if sRMSEwrpd == 1:
    print('*** RMSE wrt PD ***')
    Hrmse = ma.empty((n, lenx, leny))
    Urmse = ma.empty((n, lenx, leny))
    rmse_mask = ma.empty((n, lenx, leny))
    Hrmse_val, Urmse_val = [], []

if sGIFS == 1:
    print('*** making gifs ... (time consuming) ***')
    zmask_bed_gif = ma.empty((n, len(time2D), lenx, leny))
    if szgif == 1:
        z_srf_gif = ma.empty((n, len(time2D), lenx, leny))
    if shgif == 1:
        H_grnd_gif = ma.empty((n, len(time2D), lenx, leny))
    if sugif == 1:
        uxy_s_gif = ma.empty((n, len(time2D), lenx, leny))
    if sdugif == 1:
        duxy_s_gif = ma.empty((n, len(time2D), lenx, leny))
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
        print(vaf[-1])

    if  (sHCHANGE == 1) or (sParPlot == 1):
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            H_change[i, :, :], hmask_bed[i, :, :] = np.NaN, np.NaN
            continue
        H_grnd, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'H_grnd', locdata, time=[0, -1])
        basins, xc, yc = yf.LoadYelmo2D(experiments[i], 'basins', locdata)
        maskbed, xc, yc = yf.LoadYelmo3D(
            experiments[i], 'mask_bed', locdata, time=[0, -1])

        if control_run != None:
            ref = yf.LoadYelmo3D(control_run, 'H_grnd', locdata, time=[0, -1])
            drift = yf.Drift(ref)
            H_grnd = H_grnd - drift
        H_change[i, :, :], Hreg_change[i, :], Hbas_change[i,
                                                          :, :] = yf.Hchange(H_grnd, basins=basins, resolution=res, basins_nasa=locbasins)
        hmask_bed[i, :, :, :] = maskbed
        H_change[i, :, :] = ma.masked_where(
            hmask_bed[i, 0, :, :] == 0, H_change[i, :, :])

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
    if sgrnGL == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            grndgl[i, :, :, :] = np.NaN
            continue
        grnd, xc, yc = yf.LoadYelmo3D(experiments[i], 'H_grnd', locdata, time=[1, -1])
        grndgli, xc, yc = yf.LoadYelmo3D(experiments[i], 'mask_bed', locdata, time=[1, -1])
        grndgl[i, :, :, :] = ma.masked_where(grnd <= 0, grndgli)

    if sRMSEwrpd == 1:
        if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
            Hrmse[i, :, :] = np.NaN
            Urmse[i, :, :] = np.NaN
            continue
        Hrmse[i, :, :], xc, yc = yf.LoadYelmo3D(experiments[i], 'H_ice_pd_err', locdata, time=-1)
        Hrmse_vali = yf.Load1DYelmo2D(experiments[i], 'rmse_H', locdata, time=None)
        Urmse[i, :, :], xc, yc = yf.LoadYelmo3D(experiments[i], 'uxy_s_pd_err', locdata, time=-1)
        Urmse_vali = yf.Load1DYelmo2D(experiments[i], 'rmse_uxy', locdata, time=None)
        maskbed, xc, yc = yf.LoadYelmo3D(experiments[i], 'mask_bed', locdata, time=0)

        Hrmse[i, :, :] = ma.masked_where((maskbed == 0)&(Hrmse[i, :, :] == 0), Hrmse[i, :, :])
        Urmse[i, :, :] = ma.masked_where((maskbed == 0)&(Urmse[i, :, :] == 0), Urmse[i, :, :])
        rmse_mask[i, :, :] = maskbed
        Hrmse_val.append(exp_names[i]+', RMSE = '+str(round(Hrmse_vali[-1],2))+' m')
        Urmse_val.append(exp_names[i]+', RMSE = '+str(round(Urmse_vali[-1],2))+' m/yr')

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
        if sdugif == 1:
            if os.path.exists(locdata+experiments[i]+'/yelmo_killed.nc'):
                duxy_s_gif[i, :, :, :] = np.NaN
                continue
            u_gif, xc, yc = yf.LoadYelmo3D(experiments[i], 'uxy_s', locdata)

            u_gif = ma.masked_where(maskbedgif == 0, u_gif)
            duxy_s_gif[i, :, :, :] = u_gif - u_gif[0, :, :]
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
    ypf.comPlot1D(VAFdata, dVAFdata, r'VAF', r'm SLE', r'SLR', r'm SLE', units1D, xticks1D, xtickslab1D, vaf_lim, locplot+out_fldr, shades, text=False,
                  labels=exp_labels, color=color, linestyles=linestyles, markers=markers, linewidths=linewidths, file_name='vaf-'+plot_name, fontsize=fnt_size1D, fig1D=fig1D)
if  (sHCHANGE == 1) or (sParPlot == 1):
    if sParPlot == 1:
        ypf.ParPlot2D(H_change, par_labels, r'Grounded Ice thickness relative change', np.arange(0, 1.1, 0.1), contours=hmask_bed[:, -1, :, :], contours_levels=[1, 4], cmap='cmo.tempo', plotpath=locplot+out_fldr, file_name='ParPlot-'+plot_name, fontsize=fnt_size2D)
    else:
        ypf.Map2D(H_change, xc, yc, r'Grounded Ice thickness Relative change', exp_labels, np.arange(0, 1.1, 0.1),
                contours=hmask_bed[:, -1, :, :], contours_levels=[1, 4], cmap='cmo.tempo', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Hchange-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax, cbar_orientation=cbar_orientation, fig2D=fig2D)
if sZSRF == 1:
    ypf.Map2D(z_srf, xc, yc, r'Ice surface elevation (km)', exp_labels, np.arange(0, 4.5+0.1, 0.1),
              contours=zmask_bed, contours_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)
if sUXY == 1:
    ypf.Map2D(uxy_s, xc, yc, r'Ice surface velocity (m/yr)', exp_labels, [0, 1e4],
              contours=umask_bed, contours_levels=[1, 4], cmap='cmo.solar_r', log_scale=True, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='uxys-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)
if sGL == 1:
    ypf.com_contMap2D(gl[:, 0, :, :] == 4.0, gl[:, 1, :, :] == 4.0, xc, yc, exp_labels, [], 'b', 'GL at initial stage', 'r', 'GL at final stage', linewidths=0.5, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='GL-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)
if sgrnGL == 1:
    ypf.com_contMap2D(grndgl[:, 0, :, :] == 4.0, grndgl[:, 1, :, :] == 4.0, xc, yc, exp_labels, [], 'b', 'GL at initial stage', 'r', 'GL at final stage', linewidths=0.5, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='grndGL-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax)
if sRMSEwrpd == 1:
    ypf.Map2D(Hrmse, xc, yc, r'Ice thickness error (m)', Hrmse_val, np.arange(-1900, 1900+10, 10),
                contours=rmse_mask, contours_levels=[1, 4], cmap='cmo.balance', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Hrmse-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax, cbar_orientation=cbar_orientation, fig2D=fig2D)
    ypf.Map2D(Urmse, xc, yc, r'Ice surface velocity error (m/yr)', Urmse_val, np.arange(-6000, 6000+10, 10),
                contours=rmse_mask, contours_levels=[1, 4], cmap='cmo.balance',log_scale=True, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Urmse-'+plot_name, fontsize=fnt_size2D, set_ax=set_ax, cbar_orientation=cbar_orientation, fig2D=fig2D)

if sGIFS == 1:
    if szgif == 1:
        ygf.map2gif(xc, yc, z_srf_gif, r'Ice surface elevation (km)',
                    exp_labels, time2D, np.arange(0, 4.5+0.1, 0.1), contours=zmask_bed_gif, con_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
        # ygf.mkGif(xc, yc, z_srf_gif, r'Ice surface elevation (km)',
        #          experiments, times2plot, np.arange(0, 4.5+0.1, 0.1), contours=zmask_bed_gif, con_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+gif_name)
    if shgif == 1:
        ygf.map2gif(xc, yc, H_grnd_gif, r'Grounded ice thickness (m)',
                    exp_labels, time2D, np.arange(0, 4500+500, 500), contours=zmask_bed_gif, con_levels=[1, 4], cmap='cmo.ice_r', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Hgrnd-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sugif == 1:
        ygf.map2gif(xc, yc, uxy_s_gif, r'Ice surface velocity (m/a)',
                    exp_labels, time2D, [0, 1e4], contours=zmask_bed_gif, con_levels=[1, 4], cmap='cmo.solar_r', log_scale=True, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='uxys-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sdugif == 1:
        colorsduxy = ['#3b4cc0', '#4b65d5', '#5d7ce6', '#7092f3', '#83a6fb', '#97b8ff', '#aac7fd', '#bdd2f7', '#ced9ec', '#dfd4d2', '#ead4c8', '#f3c8b2', '#f7b89c', '#f6a586', '#f18e70', '#e7755b', '#da5948', '#c83936', '#b40426']
        ypf.ParPlot2D(duxy_s_gif[:,-1,:,:], par_labels, r'Ice surface velocity change (m/a)',[-2500,-1000,-500,-100,-50,-30,-20,-10,-0.2,0,0.2,10,20,30,50,100,500,1000,2500], contours=zmask_bed_gif[:,-1,:,:], contours_levels=[1, 4], cmap=colorsduxy, plotpath=locplot+out_fldr, file_name='duxys-'+plot_name, fontsize=fnt_size2D)
        ygf.map2gif(xc, yc, duxy_s_gif, r'Ice surface velocity change (m/a)',
                    exp_labels, time2D, [-2500,-1000,-500,-100,-50,-30,-20,-10,-0.2,0,0.2,10,20,30,50,100,500,1000,2500], contours=zmask_bed_gif, con_levels=[1, 4], colors=colorsduxy, log_scale=None, fig_size=fig_size, plotpath=locplot+out_fldr, file_name='duxys-'+gif_name, gamma=0, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sGLBMBGIF == 1:
        bmb_min, bmb_max = ma.min(bmb_gif), ma.max(bmb_gif)
        maxval = max(np.abs(bmb_min), np.abs(bmb_max))
        step_bmb = 0.1*maxval
        bmblevels = np.arange(bmb_min, bmb_max+step_bmb, step_bmb)
        ygf.map2gif(xc, yc, bmb_gif, r'Total basal mass balance (m/a)',
                    exp_labels, time2D, bmblevels, contours=[], con_levels=[0], cmap='cmo.phase', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='GLbmb-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sTAUDgif == 1:
        ygf.map2gif(xc, yc, taud_gif, r'Driving stress ($10^3$ Pa)',
                    exp_labels, time2D, taud_lvls, [], [], cmap='gist_stern', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='taud-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sTAUBgif == 1:
        ygf.map2gif(xc, yc, taub_gif, r'Basal stress ($10^3$ Pa)',
                    exp_labels, time2D, taub_lvls, [], [], cmap='gist_stern', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='taub-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sdifTAUgif == 1:
        ygf.map2gif(xc, yc, diftau_gif, r'Driving stress - Basal stress ($10^3$ Pa)',
                    exp_labels, time2D, diftau_lvls, [], [], cmap='seismic', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='diftau-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)
    if sMB == 1:
        ygf.map2gif(xc, yc, MB_gif, r'Total mass balance (m/a)',
                    exp_labels, time2D, mb_lvls, contours=[], con_levels=[0], cmap='cmo.thermal', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='MB-'+gif_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax)

if sGIFS3D == 1:
    if zgif3D == 1:
        ygf.map2gif(xc, yc, z_srf_gif3, r'Ice surface elevation (km)',
                    exp_labels, time2D, np.arange(0, 4.5+0.1, 0.1), contours=zmask_bed_gif3, con_levels=[1, 4], cmap='jet', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='zsurf-'+gif3_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax, vis='3D')
    if hgif3D == 1:
        ygf.map2gif(xc, yc, H_grnd_gif3, r'Grounded ice thickness (m)',
                    exp_labels, time2D, np.arange(0, 4500+500, 500), contours=zmask_bed_gif3, con_levels=[1, 4], cmap='cmo.ice_r', fig_size=fig_size, plotpath=locplot+out_fldr, file_name='Hgrnd-'+gif3_name, FPS=FPS, fontsize=fnt_size2D, set_ax=set_ax, vis='3D')
