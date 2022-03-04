#!/home/sergio/apps/anaconda3/envs/yelmo_tools/bin/python3
"""
Author: Sergio Pérez Montero\n
Date: 20.12.2021\n

Aim: Library for general calculations with Yelmo data\n

"""

################################################
import numpy as np
from numpy import ma
import netCDF4 as nc

#################################################

# Load functions


def LoadYelmo1D(exp_name, var_name, datapath, time=None):
    ''' Loads 1D var_name from datapath/exp_name/yelmo1D.nc '''
    # First, we read the file from exp_name experiment
    yelmo1D = nc.Dataset(datapath + exp_name + '/yelmo1D.nc')

    # Second, we select the times to load
    if time == None:
        data = yelmo1D.variables[var_name][:]
    else:
        data = yelmo1D.variables[var_name][time]
    return data


def Load1DYelmo2D(exp_name, var_name, datapath, time=None):
    ''' Loads 1D var_name from datapath/exp_name/yelmo2D.nc '''
    # First, we read the file from exp_name experiment
    yelmo1D = nc.Dataset(datapath + exp_name + '/yelmo2D.nc')

    # Second, we select the times to load
    if time == None:
        data = yelmo1D.variables[var_name][:]
    else:
        data = yelmo1D.variables[var_name][time]
    return data


def LoadYelmo2D(exp_name, var_name, datapath):
    ''' Loads 2D var_name from datapath/exp_name/yelmo2D.nc '''
    yelmo2D = nc.Dataset(datapath + exp_name + '/yelmo2D.nc')
    data = yelmo2D.variables[var_name][:, :]
    xc, yc = yelmo2D.variables['xc'][:], yelmo2D.variables['yc'][:]
    return data, xc, yc


def LoadYelmo3D(exp_name, var_name, datapath, time=None):
    ''' Loads 3D var_name from datapath/exp_name/yelmo2D.nc '''
    yelmo2D = nc.Dataset(datapath + exp_name + '/yelmo2D.nc')
    if time == None:
        data = yelmo2D.variables[var_name][:, :, :]
        xc, yc = yelmo2D.variables['xc'][:], yelmo2D.variables['yc'][:]
    else:
        data = yelmo2D.variables[var_name][time, :, :]
        xc, yc = yelmo2D.variables['xc'][:], yelmo2D.variables['yc'][:]

    return data, xc, yc

# Load another kind of files


def LoadCSVList(file, sourpath):
    ''' Loads general list of values in file.csv '''
    with open(sourpath + file, newline='') as f:
        file = pd.read_csv(f, delimiter=',', header=None)
    return file

# Operations


def SLE(data, rhoi=0.9167, rhow=1.0, A_oc=3.618*10**8):
    ''' Transforms volume data to m SLE \n
        [data] = km**3 \n
        rhow = 1 Gt/m3 -> "disregarding the minor salinity/density effects of mixing fresh meltwater with seawater"
            More about: https://sealevel.info/conversion_factors.html
    '''
    SLE = rhoi/rhow * 1e3 / A_oc * data
    return SLE


def SLR(data, rhoi=0.9167, rhow=1, A_oc=3.618*10**8):
    ''' Transforms volume data to m SLE and calculates the SLR for each time step \n
        [data] = km**3 \n
        rhow = 1 Gt/m3 -> "disregarding the minor salinity/density effects of mixing fresh meltwater with seawater"
            More about: https://sealevel.info/conversion_factors.html 
    '''
    SLE = rhoi/rhow * 1e3 / A_oc * data
    SLR = np.subtract(SLE[0], SLE)
    return SLR


def Drift(control_data):
    ''' Calculates the drift in a control run '''
    drift = control_data - control_data[0]
    return drift


def Hchange(data, resolution=32, basins=[], basins_nasa=[]):
    ''' Calculates the change in ice thickness \n
        [data] = m  \n
        data.shape = (2, :, :), data has the beginning and the final time steps \n
        If you want the basin contributions basins != [] 
    '''
    # First, we mask the first step and fill the last step with zeros
    iniH, endH = ma.masked_where(
        data[0, :, :] <= 0, data[0, :, :]), data[-1, :, :]
    endH[endH <= 0] = 0   # at the end of the simulation the mask is different

    hloss = endH - iniH  # difference
    Hchange = np.abs(hloss)/iniH    # relative change

    # Contributions
    if basins != []:
        v = -hloss/1e3 * resolution**2  # km3
        vloss_sle = SLE(v)
        vals, bvals = np.empty(3), np.empty((2, 19))
        basins_nasa = nc.Dataset(basins_nasa).variables['mask_regions'][:, :]
        for n in range(3):  # Main regions
            mvalue = ma.masked_where(basins_nasa != n+1, vloss_sle)
            mvalue = ma.sum(mvalue)
            vals[n] = mvalue
        for n in range(0, 19):  # Basins
            mvalue = ma.masked_where(basins != n+1, vloss_sle)
            mvalue = ma.sum(mvalue)
            perchange = ma.masked_where(basins != n+1, Hchange)
            perchange = ma.mean(perchange)
            bvals[0, n], bvals[1, n] = mvalue, perchange

        return Hchange, vals, bvals

# Other stuff


def years2plot(ini=0, fini=51, step=5):
    ''' Generates a list of index for time \n
        step=5 is recommended
    '''
    years_toplot = list(range(ini, fini + step, step))

    if years_toplot[-1] > 50:
        years_toplot = years_toplot[0:-1]
        if years_toplot[-1] != 50:
            years_toplot.append(50)
    if step == 1:
        years_toplot = years_toplot[1:]
        for y in range(len(years_toplot)):
            years_toplot[y] = years_toplot[y] - 1
    else:
        for y in range(len(years_toplot)):
            if y != 0:
                years_toplot[y] = years_toplot[y] - 1

    return years_toplot

def intersperse(lst, item, lenlst, factor):
    '''
        Adds item between the elements of lst \n
        Adapted from https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list 
    '''
    result = [item] * lenlst
    if lenlst % factor == 0:
        result[::factor] = lst
    else:
        result[:(len(lst)-1):factor] = lst
    return result

def convert_dtt(darray, item, time_step, len2convert):
    '''
        Change a time series with time_step to length len2convert \n
    '''
    d = darray.tolist()
    d = intersperse(d, item, len2convert, time_step)
    d = np.array(d)
    d[d == item] = np.NaN
    return d

def convert_array_dtt(darray, time_step, len2convert, lenx, leny):
    '''
        Change a time series (array version) with time_step to length len2convert \n
    '''
    d = np.empty([len2convert, lenx, leny])
    if len(darray[:,0,0]) % 2 == 0:
        d[::time_step,:,:] = darray[:-1,:,:]
    else:
        d[::time_step,:,:] = darray[:,:,:]
    return d
