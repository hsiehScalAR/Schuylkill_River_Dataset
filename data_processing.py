#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.tri as mtri
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from itertools import product


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Times New Roman"
matplotlib.rc('xtick', labelsize = 25) 
matplotlib.rc('ytick', labelsize = 25) 

"""
Author: Victoria Edwards
For issues please contact: vmedw at seas.upenn.edu 
Date: 08/04/2022
Updated: 07/27/2023

Purpose: Plot data from Depth sensor on the Heron!
"""

def read_data(filename):
    """
    Read data in generated from experiemnts run on 08/04/2022
    This expects the data as:
    - Lat
    - Lon
    - Sonar data
    """
    f = open(filename, 'r')
    lines = f.readlines()
    data = list()
    count = 0
    for line in lines:
        val = line.strip().split(', ')
        if count == 0:
            count += 1
            continue
        else:
            data.append([float(val[0]), float(val[1]), float(val[2])])

    state = pd.read_csv(filename, sep = ", ")
    lat = state['x'][1]
    lon = state['y'][1]
    state = state.rename({'x':'latitude', 'y':'longitude'}, axis = 1)

    return(np.array(data), state, lat, lon)

def convert_lat_lon_to_meters(data, origin_pt = None):
    """
    Convert lat lon to meters and return new list of all data in transformed coordinate frame.
    origin_pt: If you need a unified origin across many data sets then supply an origin lat lon coordinate. All data will be translated as necessary, if None is used than the first point of the data supplied will be treated as the origin

    This code came from this resource: https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters 
    and was modified for my purposes 

    Another more common way to have done this would be to use utm
    """
    new_data = np.zeros((np.shape(data)))
    new_data[0, 2] = data[0, 2]
    rad_of_earth = 6378.137

    # First point as origin update the poses based on the new distance plus the angle between Lat and Lon
    for i in range(1, len(data[:,0])): 
        dist_lat = data[i, 0] * math.pi / 180.0 - data[i - 1, 0] * math.pi / 180.0
        dist_lon = data[i, 1] * math.pi / 180.0 - data[i - 1, 1] * math.pi / 180.0

        a = math.sin(dist_lat / 2.0)**2 + math.cos(data[i - 1, 0] * math.pi / 180) * math.cos(data[i, 0] * math.pi / 180) * math.sin(dist_lon / 2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = rad_of_earth * c


        dist_x = data[i, 0] - data[i-1, 0]
        dist_y = data[i, 1] - data[i-1, 1]
        angle_btw_pts = math.atan2(dist_y, dist_x)

        new_data[i, 0] = new_data[i - 1, 0] + d * 1000.0 * math.cos(angle_btw_pts)  
        new_data[i, 1] = new_data[i - 1, 1] + d * 1000.0 * math.sin(angle_btw_pts)  
        new_data[i, 2] = data[i, 2]

    # Compute the current origin with the desired origin
    if origin_pt is not None:
        dist_lat = data[0, 0] * math.pi / 180.0 - origin_pt[0] * math.pi / 180.0
        dist_lon = data[0, 1] * math.pi / 180.0 - origin_pt[1] * math.pi / 180.0

        a = math.sin(dist_lat / 2.0)**2 + math.cos(origin_pt[0] * math.pi / 180) * math.cos(data[0, 0] * math.pi / 180) * math.sin(dist_lon / 2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = rad_of_earth * c


        dist_x = data[0, 0] - origin_pt[0] 
        dist_y = data[0, 1] - origin_pt[1] 
        angle_btw_pts = math.atan2(dist_y, dist_x)

        trans_x = d * 1000.0 * math.cos(angle_btw_pts)
        trans_y = d * 1000.0 * math.sin(angle_btw_pts)

        # Translate the entire dataset to the new coordinate system! 
        for i in range(len(data[:,0])):
            new_data[i, 0] += trans_x
            new_data[i, 1] += trans_y
        
    #plt.plot(new_data[:,0], new_data[:,1])
    #plt.show()

    return(new_data)

def rotate(data):
    """
    This is to straighten the data so that it is easier to do later things! 
    """

    idx_min = np.argmin(data, axis = 0)
    #print(idx_min, data[idx_min, :])
    idx_max = np.argmax(data, axis = 0)
    #print(idx_max, data[idx_max, :])

    dist_x = data[idx_min[0],0] - 0
    dist_y = data[idx_min[0],1] - 0

    theta = -math.atan2(dist_y, dist_x) + math.pi
    #print(dist_x, dist_y, theta)
    new_data = np.zeros((np.shape(data)))
    for i in range(len(data[:,0])):
        new_data[i, 0] = math.cos(theta) * data[i, 0] - math.sin(theta) * data[i, 1]
        new_data[i, 1] = math.sin(theta) * data[i, 0] + math.cos(theta) * data[i, 1]
        new_data[i, 2] = data[i, 2]
    return(new_data)

def translate(data, val_x, val_y):
    """
    Translate the data by val_x and val_y amount
    """
    new_data = np.zeros((np.shape(data)))
    for i in range(len(data[:,0])):
        new_data[i, 0] = data[i, 0] + val_x
        new_data[i, 1] = data[i, 1] + val_y
        new_data[i, 2] = data[i, 2]
    return(new_data)
        
    
def compensate_for_tides(data, data_time, match_time, tidal_vals = [0.66, 6.40, 0.37, 6.28], tidal_times = [26, 529, 1253, 1805], plot = True):
    """
    Estimate the tidal difference
    For 08/03/2022: sunrise 6:01am-8:12pm, moonrise 11:44am-11:19pm, L 12:26am 0.66ft, H 5:29am 6.40ft, L 12:53pm 0.37ft, H 6:05pm 6.28ft

    For more info on tides consult this resource:
    https://www.tides.net/pennsylvania/1591/?year=2022&month=08
    """
    low_tide0 = -tidal_vals[0] / 3.2808 # m (0:26)
    high_tide0 = -tidal_vals[1] / 3.2808# m (05:29)
    low_tide1 = -tidal_vals[2] / 3.2808 # m (12:53)
    high_tide1 = -tidal_vals[3] / 3.2808 # m (18:05)

    # Make a list of approximations between the tides using linear interpolation between values
    num_pts = 100
    low_high_0 = np.linspace(low_tide0, high_tide0, num_pts)
    high_low_01 = np.flip(np.linspace(low_tide1, high_tide0, num_pts))
    low_high_1 = np.linspace(low_tide1, high_tide1, num_pts)

    tides = np.concatenate([low_high_0, high_low_01, low_high_1])
    
    time_0 = np.linspace(tidal_times[0], tidal_times[1], num_pts)
    time_1 = np.linspace(tidal_times[1], tidal_times[2], num_pts)
    time_2 = np.linspace(tidal_times[2], tidal_times[3], num_pts)

    times = np.concatenate([time_0, time_1, time_2])

    idx = 0
    while data_time > times[idx]:
        idx+= 1

    idx_match = 0
    while match_time > times[idx_match]:
        idx_match += 1

    if plot: 
        plt.plot(time_0, low_high_0, color = "green")
        plt.plot(time_1, high_low_01, color = "red")
        plt.plot(time_2, low_high_1, color = "blue")
        plt.scatter(times[idx], tides[idx], s= 100, marker= "*")
        plt.scatter(times[idx_match], tides[idx_match], s= 100, marker= "*")
        
        plt.show()
    
    # Given the time pick the closest interpolated value to be the correction
    tidal_correction = tides[idx] - tides[idx_match] 
    #print("TIDAL CORRECTION: " + str(tidal_correction))
    
    # Shift all data to be at low tide
    new_data = np.zeros((np.shape(data)))
    for i in range(len(data[:,0])):
        new_data[i, 0] = data[i, 0]
        new_data[i, 1] = data[i, 1]
        new_data[i, 2] = data[i, 2] - tidal_correction
    return(new_data)

def downsample_data(data, downsample_rate):
    """
    Down sample the data
    - downsample_rate (how many data points do you want skipped before adding the next one).

    Returns a new list of all the data
    """
    num_pts = len(data[:,0])
    new_data = list()

    count = 0
    for i in range(num_pts):
        if count == downsample_rate:
            new_data.append([data[i, 0], data[i, 1], data[i, 2]])
            count = 0
        else:
            count += 1

    return(np.array(new_data))
    

def GP_interpolation(data, plot = True):
    """
    Use a GP (Gaussian Process) to fit the data from the Heron data

    Resources consulted:
    - https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
    - https://towardsdatascience.com/implement-a-gaussian-process-from-scratch-2a074a470bce
    """
    max_x = np.max(data[:,0])
    min_x = np.min(data[:,0])
    max_y = np.max(data[:,1])
    min_y = np.min(data[:,1])
    
    gridx = np.arange(min_x, max_x, 5.0)
    gridy = np.arange(min_y, max_y, 5.0)

    x1 = np.linspace(min_x, max_x)
    x2 = np.linspace(min_y, max_y)

    input_space = (np.array([x1, x2])).T

    kernel = RBF(length_scale = 100)

    # Data has noise so need a noise kernel
    # The parameters (0.1), lengthscale, and noise_level all impact the extrapolations outside of the data I have! 
    noise_kernel = 0.1**2 * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5))

    # Alternative method
    noise_kernel_2 = 14.2**2 * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=0.161, noise_level_bounds=(1e-5, 1e5))

    # Build the GP object
    gp = GaussianProcessRegressor(kernel= noise_kernel_2, optimizer = None)#, n_restarts_optimizer = 5)

    # Printing this with the optimzation restarts means that we get optimization estimates of what values should be! 
    #print(gp.kernel_)

    # Get the data we want to fit! 
    X = data[:,0:2]
    y = data[:,2]

    # This makes the function found 0 mean (so when we don't know things will tend to head back towards the mean
    y_mean = np.mean(y)
    y_fit = y - y_mean

    # Fit the data
    gp.fit(X, y_fit)
    x1x2 = np.array(list(product(x1, x2)))

    # Reshape the data to be the right way!
    if plot == True:
        # Make a prediction
        y_pred, MSE = gp.predict(x1x2, return_std=True)

        X0p, X1p = x1x2[:,0].reshape(50,50), x1x2[:,1].reshape(50,50)
        Zp = np.reshape(y_pred,(50,50))
        MSEp = np.reshape(MSE, (50, 50))

        # Figure 2c
        fig3 = plt.figure(figsize = (10, 4.5))
        ax3 = fig3.add_subplot()
        surf = ax3.pcolormesh(X0p, X1p, Zp + y_mean )
        #ax3.axis("off")
        ax3.set_xlim([min_x, max_x])
        ax3.set_ylim([min_y, max_y])
        ax3.set_xlabel("X (m)", fontsize = 50)
        ax3.set_ylabel("Y (m)", fontsize = 50)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig3.colorbar(surf, cax = cax)
        cbar.set_label("Depth (m)", fontsize = 50)
        fig3.tight_layout()
        plt.show()
        
    return(gp, y_mean)

def plot_all_data_lat_lon(data, data_corrected_z, date):
    """
    Plot latitude and longitude with updated z depth to have matching tide values
    """
    fig = plt.figure(figsize = (10, 4.5))
    ax = fig.add_subplot()

    min_x = 39.942938166666664
    min_y = -75.20056033333334
    max_x = 39.9443075
    max_y = -75.19805833333334

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    sc = ax.scatter(data[:,1], data[:,0], c = data_corrected_z[:,2])

    cbar = fig.colorbar(sc, cax = cax)

    ax.set_ylabel('Latitude', fontsize = 50)
    ax.set_xlabel('Longitude', fontsize = 50)
    cbar.set_label("Depth (m)", fontsize = 50)
    fig.tight_layout()

    plt.show()

def plot_data(data, date):
    """
    Plot data that has been converted to meters
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim3d(-10, 1)
    ax.scatter(data[:,0], data[:,1], data[:,2], s = 1)

    ax.set_xlabel('X m')
    ax.set_ylabel('Y m')
    ax.set_zlabel('Depth, m')
    ax.set_title('Experimental Data ' + date )
    
    plt.show()
     
if __name__ == "__main__":
    # This is a point near the peir
    origin_point = [ 39.94337933, -75.199777,    1.0]

    # 08/03/2022: Heron data
    data9, state9, lat9, lon9 = read_data('2022-08-03-09-12-48.txt')
    data10, state10, lat10, lon10 = read_data('2022-08-03-11-04-34.txt')
    data10 = data10[160:,] # Removes the startup to get to the first waypoint
    # Long square for waypoints
    data15, state15, lat15, lon15 = read_data('2022-08-03-11-34-34.txt')
    # Long run parallel
    data16, state16, lat16, lon16 = read_data('2022-08-03-13-56-12.txt')

    # Pick an origin point that is close to the pier
    data9_in_m = convert_lat_lon_to_meters(data9, origin_point)
    data10_in_m = convert_lat_lon_to_meters(data10, origin_point)
    data15_in_m = convert_lat_lon_to_meters(data15, origin_point)
    data16_in_m = convert_lat_lon_to_meters(data16, origin_point)

    match_time = 1253 #Midday Low tide
    data16_in_m = compensate_for_tides(data16_in_m, 1356, match_time, plot = False)
    data15_in_m = compensate_for_tides(data15_in_m, 1134, match_time, plot = False)
    data10_in_m = compensate_for_tides(data10_in_m, 1104, match_time, plot = False)
    data9_in_m  = compensate_for_tides(data9_in_m, 912, match_time, plot = False)

    # Get data in meters in one array
    data17 = np.concatenate((data9_in_m, data10_in_m, data15_in_m, data16_in_m))
    rotated_data17 = rotate(data17)
    # Sanity check 
    #plot_data(rotated_data17, "08/03/2022")

    # Get lat and lon in one array
    data17_raw = np.concatenate((data9, data10, data15, data16))

    # Plot lat lon vs. the depth data, data17 has tidal corrections incorporated which data17_raw does not have
    plot_all_data_lat_lon(data17_raw, data17, "")

    # Having more data with the statistics built into the function causes major slow downs! 
    data17_down = downsample_data(rotated_data17, 50)
    
    # Figure 2c
    GP_interpolation(data17_down)
