'''
Module with higher-level functions that compute (potentially multiple) summary
quantities on the same AR, where all those climate variables exist in the same
MERRA-2 dataset. This is to make the masking more efficient: if multiple variables
in the same MERRA-2 DataArrays are needed, we load up all of them once and 
compute every quantity we need.

Jimmy Butler
October 2025
'''

import xarray as xr
import numpy as np
from attribute_utils import augment_storm_da
from loading_utils import grab_MERRA2_files

def compute_raw_summaries(storm_da, func_vars_dict, cell_areas, ticker, data_path, half_hour=False):
    '''
    Compute raw AR quantities on a particular AR (raw meaning no climatology subtracted out).

    Inputs:
        storm_da (xarray.DataArray): the storm's binary mask
        func_vars_dict (dictionary): a dictionary whose keys are a 2-tuple of strings, where the first
            string is the name of the variable you'd like to create, and the second is the variable name
            in the MERRA-2 dataset that you will compute the variable summary on. The value is the actual
            function of the variable you would like to compute for your AR (mean, max, over AIS, etc.)
        cell_areas (xarray.DataArray): the DataArray with areas of each grid cell
        ticker (string): the ID for the particular MERRA-2 dataset we are taking from
        data_path (string): path to where the MERRA-2 dataset is stored
        half_hour (boolean): for some reason, some MERRA-2 datasets give times on the half-hour instead of hour.
            If you're using such a dataset, set this flag to true to subtract all times by 30 minutes.

    Outputs:
        summaries (list): a list of the summary quantities, in the order as they appear in func_vars_dict
    '''

    # avoid any issues with -0 not matching 0
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    fnames = grab_MERRA2_files(storm_da, ticker)

    # grab the list of the variables we will need to subset from the MERRA-2 data
    var_lst = np.unique(np.array(list(func_vars_dict.keys()))[:,1])

    # for every AR day, construct a Dataset of only the variables we need
    ds_lst = []
    for fname in fnames:
        ds = xr.open_dataset(data_path + fname)
        ds_lst.append(ds[var_lst].sel(time = ds.time.dt.hour % 3 == 0)) # ARs only defined 3-hourly
    obs_ds = xr.concat(ds_lst, dim='time')

    if half_hour:
        obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5), time=obs_ds.time - np.timedelta64(30, 'm'))
    else:
        obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))

    # for every AR summary quantity, grab the right DataArray and compute the quantity
    summaries = []
    for key, func in func_vars_dict.items():
        single_var_da = obs_ds[key[1]]
        summaries.append(func(storm_da, single_var_da, cell_areas))

    return summaries

def compute_anomaly_summaries(storm_da, func_vars_dict, climatology_dict, cell_areas, ticker, data_path):
    '''
    Same as compute_raw_summaries, but instead of raw quantities, we are first subtracting off climatologies
        to compute anomalies. So far, anomalies considered for IWV, SLP, and 2m-temperature

    Inputs:
        storm_da (xarray.DataArray): the storm binary mask
        func_vars_dict (dictionary): see compute_raw_summaries
        climatology_dict (dictionary): mapping between variable name (key) and monthly climatology DataArray (value)
        cell_areas (xarray.DataArray): the areas of each grid cell
        ticker (string): ID for the MERRA-2 dataset with the variables we want
        data_path (string): the path to those variables

    Outputs:
        summaries (list): the summary quantities in the order they appear in func_vars_dict
    '''
    
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    fnames = grab_MERRA2_files(storm_da, ticker)
    var_lst = np.unique(np.array(list(func_vars_dict.keys()))[:,1])
    
    ds_lst = []
    for fname in fnames:
        ds = xr.open_dataset(data_path + fname)
        ds_lst.append(ds[var_lst].sel(time = ds.time.dt.hour % 3 == 0))
        
    obs_ds = xr.concat(ds_lst, dim='time')
    obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))
    
    summaries = []
    for key, func in func_vars_dict.items():
        actual_da = obs_ds[key[1]]
        climatology = climatology_dict[key[1]]
        climatology = climatology.assign_coords(lat=climatology.lat.round(5), lon=climatology.lon.round(5))
        
        single_var_da = actual_da.groupby('time.month') - climatology
        single_var_da = single_var_da.drop_vars('month')
        summaries.append(func(storm_da, single_var_da, cell_areas))
        
    return summaries

def compute_precip_summaries(storm_da, cell_areas, agg_func, data_path):
    '''
    Function that computes summaries for precipitation variables. Precipitation requires
        separate treatment because even though the footprint of the AR may have left
        a particular cell, any precip that falls within 24 hours is still attributable
        to that AR.

    Inputs:
        storm_da (xarray.DataArray): the storm's binary mask
        cell_areas (xarray.DataArray): the DataArray with the cell areas
        agg_func (function): the function to aggregate over the spatiotemporal footprint of the AR,
            usually cumulative
    '''
    
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    augmented_da = augment_storm_da(storm_da)
    
    fnames = grab_MERRA2_files(augmented_da, 'tavg1_2d_int_Nx')
    
    var_lst = ['PRECLS', 'PRECCU', 'PRECSN']
    
    ds_lst = []
    for fname in fnames:
        ds = xr.open_dataset(data_path + fname)
        shifted = ds.assign_coords(time=ds.time - np.timedelta64(30, 'm'))
        upscaled = (shifted[var_lst]*60*60).resample(time='3h').sum() # don't remember why I did this...
        ds_lst.append(upscaled)
        
    obs_ds = xr.concat(ds_lst, dim='time')
    obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))
    
    summaries = []
    # compute aggregate rainfall
    obs_ds['tot_rainfall'] = obs_ds['PRECCU'] + obs_ds['PRECLS']
    summaries.append(agg_func(augmented_da, obs_ds['tot_rainfall'], cell_areas))
    # compute aggregate snowfall
    summaries.append(agg_func(augmented_da, obs_ds['PRECSN'], cell_areas))
    
    return summaries