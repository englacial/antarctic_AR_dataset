'''
Module with higher-level functions that compute (potentially multiple) summary
quantities on the same AR, where all those climate variables exist in the same
MERRA-2 dataset. This is to make the masking more efficient: if multiple variables
in the same MERRA-2 DataArrays are needed, we load up all of them once and 
compute every quantity we need.

This implementation is parallel to the compute_attributes.py script, but specifically
implementations of these functions for the streaming and parallelization workflows.

Jimmy Butler
November 2025
'''

import xarray as xr
import numpy as np
from .attribute_utils import augment_storm_da
from .loading_utils import grab_MERRA2_granules
import ray
import requests
import json

@ray.remote
def compute_summaries(storm_da, func_vars_dict, cell_areas, data_doi, gatekeeper=None, half_hour=False, climatology_ds=None):
    '''
    Compute AR quantities on a particular AR (raw meaning no climatology subtracted out).

    Inputs:
        storm_da (xarray.DataArray): the storm's binary mask
        func_vars_dict (dictionary): a dictionary whose keys are a 2-tuple of strings, where the first
            string is the name of the variable you'd like to create, and the second is the variable name
            in the MERRA-2 dataset that you will compute the variable summary on. The value is the actual
            function of the variable you would like to compute for your AR (mean, max, over AIS, etc.)
        cell_areas (xarray.DataArray): the DataArray with areas of each grid cell
        data_doi (string): the doi of the dataset we wish to stream from
        half_hour (boolean): for some reason, some MERRA-2 datasets give times on the half-hour instead of hour.
            If you're using such a dataset, set this flag to true to subtract all times by 30 minutes.
        climatology_ds (xarray.DataArray): if some of the elements of your func_vars_dict involve anomalies,
            you must include this so anomalies can be computed

    Outputs:
        summaries (list): a list of the summary quantities, in the order as they appear in func_vars_dict
    '''
    granule_lst = grab_MERRA2_granules(storm_da, data_doi)
        

    # handle unstreamable days (as of 11/30, misc. days were dropped from at least one MERRA-2 dataset)
    storm_days = np.unique(storm_da.time.dt.strftime("%Y%m%d"))
    if len(granule_lst) == len(storm_days):
        missing_days = []
    else:
        granule_days = []
        for granule in granule_lst:
            granule_days.append(granule.data_links()[0].split('.')[-2])
        granule_days = np.array(granule_days)
        # find missing granule days
        missing_days = np.setdiff1d(storm_days, granule_days)

        if len(granule_lst) == 0:
            # count the number of columns
            labels = [key for quantity_dict in func_vars_dict.values() for key in quantity_dict.keys()]
            
            return [np.nan]*len(labels), missing_days
        
        # exclude those missing days from the storm dataarray mask
        storm_da = storm_da.sel(time=~storm_da.time.dt.strftime("%Y%m%d").isin(missing_days))
    
    granule_pointers = ray.get(gatekeeper.get_granule_pointers.remote(granule_lst))

    # avoid any issues with -0 not matching 0
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    #granule_pointers = grab_MERRA2_granules(storm_da, data_doi)

    # lazily load the dataset
    obs_ds = xr.open_mfdataset(granule_pointers)

    if half_hour:
        obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5), time=obs_ds.time - np.timedelta64(30, 'm'))
    else:
        obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))

    # for every AR summary quantity, grab the right DataArray and compute the quantity
    summaries = []
    for var_name, quantity_dict in func_vars_dict.items():
        single_var_da = obs_ds[var_name]
        # subset in advance, and compute
        single_var_da = single_var_da.sel(lat=storm_da.lat, lon=storm_da.lon, time=storm_da.time)
        single_var_da = single_var_da.compute()

        for quantity, func in quantity_dict.items():
            # if we are computing an anomaly, access the climatology_ds
            if 'anomaly' in quantity:
                climatology = climatology_ds[var_name]
                var_anomaly = single_var_da.groupby('time.month') - climatology
                var_anomaly = var_anomaly.drop_vars('month')
                summaries.append(func(storm_da, var_anomaly, cell_areas))
            else:
                summaries.append(func(storm_da, single_var_da, cell_areas))

    return summaries, missing_days
    
@ray.remote
def compute_chunk_summaries(chunk_lst, func_vars_dict, cell_areas, data_doi, gatekeeper=None, half_hour=False, climatology_ds=False):
    '''
    Computes summaries for a list of ARs (called a chunk_lst), and loops through the chunk in sequential fashion. Provides an alternative
        way of parallelizing the storm value computations: instead of parallelizing over the list of individual storms, we parallelize over
        chunks of storms, where within each iteration of the parallel loop, we compute quantities sequentially on that chunk.

    Inputs:
        chunk_lst (pd.DataFrame): a dataframe with a data_array column
        func_vars_dict (dictionary): a dictionary whose keys are a 2-tuple of strings, where the first
            string is the name of the variable you'd like to create, and the second is the variable name
            in the MERRA-2 dataset that you will compute the variable summary on. The value is the actual
            function of the variable you would like to compute for your AR (mean, max, over AIS, etc.)
        cell_areas (xarray.DataArray): the DataArray with areas of each grid cell
        data_doi (string): the doi of the dataset we wish to stream from
        half_hour (boolean): for some reason, some MERRA-2 datasets give times on the half-hour instead of hour.
            If you're using such a dataset, set this flag to true to subtract all times by 30 minutes.
        climatology_ds (xarray.DataArray): if some of the elements of your func_vars_dict involve anomalies,
            you must include this so anomalies can be computed

    Outputs:
        summaries_lst (list of lists): a list of lists of the summary quantities, in the order as they appear in func_vars_dict,
            one list for each storm in the chunk
    '''

    summaries_lst = []
    missing_lst = []
    for index, storm in chunk_lst.iterrows():
        summaries, missing_days = compute_summaries(storm.data_array, 
                                      func_vars_dict, 
                                      cell_areas, 
                                      data_doi, 
                                      gatekeeper, 
                                      half_hour, 
                                      climatology_ds)
        summaries_lst.append(summaries)
        missing_lst.append(missing_days)

    return summaries_lst, missing_lst

@ray.remote
def compute_precip_summaries(storm_da, cell_areas, agg_func, data_doi, gatekeeper=None):
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
        data_doi (str): the doi of the MERRA-2 precip dataset

    Outputs:
        summaries (list): the summary quantities in the order they appear in func_vars_dict
    '''
    
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    augmented_da = augment_storm_da(storm_da)
    
    granule_lst = grab_MERRA2_granules(augmented_da, data_doi)

    # handle unstreamable days (as of 11/30, misc. days were dropped from at least one MERRA-2 dataset)
    augmented_days = np.unique(augmented_da.time.dt.strftime("%Y%m%d"))
    if len(granule_lst) == len(augmented_days):
        missing_days = []
    else:
        granule_days = []
        for granule in granule_lst:
            granule_days.append(granule.data_links()[0].split('.')[-2])
        granule_days = np.array(granule_days)
        # find missing granule days
        missing_days = np.setdiff1d(augmented_days, granule_days)
        # exclude those missing days from the storm dataarray mask

        if len(granule_lst) == 0:
            return [np.nan]*2, missing_days
        
        augmented_da = augmented_da.sel(time=~augmented_da.time.dt.strftime("%Y%m%d").isin(missing_days))
    
    granule_pointers = ray.get(gatekeeper.get_granule_pointers.remote(granule_lst))
    
    var_lst = ['PRECLS', 'PRECCU', 'PRECSN']

    obs_ds = xr.open_mfdataset(granule_pointers)
    obs_ds = obs_ds.assign_coords(time=obs_ds.time - np.timedelta64(30, 'm'))
    # get cumulative snowfall for each 3 hour block
    obs_ds = (obs_ds[var_lst]*60*60).resample(time='3h').sum()
    obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))
    
    summaries = []
    # compute aggregate rainfall
    obs_ds['tot_rainfall'] = obs_ds['PRECCU'] + obs_ds['PRECLS']
    summaries.append(agg_func(augmented_da, obs_ds['tot_rainfall'], cell_areas))
    # compute aggregate snowfall
    summaries.append(agg_func(augmented_da, obs_ds['PRECSN'], cell_areas))
    
    return summaries, missing_days

@ray.remote
def compute_precip_chunk_summaries(chunk_lst, cell_areas, agg_func, data_doi, gatekeeper=None):
    '''
    The analogous function to compute_chunk_summaries(), but for the precip computations.

    Inputs:
        chunk_lst (pd.DataFrame): a dataframe with a data_array column
        cell_areas (xarray.DataArray): the DataArray with the cell areas
        agg_func (function): the function to aggregate over the spatiotemporal footprint of the AR,
            usually cumulative
        data_doi (str): the doi of the MERRA-2 precip dataset

    Outputs:
        summaries_lst (list of lists): the summary quantities in the order they appear in func_vars_dict,
            one list per storm
    '''

    summaries_lst = []
    missing_lst = []
    for index, storm in chunk_lst.iterrows():
        summaries, missing_days = compute_precip_summaries(storm.data_array, 
                                             cell_areas, 
                                             agg_func, 
                                             data_doi, 
                                             gatekeeper)
        summaries_lst.append(summaries)
        missing_lst.append(missing_days)

    return summaries_lst, missing_lst
        
    
    
