'''
Module with functions to format the various AR data products generated in the course of this project.

Jimmy Butler
October 2025
'''

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

def relabel_storms(catalog, date_colname):
    '''
    Function to give each storm a label of the format 'YYYY_n', where n is the nth storm
        in the supplied dataframe from that year. Used to create prettier labels from clustering
        algorithm output, which by default is just the (n-1)th storm identified from 1/1980.

    Inputs:
        catalog (pandas.DataFrame): contains a column called 'data_array' with binary dataarrays of each storm,
            and a column with the start date of that storm (in pd.Datetime format)
        date_colname (str): the name of the column with the start date of each storm

    Outputs:
        relabelled_catalog (pandas.DataFrame): the catalog, but with the new index
    '''

    # preprocess labels
    labels_lst = []
    for year in landfalling_storms.start_date.dt.year.unique():
    
        n_storms = (landfalling_storms.start_date.dt.year == year).sum()
        year_labels = f'{year}_' + (np.arange(0, n_storms) + 1).astype(str)
        
        labels_lst = labels_lst + list(year_labels)
    
    landfalling_storms.index = labels_lst
    landfalling_storms.index.name = 'Label'

def to_stormtime_format(catalog, show_progress=False):
    '''
    Helper function which takes in the default catalog format 
        (a DataFrame whose rows contain xArray DataArray masks for each storm)
        and converts to a pandas DataFrame where rows consist of a storm at a particular time and columns 
        contain point locations associated with these storms. This data format facilitates making animations of ARs.

    Inputs
        catalog (pandas.DataFrame): contains a column called 'data_array' with binary dataarrays of each storm
        show_progress (bool): Optional; if True, displays a progress bar for the outer loop. Defaults to False.

    Outputs
        stormtime_df (pandas.DataFrame): DataFrame where a row consists of a particular AR at 
            a particular time, and columns give lists of associated coordinates as well as time
    '''
    # lists to collect storm labels, times, and corresponding storm lats and lons
    labels = []
    times = []
    lats = []
    lons = []

    # Determine whether to wrap the iterator with tqdm based on show_progress
    loop_iterable = tqdm(catalog.index, desc="Processing storms") if show_progress else catalog.index

    # for every storm
    for index in loop_iterable:

        # grab that storm's times and binary mask
        storm_da = catalog.loc[index].data_array
        storm_times, storm_grid_lats, storm_grid_lons = storm_da.coords.values()
        
        # for each time step of that storm
        for i in range(len(storm_times)):
            # get lats and lons associated with that storm at that time
            time_slice = storm_da.sel(time = storm_times[i])
            inds = np.argwhere(time_slice.to_numpy() == 1)
            storm_lats = storm_grid_lats[inds[:,0]].values
            storm_lons = storm_grid_lons[inds[:,1]].values
            
            # add the storm label, time, and lats and lons at that time to requisite lists
            labels.append(index)
            times.append(storm_times[i].values)
            lats.append(storm_lats)
            lons.append(storm_lons)

    stormtime_df = pd.DataFrame({'label':labels, 'time':times, 'lat':lats, 'lon':lons})

    return stormtime_df

def construct_da(cluster_df):
    '''
    Given a pandas.DataFrame of a particular AR in stormtime format (mentioned above),
        converts it to a binary valued xarray.DataArray whose dimensional extents
        constitute the smallest data cube that contains all AR points.
        Helper function for construct_da_series, which applies this function to a series
        of dataframes.

    Inputs:
        cluster_df (pandas.DataFrame): the particular AR in stormtime format

    Outputs:
        storm_da (xarray.DataArray): the storm in binary xArray mask format
    '''

    n_time_pts = cluster_df.shape[0]
    dfs = [None]*n_time_pts

    # grab all of the coordinates of the AR footprint and times
    for i in range(n_time_pts):
    
        lats = cluster_df.iloc[i].lats
        lons = cluster_df.iloc[i].lons
        storm_df = pd.DataFrame({'lat': lats, 'lon': lons})
        storm_df['time'] = cluster_df.iloc[i].time
    
        dfs[i] = storm_df
    
    storm_df = pd.concat(dfs, axis=0)
    storm_df['clust'] = 1

    # convert time, lat, lon to multi-index
    storm_df = storm_df.set_index(['time', 'lat', 'lon'])
    storm_da = storm_df.to_xarray()
    # np.int8 chosen to reduce the size of the dataarray
    storm_da = storm_da.fillna(0).astype(np.int8).clust
    
    return storm_da

def construct_da_series(stormtime_df):
    '''
    Function which applies construct_da to each storm in a big dataframe of storms in stormtime format.

    Inputs:
        stormtime_df (pandas.DataFrame): the dataframe of storms in stormtime format, all concatenated together

    Outputs:
        data_arrays (pandas.Series): the series of xarray.DataArrays
    '''

    # construct dataarrays for each storm, organize into series
    data_arrays = stormtime_df.groupby('cluster').apply(construct_da, include_groups=False)
    data_arrays.name = 'data_array'

    return data_arrays

def from_stormtime_format(stormtime_df, coord_dict):
    '''
    Function to convert a DataFrame in stormtime_df into a binary valued xarray.DataArray.
        This is meant to be used on several ARs at once, particularly to convert a record of
        all ARs in stormtime format into a format comparable with the Wille 2022 catalogs.

    Inputs:
        stormtime_df (pandas.DataFrame): the catalog in stormtime format

    Outputs:
        coord_dict (dictionary): a dictionary whose keys are 'lats' and 'lons', and provide
            all of the latitudes and longitudes you would like to include in your data cube.
    '''


    # also convert to one-hot encoded raster format like Jonathan's catalogs
    times = np.sort(stormtime_df.time.unique())
    da_lst = [None]*len(times)
    for i in range(len(times)):
        single_df = stormtime_df[stormtime_df.time == times[i]]
        n_storms = single_df.shape[0]
        storm_df = [None]*n_storms

        for j in range(single_df.shape[0]):
            lats = pd.Series(single_df[['cluster', 'lats', 'lons']].lats.iloc[j])
            lons = pd.Series(single_df[['cluster', 'lats', 'lons']].lons.iloc[j])

            points = pd.DataFrame({'lat':lats, 'lon':lons})
            points['cluster'] = single_df['cluster'].iloc[j]
            storm_df[j] = points
	    
        time_df = pd.concat(storm_df, axis=0)
        raster_day = time_df.set_index(['lat', 'lon']).to_xarray()
        da_lst[i] = raster_day

    da = xr.concat(da_lst, dim='time')
    da = da.assign_coords(time = times)

    # fill in times for which there was no AR originally
    years = stormtime_df.time.dt.year.unique()
    augmented_times_tot = [None]*len(years)
    for i, year in enumerate(years):
        augmented_times_tot[i] = np.array(pd.date_range(f'{year}-01-01T00:00:00.000000000', f'{year}-12-31T21:00:00.000000000', freq='3h'))
    
    augmented_times = np.concatenate(augmented_times_tot)

    da = da.reindex(lat=coord_dict['lats'], lon=coord_dict['lons'], time=augmented_times)

    da = da.fillna(0)
    da = da.cluster.astype(np.int16)
    da = da.chunk('auto')

    return da