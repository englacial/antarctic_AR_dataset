'''
A module with functions to load up any datasets that may be useful in other computations,
including the Wille (2024) catalogs, and the AIS points.

Jimmy Butler
October 2025
'''

from pathlib import Path
import xarray as xr
import os
import numpy as np
import earthaccess
import ray
import pandas as pd
from huggingface_hub import hf_hub_download

def load_wille_catalogs(dir_path, years=None, exclude_empty_times=True):
    '''
    Load up the Wille 2024 catalogs. By default 1980-2022 are loaded up.
        Removes any times for which there are no ARs present, and subsets to
        just the region we are interested in (AIS + Southern Ocean)

    Inputs:
        dir_path (string): path to directory where the catalog netcdf files are stored. Only .nc files stored
            in this directory should be the Wille catalogs.
        years (list): the years you would like to load, if not all of them
        exclude_empty_times (boolean): whether to remove times for which there are no AR pixels present (default: True)
        
    Outputs:
        catalog_subset (xarray.DataArray): the binary valued DataArray from Wille 2024
    '''

    dir_path_obj = Path(dir_path)
    
    if years is not None:
        catalog_paths = []
        for year in years:
            # uses catalog date signature to grab only the requested years
            year_files = list(dir_path_obj.glob(f'*{year}0101-{year}1231.nc'))
            catalog_paths.extend(year_files)
            
        if not catalog_paths:
            raise FileNotFoundError(f"No catalog files found for the years {years} in {dir_path}")
            
    else:
        # fall back to grabbing all .nc files if no years are specified
        catalog_paths = str(dir_path_obj / '*.nc')

    full_catalog = xr.open_mfdataset(catalog_paths)

    # get rid of all non-antarctic points
    catalog_subset = full_catalog.sel(lat=slice(-86, -39)).ar_binary_tag

    if exclude_empty_times:
        # get rid of all time steps for which there is no AR present
        is_ar_time = catalog_subset.any(dim=['lat', 'lon'])
        catalog_subset = catalog_subset.sel(time=is_ar_time)

    # rounding to avoid floating point errors with 0
    catalog_subset = catalog_subset.assign_coords(
        lat=catalog_subset.lat.round(5),
        lon=catalog_subset.lon.round(5)
    )

    return catalog_subset

def load_catalog(fname):
    '''
    Function that downloads a catalog from the HuggingFace repository and loads
        it up into a pandas DataFrame.

    Inputs:
        fname (string): the name of the catalog you wish to upload from HuggingFace

    Outputs:
        catalog (pd.DataFrame): the DataFrame catalog
    '''

    file_path = hf_hub_download(
        repo_id='butlerj/antarctic_AR_catalogs',
        filename=fname,
        repo_type='dataset')

    catalog = pd.read_hdf(file_path)

    return catalog
    

def load_ais(dir_path, points=False):
    '''
    Load up the AIS mask.

    Inputs:
        dir_path (string): path to directory where AIS mask is stored
        points (boolean): if True, gives a list of coordinate cells that correspond to the AIS.
            By default, loads up the binary valued xarray.DataArray mask.
    Outputs:
        Depends on points, as above.
    '''

    dir_path_obj = Path(dir_path)
    # Load up the AIS mask
    mask_path = dir_path_obj/'AIS_Full_basins_Zwally_MERRA2grid_new.nc'
    full_ais_mask = xr.open_dataset(mask_path).Zwallybasins > 0
    # grab only points in the Southern Ocean area
    ais_mask = full_ais_mask.sel(lat=slice(-86, -39))

    # rounding to get rid of any floating point errors
    ais_mask = ais_mask.assign_coords(
        lat=ais_mask.lat.round(5),
        lon=ais_mask.lon.round(5))

    if points:
        # get ais points
        ais_mask_lats = ais_mask.lat[np.where(ais_mask.to_numpy())[0]].to_numpy()
        ais_mask_lons = ais_mask.lon[np.where(ais_mask.to_numpy())[1]].to_numpy()
        ais_pts = set(zip(ais_mask_lats, ais_mask_lons))

        return ais_pts

    return ais_mask

def load_cell_areas(dir_path):
    '''
    Load up the xarray.DataArray with the grid cell areas.

    Inputs:
        dir_path (string): the path to the directory where the cell areas file is stored.
    Outputs:
        cell_areas (xarray.DataArray): the DataArray in our region of interest with the area of
            each grid cell provided
    '''

    dir_path_obj = Path(dir_path)
    areas_path = dir_path_obj/'MERRA2_gridarea.nc'
    cell_areas = xr.open_dataset(areas_path)
    cell_areas = cell_areas.cell_area

    # rounding to avoid floating point errors with 0
    cell_areas = cell_areas.assign_coords(
        lat=cell_areas.lat.round(5),
        lon=cell_areas.lon.round(5))
    

    return cell_areas

def load_elevation(dir_path):
    '''
    Load up the xarray.DataArray with elevation at each grid cell.

    Inputs:
        dir_path (string): the path to the directory where the elevation netcdf file is stored.
    Outputs:
        elevations (xarray.DataArray): the DataArray in our region of interest with elevations at each grid cell.
    '''
    dir_path_obj = Path(dir_path)
    elevations_path = dir_path_obj/'Elevation_MERRA2.nc'
    elevations = xr.open_dataset(elevations_path)
    elevations = elevations.PHIS

    # rounding to avoid floating point errors with 0
    elevations = elevations.assign_coords(
        lat=elevations.lat.round(5),
        lon=elevations.lon.round(5))

    return elevations

def grab_MERRA2_files(storm_da, ticker):
    '''
    Grab a list of the MERRA-2 files needed to mask a particular storm.

    Inputs:
        storm_da (xarray.DataArray): the AR's binary mask
        ticker (string): the desired dataset's ID

    Outputs:
        fnames (list): list of the MERRA-2 file names
    '''
    
    dates = np.unique(storm_da.time.dt.date.values)

    fnames = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        fname = ticker + '.' + date_str + '.nc4.nc4'
        fnames.append(fname)

    return fnames

def grab_MERRA2_granules(storm_da, data_doi):
    '''
    Grab a list of data granules from a specific MERRA-2 dataset for an AR,
        specifically pointers to granules stored in Amazon S3 bucket.

    Inputs:
        storm_da (xarray.DataArray) the AR's binary mask
        data_doi (str): the doi of the MERRA-2 dataset

    Outputs:
        list of granule pointers
    '''
    first = np.min(storm_da.time.dt.date.to_numpy())
    last = np.max(storm_da.time.dt.date.to_numpy())
    # stream the data only between those two dates
    granule_lst = earthaccess.search_data(doi=data_doi, 
                                  temporal=(f'{first.year}-{first.month}-{first.day}', 
                                            f'{last.year}-{last.month}-{last.day}'))

    return granule_lst

@ray.remote
class EarthdataGatekeeper:
    '''
    A Ray Actor that makes the open requests to NASA's servers sequentially so that we don't get
        rate limited by NASA.
    '''
    def __init__(self):
        self.auth = earthaccess.login()
    
    def get_granule_pointers(self, granule_lst):
        return earthaccess.open(granule_lst, show_progress=False)