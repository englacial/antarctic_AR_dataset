# File to run the clustering algorithm on the Wille 2021 catalog, with a particular set of hyperparameters
#
# Jimmy Butler
# September 2025

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.append(str(Path(__file__).parents[3]/'packages'))

from st_dbscan.st_dbscan import ST_DBSCAN
from attribute_utils import is_landfalling
from loading_utils import load_catalogs
from loading_utils import load_ais
from format_utils import construct_da_series

# parsing arguments submitted by user, 
parser = argparse.ArgumentParser(description="Parsing arguments to run the clustering algorithm.")
parser.add_argument("--synoptic_scale_frac", type=float, default=0.5, help="The fraction of a synoptic scale to use for the spatial epsilon.")
parser.add_argument("--eps_time_hours", type=float, default=12, help="The time epsilon to use in hours.")
parser.add_argument("--minpts", type=int, default=5, help="The min number of points to define a core point in DBSCAN.")
parser.add_argument("--n_rep_pts", type=int, default=10, help="The number of points to sample from each cluster in the spatial clustering step.")
parser.add_argument("--seed", type=int, default=12345, help="Random seed, for reproducibility's sake.")
parser.add_argument("--save_path", type=str, required=True, help="The path to the directory where you want to save the results.")

args = parser.parse_args()

# combining into a dictionary of parameters to then loop through
par_dict = {'seed':args.seed,
                     'eps_time':args.eps_time_hours,
                     'eps_space': args.synoptic_scale_frac,
                     'rep_pts': args.n_rep_pts,
                     'min_pts': args.minpts}


catalog_subset = load_catalogs()
ais_pts = load_ais()

# hyperparameters
synoptic_scale = 10**3
km_per_radian = 6.371*(10**3) # arclength (km) on earth subtended by 1 radian

eps_space = par_dict['eps_space']*(synoptic_scale/km_per_radian) # converted to radians for Haversine metric
eps_space_1 = eps_space
eps_space_2 = eps_space
eps_time = par_dict['eps_time']/24
minpts_1 = par_dict['min_pts']
minpts_2 = par_dict['min_pts']
n_rep_pts = par_dict['rep_pts']

# instantiating the clustering object
cluster_obj = ST_DBSCAN(eps_space_1, eps_space_2, eps_time, minpts_1, minpts_2, n_rep_pts, par_dict['seed'])
# doing the spatiotemporal clustering
cluster_infos_df = cluster_obj.fit(catalog_subset)

########### POST-PROCESSING ###########
# remove noise clusters
obj_subset = cluster_infos_df[['cluster', 'lats', 'lons', 'time']]
obj_subset = obj_subset[obj_subset['cluster'] != -1]

storm_df = construct_da_series(obj_subset)
# add whether each storm is landfalling
storm_landfalls = storm_df.apply(is_landfalling)

storm_df = pd.DataFrame({'data_array':storm_df, 'is_landfalling':storm_landfalls})

storm_df.to_hdf(args.save_path + f'/epsspace{par_dict['eps_space']}_epstime{par_dict['eps_time']}_minpts{par_dict['min_pts']}_nreppts{par_dict['rep_pts']}_seed{par_dict['seed']}.h5', key='df')

