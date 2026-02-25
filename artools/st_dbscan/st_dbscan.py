# Implementation of ST_DBSCAN, a version of DBSCAN suited to spatiotemporal clustering problems
# Uses DBSCAN to spatially cluster and reduce data size, then ST-DBSCAN to cluster reduced dataset
# Originally found in "ST-DBSCAN: An algorithm for clustering spatial-temporal data," Birant and Kut, 2007
#
# Code by James Butler
# Date: June 2024

import numpy as np
import pandas as pd
import math
import os
from pathlib import Path
print(os.getcwd())
from .utils import average_angle
from .utils import arctan
from .utils import retrieve_neighbors
from sklearn.cluster import DBSCAN
from tqdm import tqdm

class ST_DBSCAN:
    '''
    Instantiates an object with methods that perform the st-dbscan algorithm on some input data.
    '''
    
    def __init__(self, eps_space_1, eps_space_2, eps_time, minpts_1, minpts_2, n_rep_pts, seed=None):

        self.eps_space_1 = eps_space_1
        self.eps_space_2 = eps_space_2
        self.eps_time = eps_time
        self.minpts_1 = minpts_1
        self.minpts_2 = minpts_2
        self.n_rep_pts = n_rep_pts

        # if the seed has not been specified, using numpys default rng
        if not seed:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
    
    def __repr__(self):

        str_rep = f'ST-DBSCAN object \n Spatial Clustering Hyperparams: \n spatial epsilon: {self.eps_space_1} \
        \n min points: {self.minpts_1} \n Spatiotemporal Clustering Hyperparams: \n spatial epsilon: {self.eps_space_2} \
        \n time epsilon: {self.eps_time} \n min points: {self.minpts_2} \n number of representative points: {self.n_rep_pts}'

        return str_rep

    def fit(self, da):

        times = da.time.to_numpy()
        cluster_infos = [None]*len(times)

        print('Beginning spatial clustering step.')

        for i in tqdm(range(len(times))):

            time_slice = da.sel(time = times[i])
            # find lats/lons of AR points in this time step
            inds = np.argwhere(time_slice.to_numpy() == 1)
            storm_lats = time_slice.lat[inds[:,0]]
            storm_lons = time_slice.lon[inds[:,1]]

            # cluster spatially using DBSCAN, synoptic scale neighborhood size
            clustering = self.__fit_spatial(np.column_stack((np.radians(storm_lats), np.radians(storm_lons))))
            fixed_time_df = pd.DataFrame({'lats': storm_lats, 'lons': storm_lons, 'cluster': clustering})

            # get the average lat-lon of each cluster, according to average_angle function above
            avg_positions = pd.DataFrame(fixed_time_df.groupby('cluster').apply(average_angle, include_groups=False).to_list(), 
                                columns=['cluster', 'mean_lat', 'mean_lon'])

            # randomly sample n_rep_pts-many points (without replacement) from each cluster and store
            # uses the seed that we pass in
            rep_pts = fixed_time_df.groupby('cluster', as_index=False)[['lats', 'lons']].agg(lambda x: list(self.rng.choice(x, min(self.n_rep_pts, len(x)), replace=False)))
            rep_pts.rename(columns={'lats':'rep_lats', 'lons':'rep_lons'}, inplace=True)

            rep_pts_df = pd.merge(rep_pts, avg_positions, on='cluster')

            # aggregate ALL lats and lons for each cluster into lists as well
            cluster_info = fixed_time_df.groupby('cluster', as_index=False)[['lats', 'lons']].agg(list)

            # combine all this info into single dataframe
            cluster_info = pd.merge(cluster_info, rep_pts_df, on='cluster')
            # add time into column
            cluster_info['time'] = times[i]
            # add this time-specific df into list of dfs
            cluster_infos[i] = cluster_info

        # stitch list of dataframes across time into big dataframe
        cluster_infos_df = pd.concat(cluster_infos, axis=0)
        cluster_infos_df.reset_index(drop=True, inplace=True)
        # preallocate empty column for cluster labels for each AR timestep
        ar_pt_df = cluster_infos_df[['mean_lat', 'mean_lon', 'rep_lats', 'rep_lons', 'time']]
        ar_pt_df['cluster'] = np.full(cluster_infos_df.shape[0], np.nan)

        print('Beginning spatiotemporal clustering step.')

        # loop through the dataframe made and unpack all of the representative points sampled for each cluster
        # resulting dataframe will have rows consisting of representative storm point
        # this is basically getting the data in the right format to do the ST-DBSCAN step
        unpacked_df = self.unpack_df(ar_pt_df)

        # add cluster membership column back to original df
        cluster_infos_df['cluster'] = self.__fit_spatiotemporal(unpacked_df)

        return cluster_infos_df 

    def __fit_spatial(self, points):

        clustering = DBSCAN(eps=self.eps_space_1, min_samples=self.minpts_1, metric='haversine').fit_predict(points)

        return clustering

    def unpack_df(self, ar_pt_df, clustered_label=None):
        '''
        A helper method to unpack a DataFrame whose rows are spatial AR clusters at a point in time. Unpacks
        the lists of assoicated points into a larger dataframe where each row consists of an AR pixel at a particular
        lat, lon, and time. Also stored is a label tagging which points where part of spatial clusters together.

        Inputs:
            ar_pt_df (pandas.DataFrame):  a DataFrame in the format described above
            clustered_label (String): string label for a column of cluster labels in ar_pt_df. If provided,
                unpacking the ar_pt_df will also include a column of the cluster labels for each point after
                spatiotemporal clustering. By default, this method behaves as a helper method preparing the data 
                for ST-DBSCAN, so the column will just contain NaNs that will later be replaced with labels upon
                spatiotemporal clustering.
        '''

        unpacked_indices = []
        unpacked_lats = []
        unpacked_lons = []
        unpacked_times = []

        for index in list(ar_pt_df.index):
            num_pts = len(ar_pt_df.loc[index].rep_lats) + 1
            unpacked_indices = unpacked_indices + [index]*num_pts
            unpacked_times = unpacked_times + [ar_pt_df.loc[index].time]*num_pts
            unpacked_lats = unpacked_lats + list(np.radians(ar_pt_df.loc[index].rep_lats)) + [np.radians(ar_pt_df.loc[index].mean_lat)]
            unpacked_lons = unpacked_lons + list(np.radians(ar_pt_df.loc[index].rep_lons)) + [np.radians(ar_pt_df.loc[index].mean_lon)]

        if clustered_label:
            cluster_labs = ar_pt_df[clustered_label]
            unpacked_cluster_labs = []
            
            for index in list(ar_pt_df.index):
                num_pts = len(ar_pt_df.loc[index].rep_lats) + 1
                unpacked_cluster_labs = unpacked_cluster_labs + [cluster_labs.loc[index]]*num_pts


            unpacked_df = pd.DataFrame({'cluster':unpacked_cluster_labs, 'space_cluster':unpacked_indices,'time':unpacked_times, 'lat':unpacked_lats, 'lon':unpacked_lons})

        else:
            unpacked_df = pd.DataFrame({'cluster':np.full(len(unpacked_indices), np.nan), 'space_cluster':unpacked_indices,'time':unpacked_times, 'lat':unpacked_lats, 'lon':unpacked_lons})

        return unpacked_df

    def __fit_spatiotemporal(self, point_df):

        cluster_label = 0
        noise_label = -1
        
        # for each point
        for i in range(point_df.shape[0]):
            cur_obj = point_df.iloc[[i]]
            # if either unclustered or noise
            if math.isnan(point_df.loc[i, 'cluster']) or point_df.loc[i, 'cluster'] == noise_label:
        
                neighbors = retrieve_neighbors(cur_obj, point_df, self.eps_space_2, self.eps_time)
                # if less than min_pts neighbors, accounting for the point itself
                if neighbors.shape[0] < self.minpts_2 + 1:
                    point_df.loc[i, 'cluster'] = noise_label
                # otherwise, start new cluster and label your neighbors accordingly
                else:
                    cluster_label = cluster_label + 1
                    point_df.loc[neighbors.index, 'cluster'] = cluster_label
                    # indices to keep track of which points will be in cluster
                    cluster_inds = list(neighbors.drop(i).index)

                    # while we still have unprocessed cluster points
                    while cluster_inds:
                        new_cur_obj = point_df.loc[[cluster_inds.pop()]]
                        new_neighbors = retrieve_neighbors(new_cur_obj, point_df, self.eps_space_2, self.eps_time)
                
                        if new_neighbors.shape[0] >= self.minpts_2 + 1:
                            # if neighboring point unlabelled, endow with cluster label
                            unlabelled = new_neighbors.loc[new_neighbors['cluster'].isnull()]
                            point_df.loc[unlabelled.index, 'cluster'] = cluster_label
                            # add these newly clustered points to list of unprocessed cluster points
                            cluster_inds = cluster_inds + list(unlabelled.index)

        cluster_assignments = point_df.groupby('space_cluster')['cluster'].apply(lambda series: series.value_counts().idxmax())

        return cluster_assignments
