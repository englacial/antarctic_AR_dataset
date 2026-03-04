import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.path as mpath
import matplotlib.colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

from matplotlib import animation
import matplotlib
from matplotlib.cm import prism
from tqdm.auto import tqdm

from io import BytesIO
import base64

def construct_thumbnail(storm):
    '''
    Make a thumbnail image for a particular AR stor.

    Inputs:
        storm (xarray.DataArray): a binary-valued dataarray indicating pixels associated with this storm

    Outputs:
        imgstr (string): an html image tag for the thumbnail
    '''
    rep_time = storm.sel(time=storm.sum(dim=['lat', 'lon']).idxmax())
    fig, ax = plt.subplots(1)
    ax.imshow(rep_time.to_numpy())
    ax.invert_yaxis()
    ax.axis('off')
    fig.set_size_inches((0.5,0.5))
    plt.close()

    # following strategy taken from the following stackexchange
    # https://stackoverflow.com/questions/47038538/insert-matplotlib-images-into-a-pandas-dataframe
    figfile = BytesIO()
    fig.savefig(figfile, format='png', pad_inches=0, bbox_inches='tight')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode()
    imgstr = '<img src="data:image/png;base64,{}" />'.format(figdata_png)

    return imgstr


def display_catalog(catalog_df, nrows=None):
    '''
    Render the catalog as a standard pd.DataFrame, but show the data_array column as a column of thumbnails.
        By default, pandas seeks to render a string representation of each DataFrame. This prevents this messy
        output from being shown. Acts like df.head(nrows) in pandas.

    Inputs:
        catalog_df (pd.DataFrame): the catalog
        nrows (int): the number of rows you wish to show. If None, display the whole catalog.

    Outputs:
        A display of the catalog.
    '''
    if nrows:
        return catalog_df.head(nrows).style.format({'data_array': lambda x: construct_thumbnail(x)}).format_index(precision=0)
    else:
        return catalog_df.style.format({'data_array': construct_thumbnail}).format_index(precision=0)


def format_polar_axis(ax):
    '''
    Helper function with all of the routines to generate a stereographic projection of Antarctica.

    Inputs:
        ax (matplotlib.axes.Axes): the axis on which you wish to plot the map of Antarctica
    '''
    ax.set_extent([-180, 180, -90, -39], ccrs.PlateCarree())
    ice_shelf_poly = cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='none', facecolor='lightcyan')
    ax.add_feature(ice_shelf_poly, linewidth=3)
    ice_shelf_line = cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_lines', '50m', edgecolor='black', facecolor='none')
    ax.add_feature(ice_shelf_line, linewidth=1, zorder=13)
    ax.coastlines(resolution='110m', linewidth=1, zorder=32)

    # Map extent 
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.gridlines(alpha=0.5, zorder=33)


def plot_stormtime_frame(ax, time_pt, stormtime_df, color_mapping=None):
    '''
    Plot all labelled ARs present at a particular time.

    Inputs:
        ax (matplotlib.axes.Axes): the axis on which to plot the ARs at a particular time
        time_pt (pd.DatetimeIndex): the time
        stormtime_df (pd.DataFrame): a dataframe in stormtime format (each row is an AR at a particular time)
        color_mapping (dict): a dictionary mapping AR labels to colors
    '''
    # auto-generate color mapping if not provided (useful for standalone plotting)
    if color_mapping is None:
        unique_clusters = stormtime_df['label'].unique()
        prism_cmap = plt.get_cmap('prism')
        color_mapping = {unique_clusters[j]: prism_cmap(j/len(unique_clusters)) for j in range(len(unique_clusters))}

    if (time_pt == stormtime_df.time).any():
        dat = stormtime_df[stormtime_df['time'] == time_pt]
        n_clusts = dat.shape[0]

        for i in range(n_clusts):
            cluster = dat['label'].iloc[i]
            ax.scatter(dat['lon'].iloc[i], dat['lat'].iloc[i], transform=ccrs.PlateCarree(), 
                       s=1, color=color_mapping[cluster], label=str(cluster), zorder=30)
        
        # deduplicate the legend so you don't get a key for every single point
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

    format_polar_axis(ax)
    time_ts = pd.Timestamp(time_pt)
    ax.set_title(f'{time_ts.month}/{time_ts.day}/{time_ts.year}, {time_ts.hour}:00')


def plot_eulerian_frame(ax, time_pt, data_array):
    '''
    Plot unlabelled AR pixels at a particular point in time. Used to plot
        AR pixels from the Wille et al. 2021 vIVT threshold catalog.

    Inputs: 
        ax (matplotlib.axes.Axes): the axis object to generate the plot on
        time_pt (np.datetime64): the time to plot
        data_array (xarray.DataArray): the binary valued DataArray
    '''
    dat = data_array.sel(time=time_pt)
    
    # mask out zeros so they remain transparent over the map
    dat_masked = dat.where(dat > 0)
    
    # plot the binary values
    ax.pcolormesh(dat.lon, dat.lat, dat_masked, transform=ccrs.PlateCarree(), 
                  cmap='Blues', vmin=0, vmax=1, zorder=30)
    
    format_polar_axis(ax)
    
    # ensure time formatting works smoothly with numpy.datetime64 from xarray
    time_val = time_pt.values if hasattr(time_pt, 'values') else time_pt
    time_ts = pd.Timestamp(time_val)
    ax.set_title(f'{time_ts.month}/{time_ts.day}/{time_ts.year}, {time_ts.hour}:00')


def make_movie(stormtime_df, title, save_path):
    '''
    Create and save an animation showing labelled ARs from the storm catalog.

    Inputs:
        stormtime_df (pd.DataFrame): the DataFrame of storms to plot, in stormtime format
        title (string): the title to plot on the figure
        save_path (string): where to save the figure, along with the filename. Extension must be mp4 or gif.

    Outputs:
        ani (matplotlib.animation.FuncAnimation): the animation object
    '''
    movie_times = pd.date_range(start=stormtime_df.time.min(), end=stormtime_df.time.max(), freq='3h')
    unique_clusters = stormtime_df['label'].unique()
    
    prism_cmap = plt.get_cmap('prism')
    color_mapping = {unique_clusters[j]: prism_cmap(j/len(unique_clusters)) for j in range(len(unique_clusters))}

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.)))
    plot_stormtime_frame(ax, movie_times[0], stormtime_df, color_mapping)
    fig.suptitle(title, fontsize=16)

    def update_img(i):
        ax.clear()
        plot_stormtime_frame(ax, movie_times[i], stormtime_df, color_mapping)

    ani = animation.FuncAnimation(fig, update_img, frames=len(movie_times))
    print(f"Saving animation to {save_path}...")
    filetype = save_path.split('.')[-1]

    if filetype == 'mp4':  
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, writer='ffmpeg', progress_callback=lambda i, n: pbar.update(1))
    elif filetype == 'gif':
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, progress_callback=lambda i, n: pbar.update(1))
    else:
        raise Exception('Unsupported file extension for animation.')
    
    plt.close(fig)
    return ani


def make_eulerian_movie(data_array, title, save_path):
    '''
    Create and save an animation showing the AR pixels from an eulerian pixelwise threshold catalog.

    Inputs:
        data_array (xarray.DataArray): the binary valued dataarray indicating AR pixels
        title (string): the title to show over the movie
        save_path (string): where to save the movie, along with the filename. Only mp4 and gif supported.
    '''
    movie_times = data_array.time.values
    
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.)))
    plot_eulerian_frame(ax, movie_times[0], data_array)
    fig.suptitle(title, fontsize=16)

    def update_img(i):
        ax.clear()
        plot_eulerian_frame(ax, movie_times[i], data_array)

    ani = animation.FuncAnimation(fig, update_img, frames=len(movie_times))
    print(f"Saving animation to {save_path}...")
    filetype = save_path.split('.')[-1]

    if filetype == 'mp4':  
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, writer='ffmpeg', progress_callback=lambda i, n: pbar.update(1))
    elif filetype == 'gif':
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, progress_callback=lambda i, n: pbar.update(1))
    else:
        raise Exception('Unsupported file extension for animation.')
    
    plt.close(fig)
    return ani