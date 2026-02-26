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
from tqdm import tqdm

from io import BytesIO
import base64

def construct_thumbnail(storm):

    rep_time = storm.sel(time=storm.sum(dim=['lat', 'lon']).idxmax())
    fig, ax = plt.subplots(1)
    ax.imshow(rep_time.to_numpy());
    ax.invert_yaxis()
    ax.axis('off');
    fig.set_size_inches((0.5,0.5))
    plt.close()

    # following strategy taken from the followign stackexchange
    # https://stackoverflow.com/questions/47038538/insert-matplotlib-images-into-a-pandas-dataframe
    figfile = BytesIO()
    fig.savefig(figfile, format='png', pad_inches=0, bbox_inches='tight')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode()
    imgstr = '<img src="data:image/png;base64,{}" />'.format(figdata_png)

    return imgstr


def display_catalog(catalog_df, nrows=None):
    if nrows:
        return catalog_df.head(nrows).style.format({'data_array': lambda x: construct_thumbnail(x)}).format_index(precision=0)
    else:
        return catalog_df.style.format({'data_array': construct_thumbnail}).format_index(precision=0)

def make_movie(stormtime_df, title, save_path):
    '''
    Function that constructs an animation of ARs in a stormtime version of the AR catalog
    and saves it to a specified path with a progress bar.

    Inputs
        stormtime_df (pd.DataFrame): same as above
        title (string): the title to print on the animation
        save_path (string): the file path to save the animation (e.g., 'movie.mp4' or 'movie.gif')
    '''

    # define the times of the movie, and mappings between AR labels and colors
    movie_times = pd.date_range(start=stormtime_df.time.min(), end=stormtime_df.time.max(), freq='3h')
    unique_clusters = stormtime_df['label'].unique()
    
    # Grab the colormap object instead of the pyplot convenience function
    prism_cmap = plt.get_cmap('prism')
    color_mapping = {unique_clusters[j]: prism_cmap(j/len(unique_clusters)) for j in range(len(unique_clusters))}

    # plot the jth frame of the movie (need as input to matplotlib animation constructor)
    def plt_time(j):
        time_pt = movie_times[j]

        if (time_pt == stormtime_df.time).any():
            dat = stormtime_df[stormtime_df['time'] == time_pt]
            n_clusts = dat.shape[0]

            for i in range(n_clusts):
                cluster = dat['label'].iloc[i]
                ax.scatter(dat['lon'].iloc[i], dat['lat'].iloc[i], transform=ccrs.PlateCarree(), s=1, color=color_mapping[cluster], label=str(cluster), zorder=30)
        
            ax.legend()

        ax.set_extent([-180,180,-90,-39], ccrs.PlateCarree())
        ice_shelf_poly = cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m',edgecolor='none',facecolor='lightcyan') # 10m, 50m, 110m
        ax.add_feature(ice_shelf_poly,linewidth=3)
        ice_shelf_line = cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_lines', '50m',edgecolor='black',facecolor='none') # 10m, 50m, 110m
        ax.add_feature(ice_shelf_line,linewidth=1,zorder=13)
        ax.coastlines(resolution='110m',linewidth=1,zorder=32)

    
        # Map extent 
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(alpha=0.5, zorder=33)
    
        time_ts = pd.Timestamp(time_pt)
        ax.set_title(f'{time_ts.month}/{time_ts.day}/{time_ts.year}, {time_ts.hour}:00')

    # instantiate the animation
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.)))
    plt_time(0)
    fig.suptitle(title, fontsize=16)

    def update_img(i):
        ax.clear()
        plt_time(i)

    ani = animation.FuncAnimation(fig, update_img, frames=len(movie_times))

    # Save the animation with a tqdm progress bar
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
    
    # close the figure so it doesn't accidentally display a static plot output in a Jupyter Notebook
    plt.close(fig)

    return ani