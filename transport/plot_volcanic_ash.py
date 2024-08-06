"""
Plots the volcanic ash dispersion test case.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from matplotlib.colors import ListedColormap
from tomplot import (set_tomplot_style, tomplot_contours,
                     plot_contoured_field,
                     tomplot_field_title, extract_gusto_coords,
                     extract_gusto_field, plot_field_quivers)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/home/thomas/firedrake/src/gusto/case_studies/results/volcanic_ash/'
plot_dir = '/home/thomas/firedrake/src/gusto/case_studies/figures'
results_file_name = f'{results_dir}/field_output.nc'
plot_stem = f'{plot_dir}/volcanic_ash'
# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
field_name = 'ash'
colour_scheme = 'gist_heat_r'
field_label = 'Ash / ppm'
contour_method = 'tricontour'
plot_wind = False
wind_name_X, wind_name_Y = 'VelocityX', 'VelocityY'
pole_longitude = 175.0
pole_latitude = 35.0
new_lon_extents = (-30, 30)
new_lat_extents = (-30, 30)
new_lon_xlims = (-35, 35)
new_lat_ylims = (-35, 35)
Lx = 1000.0
# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')
# ---------------------------------------------------------------------------- #
# Work out time idxs and overall min/max
# ---------------------------------------------------------------------------- #
time_idxs = range(len(data_file['time'][:]))
universal_field_data = extract_gusto_field(data_file, field_name)
contours = tomplot_contours(universal_field_data)
projection = ccrs.RotatedPole(pole_longitude=pole_longitude, pole_latitude=pole_latitude)
# ---------------------------------------------------------------------------- #
# Make a modified cmap with transparent colour instead of white
# ---------------------------------------------------------------------------- #
ncolours = len(contours)-1
cmap = plt.cm.get_cmap(colour_scheme, ncolours)
colours = cmap(np.linspace(0, 1, ncolours))
# Set transparency for most colours
for i in range(ncolours):
    colours[i, -1] = 0.75
# Set first colour to transparent
colours[0, -1] = 0.0
cmap = ListedColormap(colours)
# ---------------------------------------------------------------------------- #
# Loop through points in time
# ---------------------------------------------------------------------------- #
for time_idx in time_idxs:
    # ------------------------------------------------------------------------ #
    # Data extraction
    # ------------------------------------------------------------------------ #
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx] / (24*60*60)
    # ------------------------------------------------------------------------ #
    # Extract wind data
    # ------------------------------------------------------------------------ #
    if plot_wind:
        wind_data_X = extract_gusto_field(data_file, wind_name_X, time_idx=time_idx)
        wind_data_Y = extract_gusto_field(data_file, wind_name_Y, time_idx=time_idx)
        wind_coords_X, wind_coords_Y = extract_gusto_coords(data_file, wind_name_X)
    # ------------------------------------------------------------------------ #
    # Transform coordinates
    # ------------------------------------------------------------------------ #
    coords_X = (new_lon_extents[0]
                + coords_X * (new_lon_extents[1] - new_lon_extents[0]) / Lx)
    coords_Y = (new_lat_extents[0]
                + coords_Y * (new_lat_extents[1] - new_lat_extents[0]) / Lx)
    if plot_wind:
        wind_coords_X = (wind_coords_X - new_lon_extents[0]) * (new_lon_extents[1] - new_lon_extents[0]) / Lx
        wind_coords_Y = (wind_coords_Y - new_lat_extents[0]) * (new_lat_extents[1] - new_lat_extents[0]) / Lx
    # ------------------------------------------------------------------------ #
    # Plot data
    # ------------------------------------------------------------------------ #
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.stock_img()
    ax.coastlines()

    cf, _ = plot_contoured_field(ax, coords_X, coords_Y, field_data, contour_method,
                                 contours, cmap=cmap, plot_contour_lines=False,
                                 projection=None, transparency=None)

    if plot_wind:
        _ = plot_field_quivers(ax, wind_coords_X, wind_coords_Y,
                               wind_data_X, wind_data_Y,
                               spatial_filter_step=5, projection=projection)

    tomplot_field_title(ax, f't = {time:.2f} days', minmax=True, field_data=field_data,
                        minmax_format=".2f")
    ax.set_xlim(new_lon_xlims)
    ax.set_ylim(new_lat_ylims)

    # ------------------------------------------------------------------------ #
    # Manually add colorbar as it is difficult to get it into the correct position
    # ------------------------------------------------------------------------ #
    cbar_format = '{x:.0f}'
    cbar_ticks = [0, 2]
    cbar_ax = fig.add_axes([0.925, 0.11, 0.025, 0.7725])
    cb = fig.colorbar(cf, cax=cbar_ax, format=cbar_format, ticks=cbar_ticks,
                      orientation='vertical', ticklocation='right')
    cb.set_label(field_label, labelpad=-5)

    # ------------------------------------------------------------------------ #
    # Save figure
    # ------------------------------------------------------------------------ #
    plot_name = f'{plot_stem}_{time_idx:02d}.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()
