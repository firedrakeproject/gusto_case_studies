"""
Plots the depth contours overlaid with velocity quivers for the shallow water
simulation initialised from ERA5 data.
"""
from os.path import abspath, dirname
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, plot_field_quivers, tomplot_field_title,
    extract_gusto_coords, extract_gusto_field, regrid_horizontal_slice,
)

test = 'era5'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../figures/{test}'

# ---------------------------------------------------------------------------- #
# Plot details
# ---------------------------------------------------------------------------- #
time_idxs = [0, 4, 8, 12, 16, -1]
field_name = 'D'

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
field_lines = np.linspace(5350, 6200, 18)
field_contours = np.linspace(5350, 6200, 35)
field_colour_scheme = 'YlOrBr'
field_cbar_label = r'$D$ (m)'
contour_method = 'tricontour'
xlims = [-180, 180]
ylims = [-90, 90]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
fig = plt.figure(figsize=(15, 12))
projection = ccrs.PlateCarree()

for i, time_idx in enumerate(time_idxs):

    ax = fig.add_subplot(2, 3, 1+i, projection=projection)

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx] / (24.*60.*60.)

    # Select options for each field --------------------------------------------
    contours = field_contours
    colour_scheme = field_colour_scheme
    field_label = field_cbar_label
    cmap, _ = tomplot_cmap(
        contours, colour_scheme, cmap_rescale_type='top',
    )
    _, lines = tomplot_cmap(
        field_lines, colour_scheme
    )

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines, plot_contour_lines=False
    )

    add_colorbar_ax(ax, cf, field_label, cbar_format='%4.0f', location='bottom', cbar_labelpad=-10, pad=0.1)

    tomplot_field_title(ax, f'{time:.1f} days', minmax=True, field_data=field_data)

    # Add quivers --------------------------------------------------------------
    # Need to re-grid to lat-lon grid to get sensible looking quivers
    zonal_data = extract_gusto_field(data_file, 'u_zonal', time_idx=time_idx)
    meridional_data = extract_gusto_field(data_file, 'u_meridional', time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, 'u_zonal')

    lon_1d = np.linspace(-180.0, 180.0, 91)
    lat_1d = np.linspace(-90.0, 90.0, 81)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')
    regrid_zonal_data = regrid_horizontal_slice(
        lon_2d, lat_2d, coords_X, coords_Y, zonal_data,
        periodic_fix='sphere'
    )
    regrid_meridional_data = regrid_horizontal_slice(
        lon_2d, lat_2d, coords_X, coords_Y, meridional_data,
        periodic_fix='sphere'
    )
    plot_field_quivers(
        ax, lon_2d, lat_2d, regrid_zonal_data, regrid_meridional_data,
        spatial_filter_step=(2, 2), magnitude_filter=1.0, scale=1.0
    )

    # Labels -------------------------------------------------------------------
    if i in [0, 3]:
        ax.set_ylabel(r'$\vartheta$ (deg)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    if i in [3, 4, 5]:
        ax.set_xlabel(r'$\lambda$ (deg)', labelpad=-10)
        ax.set_xlim(xlims)
        ax.set_xticks(xlims)
        ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(wspace=0.15, hspace=-0.35)
plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
