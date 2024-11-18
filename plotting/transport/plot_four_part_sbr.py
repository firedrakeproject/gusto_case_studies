"""
Plots the transported vector field from the four-part SBR test, plotting:
(a) the initial transported field, (b) the field at the midpoint,
(c) the final transported field, (d) the difference between initial and final.
"""
from os.path import abspath, dirname
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_contours, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, plot_field_quivers, tomplot_field_title,
    extract_gusto_coords, extract_gusto_field, regrid_horizontal_slice,
    plot_cubed_sphere_panels
)

# Whether to plot 'divergent' or 'non_divergent' case
# Only the directory is changed dependent on the test
test = 'four_part_sbr_vorticity'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/{test}'

# ---------------------------------------------------------------------------- #
# Plot details
# ---------------------------------------------------------------------------- #
time_idxs = [0, 'midpoint', -1, -1]
field_names = ['F', 'F', 'F', 'F_difference']
colour_bars = [False, False, True, True]
titles = ['field', 'field', 'field', 'difference']

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
field_contours = np.linspace(0.0, 3.0, 11)
field_colour_scheme = 'YlOrBr'
field_cbar_label = r'$|F|$ (m s$^{-1}$)'
diff_colour_scheme = 'GnBu'
diff_cbar_label = r'$|F - F_{true}|$ (m s$^{-1})$'
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

for i, (time_idx, field_name, colour_bar, title) in \
        enumerate(zip(time_idxs, field_names, colour_bars, titles)):

    ax = fig.add_subplot(2, 2, 1+i, projection=projection)
    if time_idx == 'midpoint':
        time_idx = int((len(data_file['time'][:]) - 1) / 2)

    # Data extraction ----------------------------------------------------------
    if field_name == 'F_difference':
        final_zonal_data = extract_gusto_field(data_file, 'F_zonal', time_idx=time_idx)
        initial_zonal_data = extract_gusto_field(data_file, 'F_zonal', time_idx=0)
        zonal_data = final_zonal_data - initial_zonal_data

        final_meridional_data = extract_gusto_field(data_file, 'F_meridional', time_idx=time_idx)
        initial_meridional_data = extract_gusto_field(data_file, 'F_meridional', time_idx=0)
        meridional_data = final_meridional_data - initial_meridional_data

    else:
        zonal_data = extract_gusto_field(data_file, 'F_zonal', time_idx=time_idx)
        meridional_data = extract_gusto_field(data_file, 'F_meridional', time_idx=time_idx)

    coords_X, coords_Y = extract_gusto_coords(data_file, 'F_zonal')
    mag_data = np.sqrt(zonal_data**2 + meridional_data**2)
    time = data_file['time'][time_idx]

    # Select options for each field --------------------------------------------
    if field_name == 'F':
        contours = field_contours
        colour_scheme = field_colour_scheme
        field_label = field_cbar_label
        cmap, lines = tomplot_cmap(contours, colour_scheme, cmap_rescale_type='top')

    else:
        contours = tomplot_contours(mag_data)
        colour_scheme = diff_colour_scheme
        field_label = diff_cbar_label
        cmap, lines = tomplot_cmap(contours, colour_scheme)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, mag_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    plot_cubed_sphere_panels(ax)

    if colour_bar:
        add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)

    tomplot_field_title(ax, f'{title}, {time:.1f} s', minmax=True, field_data=mag_data)

    # Add quivers --------------------------------------------------------------
    # Need to re-grid to lat-lon grid to get sensible looking quivers
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
        spatial_filter_step=(3, 3), magnitude_filter=0.1, scale=0.2
    )

    # Labels -------------------------------------------------------------------
    if i in [0, 2]:
        ax.set_ylabel(r'$\vartheta$ (deg)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    if i in [2, 3]:
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
