"""
Plots the non-divergent and divergence Nair-Lauritzen tests, plotting:
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
test = 'non_divergent'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/nair_lauritzen_{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../figures/nair_lauritzen_{test}'

# ---------------------------------------------------------------------------- #
# Plot details
# ---------------------------------------------------------------------------- #
time_idxs = [0, 'midpoint', -1, -1]
field_names = ['D', 'D', 'D', 'D_difference']
colour_bars = [False, False, True, True]
titles = ['field', 'field', 'field', 'difference']

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
field_lines = np.linspace(0.0, 1.2, 13)  # Remove every other line to declutter
field_contours = np.linspace(0.0, 1.2, 25)
field_contour_to_remove = 0.1
field_colour_scheme = 'YlOrBr'
field_cbar_label = r'$D$ (kg kg$^{-1}$)'
diff_contour_to_remove = 0.0
diff_colour_scheme = 'RdBu_r'
diff_cbar_label = r'$D - D_{true}$ (kg kg$^{-1})$'
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
    if field_name == 'D_difference':
        final_data = extract_gusto_field(data_file, 'D', time_idx=time_idx)
        initial_data = extract_gusto_field(data_file, 'D', time_idx=0)
        field_data = final_data - initial_data
        coords_X, coords_Y = extract_gusto_coords(data_file, 'D')

    else:
        field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
        coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx] / (24.*60.*60.)

    # Select options for each field --------------------------------------------
    if field_name == 'D':
        contours = field_contours
        colour_scheme = field_colour_scheme
        field_label = field_cbar_label
        cmap, _ = tomplot_cmap(
            contours, colour_scheme, cmap_rescale_type='top',
            remove_contour=field_contour_to_remove
        )
        _, lines = tomplot_cmap(
            field_lines, colour_scheme, remove_contour=field_contour_to_remove
        )

    else:
        contours = tomplot_contours(field_data)
        colour_scheme = diff_colour_scheme
        field_label = diff_cbar_label
        cmap, lines = tomplot_cmap(
            contours, colour_scheme, remove_contour=diff_contour_to_remove
        )

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    plot_cubed_sphere_panels(ax)

    if colour_bar:
        add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)

    tomplot_field_title(ax, f'{title}, {time:.1f} days', minmax=True, field_data=field_data)

    # Add quivers --------------------------------------------------------------
    if field_name != 'D_difference':
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
            spatial_filter_step=(5, 5), magnitude_filter=1.0, scale=4.0
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
