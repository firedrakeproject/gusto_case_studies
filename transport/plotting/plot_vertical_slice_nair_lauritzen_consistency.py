"""
Plots the vertical slice Nair-Lauritzen test case, in the convergence
configuration.

This plots:
(a) rho @ t = 0 s, (b) rho @ t = 1000 s, (c) rho @ t = 2000 s
(d) m_X @ t = 0 s, (e) m_X @ t = 1000 s, (f) rho @ t = 2000 s
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_fig, plot_field_quivers, tomplot_field_title,
    extract_gusto_coords, extract_gusto_field,
    regrid_horizontal_slice
)

test = 'vertical_slice_nair_lauritzen_consistency'

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
field_names = ['rho_d', 'rho_d', 'rho_d',
               'm_X', 'm_X', 'm_X']
time_idxs = [0, 'midpoint', -1,
             0, 'midpoint', -1]
cbars = [False, False, True,
         False, False, True]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
rho_contours = np.linspace(0.0, 1.8, 19)
rho_colour_scheme = 'gist_earth_r'
rho_field_label = r'$\rho_d$ (kg m$^{-3}$)'
m_contours = np.linspace(0.01999, 0.02001, 6)
m_colour_scheme = 'RdBu_r'
m_field_label = r'$m_X$ (kg kg$^{-1}$)'
contour_method = 'tricontour'
xlims = [0., 2.]
ylims = [0., 2.]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(2, 3, figsize=(20, 12), sharex='all', sharey='all')
time_idx = 0

for i, (ax, time_idx, field_name, cbar) in \
        enumerate(zip(axarray.flatten(), time_idxs, field_names, cbars)):

    if time_idx == 'midpoint':
        time_idx = int((len(data_file['time'][:]) - 1) / 2)

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    # Select options for each field --------------------------------------------
    if field_name == 'rho_d':
        contours = rho_contours
        colour_scheme = rho_colour_scheme
        field_label = rho_field_label
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        cbar_labelpad = -10
        data_format = '.2f'

    elif field_name == 'm_X':
        contours = m_contours
        colour_scheme = m_colour_scheme
        field_label = m_field_label
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        cbar_labelpad = -45
        data_format = '.3e'

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    if cbar:
        add_colorbar_fig(
            fig, cf, field_label, ax_idxs=[i], location='right',
            cbar_labelpad=cbar_labelpad, cbar_format=data_format
        )
    tomplot_field_title(
        ax, f't = {time:.1f} s', minmax=True, minmax_format=data_format,
        field_data=field_data
    )

    # Add quivers --------------------------------------------------------------
    if field_name == 'rho_d':
        # Need to re-grid to lat-lon grid to get sensible looking quivers
        u_x_data = extract_gusto_field(data_file, 'u_x', time_idx=time_idx)
        u_z_data = extract_gusto_field(data_file, 'u_z', time_idx=time_idx)
        coords_X, coords_Y = extract_gusto_coords(data_file, 'u_x')

        x_1d = np.linspace(0.0, 2.0, 61)
        z_1d = np.linspace(0.0, 2.0, 61)
        x_2d, z_2d = np.meshgrid(x_1d, z_1d, indexing='ij')
        regrid_u_x_data = regrid_horizontal_slice(
            x_2d, z_2d, coords_X, coords_Y, u_x_data
        )
        regrid_u_z_data = regrid_horizontal_slice(
            x_2d, z_2d, coords_X, coords_Y, u_z_data
        )

        plot_field_quivers(
            ax, x_2d, z_2d, regrid_u_x_data, regrid_u_z_data,
            spatial_filter_step=6, spatial_filter_offset=3,
            magnitude_filter=0.01, scale=8.0
        )

    # Labels -------------------------------------------------------------------
    if i in [0, 3]:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    if i in [3, 4, 5]:
        ax.set_xlabel(r'$x$ (km)', labelpad=-10)
        ax.set_xlim(xlims)
        ax.set_xticks(xlims)
        ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(wspace=0.25)
plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
