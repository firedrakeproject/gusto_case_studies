"""
Plot from the Terminator Toy test case.
We examine the distribution of the dry density (rho_d)
 and species (X and X2) at different times.

This plots:
(a) rho_d @ t = 0 days, (b) rho_d @ t = 518400 s (6 days), (c) rho_d @ t = 1036800 s (12 days, final time)
(d) X @ t = 0 days, (e) X @ t = 518400 s (6 days), (f) X @ t = 1036800 s (12 days, final time)
(g) X2 @ t = 0 days, (h) X2 @ t = 518400 s (6 days), (i) X2 @ t = 1036800 s (12 days, final time)
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field, add_colorbar_fig,
    tomplot_field_title, extract_gusto_coords, extract_gusto_field
)

test = 'terminator_toy'

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
               'X', 'X', 'X',
               'X2', 'X2', 'X2']
time_idxs = [0, 'midpoint', -1,
             0, 'midpoint', -1,
             0, 'midpoint', -1]
cbars = [False, False, True,
         False, False, True,
         False, False, True]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #

rho_contours = np.linspace(0.0, 1.00, 15)
rho_colour_scheme = 'gist_earth_r'
rho_field_label = r'$\rho_d$ (kg m$^{-3}$)'

X_contours = np.linspace(0, 4e-6, 15)
X_colour_scheme = 'Purples'
X_field_label = r'$X$ (kg kg$^{-1}$)'

X2_contours_first = np.linspace(0.0, 2.000000000001e-6, 15)
X2_contours = np.linspace(0.0, 2.0e-6, 15)
X2_colour_scheme = 'Greens'
X2_field_label = r'$X2$ (kg kg$^{-1}$)'

contour_method = 'tricontour'
xlims = [-180, 180]
ylims = [-90, 90]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(3, 3, figsize=(20, 12), sharex='all', sharey='all')

for i, (ax, time_idx, field_name, cbar) in \
        enumerate(zip(axarray.flatten(), time_idxs, field_names, cbars)):

    if time_idx == 'midpoint':
        time_idx = int((len(data_file['time'][:]) - 1) / 2)

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    # Quote time in days:
    time = data_file['time'][time_idx] / (24.*60.*60.)

    # Select options for each field --------------------------------------------
    if field_name == 'rho_d':
        contours = rho_contours
        colour_scheme = rho_colour_scheme
        field_label = rho_field_label
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        cbar_labelpad = -80
        data_format = '.2e'

    elif field_name == 'X':
        contours = X_contours
        colour_scheme = X_colour_scheme
        field_label = X_field_label
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        cbar_labelpad = -80
        data_format = '.2e'

    elif field_name == 'X2':
        if time_idx == 0:
            contours = X2_contours_first
        else:
            contours = X2_contours
        colour_scheme = X2_colour_scheme
        field_label = X2_field_label
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=None)
        cbar_labelpad = -80
        data_format = '.2e'

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

    # Labels -------------------------------------------------------------------
    if i in [0, 3, 6]:
        ax.set_ylabel(r'$\vartheta$ (deg)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    if i in [6, 7, 8]:
        ax.set_xlabel(r'$\lambda$ (deg)', labelpad=-10)
        ax.set_xlim(xlims)
        ax.set_xticks(xlims)
        ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.subplots_adjust(wspace=0.25)
plot_name = f'{plot_stem}.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
