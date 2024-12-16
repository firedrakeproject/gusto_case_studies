"""
Plots the <my_test_name> test case

This plots:
(a) <field1> @ t = 0 s, (b) <field1> @ t = <final_time> s
(d) <field2> @ t = 0 s, (e) <field2> @ t = <final_time> s
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field, add_colorbar_fig,
    tomplot_field_title, extract_gusto_coords, extract_gusto_field
)

test = 'my_test_name'

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
subplots_x = 2
subplots_y = 2
field_names = ['field1', 'field1',
               'field2', 'field2']
time_idxs = [0, -1,
             0, -1]
cbars = [False, True,
         False, True]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
# First field
f1_contours = np.linspace(0.0, 3.6, 37)
f1_colour_scheme = 'RdBu_r'
f1_contour_to_remove = None
f1_field_label = r'$f_1$ (kg m$^{-3}$)'

# Second field
f2_contours = np.linspace(0.01, 0.07, 13)
f2_colour_scheme = 'Purples'
f2_contour_to_remove = 0.02
f2_field_label = r'$f_2$ (kg kg$^{-1}$)'

# General
contour_method = 'tricontour'
xmax = 2.0
ymax = 2.0
xlims = [0., xmax]
ylims = [0., ymax]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
assert len(field_names) == subplots_x*subplots_y
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(
    subplots_y, subplots_x, figsize=(16, 12), sharex='all', sharey='all'
)

for i, (ax, time_idx, field_name, cbar) in \
        enumerate(zip(axarray.flatten(), time_idxs, field_names, cbars)):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    # Select options for each field --------------------------------------------
    if field_name == 'field1':
        contours = f1_contours
        colour_scheme = f1_colour_scheme
        field_label = f1_field_label
        contour_to_remove = f1_contour_to_remove

    elif field_name == 'field2':
        contours = f2_contours
        colour_scheme = f2_colour_scheme
        field_label = f2_field_label
        contour_to_remove = f2_contour_to_remove

    cmap, lines = tomplot_cmap(
        contours, colour_scheme, remove_contour=contour_to_remove
    )

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    if cbar:
        add_colorbar_fig(
            fig, cf, field_label, ax_idxs=[i], location='right'
        )
    tomplot_field_title(
        ax, f't = {time:.1f} s', minmax=True, field_data=field_data
    )

    # Labels -------------------------------------------------------------------
    if i % subplots_x == 0:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    if i > (subplots_y - 1)*subplots_x - 1:
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
