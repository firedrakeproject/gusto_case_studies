"""
Plots the evolution of the tropical cyclone
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tomplot import (set_tomplot_style, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     extract_gusto_vertical_slice,
                     tomplot_field_title, regrid_vertical_slice)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/home/thomas/firedrake/src/gusto/results/tropical_cyclone_dt1200_n11_degree1_vector_advective'
plot_dir = results_dir
results_file_name = f'{results_dir}/field_output.nc'
plot_stem = 'tropical_cyclone_vert'

# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
# Specify lists for variables that are different between subplots
field_names = ['Pressure_Vt_raw_pert', 'u_zonal']
field_labels = [r'$p \ / $ Pa', r'$u \ / $ m s$^{-1}$']
colour_schemes = ['Blues_r', 'RdYlBu_r']
all_contours = [np.linspace(-1100, 0, 12), np.linspace(-30, 30, 11)]
levels = [0, 0]
# Things that are the same for all subplots
time_idxs = [-1]
contour_method = 'tricontour'
slice_at = 0.0
slice_along = 'lon'
# Focus domain on an area
xlabel = r'$\phi \ / $ deg'
ylabel = r'$z \ / $ km'
xlims = [0, 20]
ylims = [0, 30]
# 1D grids for vertical regridding
plotting_grid = np.linspace(0, 20, 50)
# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# Loop through time points
# ---------------------------------------------------------------------------- #
for time_idx in time_idxs:

    fig, axarray = plt.subplots(2, 1, figsize=(10, 10), sharex='col')

    # Loop through subplots
    for i, (ax, field_name, field_label, colour_scheme, contours, level) in \
        enumerate(zip(axarray.flatten(), field_names, field_labels,
                    colour_schemes, all_contours, levels)):
        # -------------------------------------------------------------------- #
        # Data extraction
        # -------------------------------------------------------------------- #
        orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
            extract_gusto_vertical_slice(data_file, field_name, time_idx,
                                        slice_along=slice_along, slice_at=slice_at)

        # Slices need regridding as points don't cleanly live along lon or lat = 0.0
        field_data, coords_X, coords_Y = regrid_vertical_slice(plotting_grid,
                                                               slice_along, slice_at,
                                                               orig_coords_X, orig_coords_Y,
                                                               orig_coords_Z, orig_field_data)

        coords_Y /= 1000.

        # -------------------------------------------------------------------- #
        # Plot data
        # -------------------------------------------------------------------- #
        cmap, lines = tomplot_cmap(contours, colour_scheme)
        cf, _ = plot_contoured_field(ax, coords_X, coords_Y, field_data,
                                    contour_method, contours, cmap=cmap,
                                    line_contours=lines)
        add_colorbar_ax(ax, cf, field_label, location='right')
        tomplot_field_title(ax, None, minmax=True, field_data=field_data)

        if i == 1:
            ax.set_xlim(xlims)
            ax.set_xticks(xlims)
            ax.set_xticklabels(xlims)
            ax.set_xlabel(xlabel)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)
        ax.set_ylabel(ylabel)

    # ------------------------------------------------------------------------ #
    # Save figure
    # ------------------------------------------------------------------------ #
    plot_name = f'{plot_dir}/{plot_stem}_{time_idx:02d}.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()


