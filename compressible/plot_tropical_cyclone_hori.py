"""
Plots the evolution of the tropical cyclone
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tomplot import (set_tomplot_style, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     extract_gusto_field, extract_gusto_coords,
                     tomplot_field_title, reshape_gusto_data,
                     plot_field_quivers)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/data/users/tbendall/results/tropical_cyclone'
plot_dir = results_dir
results_file_name = f'{results_dir}/field_output.nc'
plot_stem = 'tropical_cyclone_hori'

# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
# Specify lists for variables that are different between subplots
field_names = ['Pressure_Vt_perturbation', 'u']
field_labels = [r"$p' \ / $ Pa", r'$|u| \ / $ m s$^{-1}$']
colour_schemes = ['Blues_r', 'RdYlBu_r']
all_contours = [np.linspace(-1100, 0, 12), np.linspace(0, 30, 11)]
levels = [0, 0]
# Things that are the same for all subplots
time_idxs = [-1]
contour_method = 'tricontour'
slice_at = 0.0
slice_along = 'z'
# Level for horizontal slices
level = 0
# Focus domain on an area
xlabel = r'$\lambda \ / $ deg'
ylabel = r'$\phi \ / $ deg'
xlims = [-20, 20]
ylims = [-10, 30]
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
        # Handling of velocity
        # -------------------------------------------------------------------- #
        if field_name == 'u':

            # ---------------------------------------------------------------- #
            # Data extraction
            # ---------------------------------------------------------------- #

            field_X_full = extract_gusto_field(data_file, 'u_zonal', time_idx)
            field_Y_full = extract_gusto_field(data_file, 'u_meridional', time_idx)
            coords_X_full, coords_Y_full, coords_Z_full = \
                extract_gusto_coords(data_file, 'u_zonal')

            # Reshape
            field_X_full, coords_X_full, coords_Y_full, coords_Z_full, other_arrays = \
                reshape_gusto_data(field_X_full, coords_X_full,
                                   coords_Y_full, coords_Z_full,
                                   other_arrays=[field_Y_full])
            field_Y_full = other_arrays[0]

            # Take level for a horizontal slice
            field_data_X = field_X_full[:,level]
            field_data_Y = field_Y_full[:,level]
            coords_X = coords_X_full[:,level]
            coords_Y = coords_Y_full[:,level]

            # ---------------------------------------------------------------- #
            # Filter data to zoom in on area
            # ---------------------------------------------------------------- #
            data_dict = {'field_X': field_data_X,
                         'field_Y': field_data_Y,
                         'X': coords_X,
                         'Y': coords_Y}
            dataframe = pd.DataFrame(data_dict)
            dataframe = dataframe[(dataframe['X'] <= xlims[1]) &
                                (dataframe['X'] >= xlims[0]) &
                                (dataframe['Y'] <= ylims[1]) &
                                (dataframe['Y'] >= ylims[0])]
            field_data_X = dataframe['field_X'].values
            field_data_Y = dataframe['field_Y'].values
            field_data = np.sqrt(field_data_X**2 + field_data_Y**2)
            coords_X = dataframe['X'].values
            coords_Y = dataframe['Y'].values

            # ---------------------------------------------------------------- #
            # Plot data
            # ---------------------------------------------------------------- #
            cmap, lines = tomplot_cmap(contours, colour_scheme)
            cf, _ = plot_contoured_field(ax, coords_X, coords_Y, field_data,
                                         contour_method, contours, cmap=cmap,
                                         plot_contour_lines=False)
            _ = plot_field_quivers(ax, coords_X, coords_Y,
                                   field_data_X, field_data_Y,
                                   magnitude_filter=2, scale=5)

        # -------------------------------------------------------------------- #
        # Scalar fields
        # -------------------------------------------------------------------- #
        else:
            # ---------------------------------------------------------------- #
            # Data extraction
            # ---------------------------------------------------------------- #

            field_full = extract_gusto_field(data_file, field_name, time_idx)
            coords_X_full, coords_Y_full, coords_Z_full = \
                extract_gusto_coords(data_file, field_name)

            # Reshape
            field_full, coords_X_full, coords_Y_full, coords_Z_full = \
                reshape_gusto_data(field_full, coords_X_full,
                                    coords_Y_full, coords_Z_full)

            # Take level for a horizontal slice
            field_data = field_full[:,level]
            coords_X = coords_X_full[:,level]
            coords_Y = coords_Y_full[:,level]

            # ---------------------------------------------------------------- #
            # Filter data to zoom in on area
            # ---------------------------------------------------------------- #
            data_dict = {'field': field_data,
                        'X': coords_X,
                        'Y': coords_Y}
            dataframe = pd.DataFrame(data_dict)
            dataframe = dataframe[(dataframe['X'] <= xlims[1]) &
                                (dataframe['X'] >= xlims[0]) &
                                (dataframe['Y'] <= ylims[1]) &
                                (dataframe['Y'] >= ylims[0])]
            field_data = dataframe['field'].values
            coords_X = dataframe['X'].values
            coords_Y = dataframe['Y'].values

            # ---------------------------------------------------------------- #
            # Plot data
            # ---------------------------------------------------------------- #
            cmap, lines = tomplot_cmap(contours, colour_scheme)
            cf, _ = plot_contoured_field(ax, coords_X, coords_Y, field_data,
                                        contour_method, contours, cmap=cmap,
                                        line_contours=lines)

        # -------------------------------------------------------------------- #
        # Generic details
        # -------------------------------------------------------------------- #
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


