"""
Plots the non-hydrostatic mountain test case.

This plots:
(a) w @ t = 9000 s, (b) theta @ t = 9000 s
"""
from os.path import abspath, dirname
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, tomplot_contours,
    extract_gusto_coords, extract_gusto_field,
)

test = 'mountain_nonhydrostatic'

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these paths need editing, which will usually involve
# removing the abspath part to set directory paths relative to this file
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
final_field_names = ['u_z', 'theta_perturbation']
final_colour_schemes = ['PiYG', 'RdBu_r']
final_field_labels = [r'$w$ (m s$^{-1}$)', r'$\Delta\theta$ (K)']

# ---------------------------------------------------------------------------- #
# Initial plot details
# ---------------------------------------------------------------------------- #
initial_field_names = ['Exner', 'theta']
initial_colour_schemes = ['PuBu', 'Reds']
initial_field_labels = [r'$\Pi$', r'$\theta$ (K)']

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
xlims = [0., 144.]
ylims = [0., 35.]

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

# ---------------------------------------------------------------------------- #
# INITIAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(1, 2, figsize=(18, 6), sharex='all', sharey='all')
time_idx = 0

for i, (ax, field_name, colour_scheme, field_label) in \
        enumerate(zip(
            axarray.flatten(), initial_field_names, initial_colour_schemes,
            initial_field_labels
        )):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    contours = tomplot_contours(field_data)
    cmap, lines = tomplot_cmap(contours, colour_scheme)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(ax, cf, field_label, location='bottom')
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

    # Labels -------------------------------------------------------------------
    if i == 0:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    ax.set_xlabel(r'$x$ (km)', labelpad=-10)
    ax.set_xlim(xlims)
    ax.set_xticks(xlims)
    ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.suptitle(f't = {time:.1f} s')
fig.subplots_adjust(wspace=0.25)
plot_name = f'{plot_stem}_initial.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------- #
# FINAL PLOTTING
# ---------------------------------------------------------------------------- #
fig, axarray = plt.subplots(1, 2, figsize=(18, 6), sharex='all', sharey='all')
time_idx = -1

for i, (ax, field_name, colour_scheme, field_label) in \
        enumerate(zip(
            axarray.flatten(), final_field_names, final_colour_schemes,
            final_field_labels
        )):

    # Data extraction ----------------------------------------------------------
    field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    time = data_file['time'][time_idx]

    contours = tomplot_contours(field_data)
    cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=0.0)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(ax, cf, field_label, location='bottom')
    tomplot_field_title(
        ax, None, minmax=True, field_data=field_data, minmax_format='.3f'
    )

    # Labels -------------------------------------------------------------------
    if i == 0:
        ax.set_ylabel(r'$z$ (km)', labelpad=-20)
        ax.set_ylim(ylims)
        ax.set_yticks(ylims)
        ax.set_yticklabels(ylims)

    ax.set_xlabel(r'$x$ (km)', labelpad=-10)
    ax.set_xlim(xlims)
    ax.set_xticks(xlims)
    ax.set_xticklabels(xlims)

# Save figure ------------------------------------------------------------------
fig.suptitle(f't = {time:.1f} s')
fig.subplots_adjust(wspace=0.25)
plot_name = f'{plot_stem}_final.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()
