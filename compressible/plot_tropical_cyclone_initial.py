"""
Plots the initial condition of the tropical cyclone
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tomplot import (set_tomplot_style, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     regrid_vertical_slice, tomplot_field_title,
                     extract_gusto_vertical_slice, apply_gusto_domain)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/home/thomas/firedrake/src/gusto/results/tropical_cyclone_dt1200_n11_degree1_vector_advective'
plot_dir = results_dir
results_file_name = f'{results_dir}/field_output.nc'
plot_name = f'{plot_dir}/tropical_cyclone_initial.png'

# ---------------------------------------------------------------------------- #
# Things that should be altered based on the plot
# ---------------------------------------------------------------------------- #
# Specify lists for variables that are different between subplots
field_names = ['Temperature', 'water_vapour',
               'Theta_d', 'Pressure_Vt_bar']
field_labels = [r'$T \ / $ K', r'$m_v \ / $ kg kg$^{-1}$',
                r'$\theta \ / $ K', r'$\bar{p} \ / $ Pa']
all_contours = [np.linspace(190, 320, 14),
                np.linspace(0.0, 0.02, 11),
                [250,260,270,280,290,300,310,320,330,340,350,360,390,420,450,480,510,540,570,600,630,660,690],
                [0,1000,2500,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000]]
# Things that are the same for all subplots
time_idx = 0
contour_method = 'tricontour'
slice_at = 0.0
slice_along = 'lon'
colour_scheme = 'turbo'
# 1D grids for vertical regridding
plotting_grid = np.linspace(-180, 180, 50)
# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')
fig, axarray = plt.subplots(2, 2, figsize=(16, 16), sharey='row', sharex='col')

# Loop through subplots
for i, (ax, field_name, field_label, contours) in \
    enumerate(zip(axarray.flatten(), field_names, field_labels, all_contours)):
    # ------------------------------------------------------------------------ #
    # Data extraction
    # ------------------------------------------------------------------------ #
    orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
        extract_gusto_vertical_slice(data_file, field_name, time_idx,
                                    slice_along=slice_along, slice_at=slice_at)

    # Slices need regridding as points don't cleanly live along lon or lat = 0.0
    field_data, coords_hori, coords_Z = regrid_vertical_slice(plotting_grid,
                                                              slice_along, slice_at,
                                                              orig_coords_X, orig_coords_Y,
                                                              orig_coords_Z, orig_field_data)

    coords_Z /= 1000.

    # ------------------------------------------------------------------------ #
    # Plot data
    # ------------------------------------------------------------------------ #
    cmap, lines = tomplot_cmap(contours, colour_scheme)
    cf, _ = plot_contoured_field(ax, coords_hori, coords_Z, field_data,
                                 contour_method, contours, cmap=cmap,
                                 line_contours=lines)
    add_colorbar_ax(ax, cf, field_label, location='right')
    # Don't add ylabels unless left-most subplots
    xlabel = True if i > 1 else None
    ylabel = True if i % 2 == 0 else None
    apply_gusto_domain(ax, data_file, slice_along=slice_along,
                       xlabel=xlabel, ylabel=ylabel,
                       xlabelpad=-15, vertical_units='km')
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

# ---------------------------------------------------------------------------- #
# Save figure
# ---------------------------------------------------------------------------- #
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()


