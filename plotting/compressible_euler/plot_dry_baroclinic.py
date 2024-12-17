"""
Plot for a 2x1 graph of surface temperature and surface pressure fields.
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tomplot import (tomplot_contours, tomplot_cmap,
                     plot_contoured_field, add_colorbar_ax,
                     regrid_vertical_slice, tomplot_field_title,
                     extract_gusto_vertical_slice,
                     reshape_gusto_data, extract_gusto_field,
                     extract_gusto_coords, area_restriction)


# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
# When copying this example these should not be relative to this file
results_dir = '/data/home/dw603/firedrake-08_05_24/src/gusto/gusto_case_studies/case_studies/compressible_euler/results/dry_baroclinic_sphere'  # This needs to point to a data file
results_file_name = f'{results_dir}/field_output.nc'
plot_dir = f'plot_tests'
plot_stem = f'{plot_dir}/output_testing'

# ---------------------------------------------------------------------------- #
#Initial plot details
# ---------------------------------------------------------------------------- #

initial_field_names = ['Temperature', 'u_zonal']
initial_titles = ['Temperature', 'Zonal wind']
initial_slices = ['lon', 'lon']
initial_slice_ats = [0, 0]
initial_field_labels = [r'$T \ / $K', r'$u \ / $m/s']

## ----------------------------------------------------------------------------#
# Main plot details
# ---------------------------------------------------------------------------- #
field_names = ['Temperature', 'Pressure_Vt']
titles = ['Temperature', 'Pressure']
slices = ['z', 'z']
levels = [0, 0]
field_labels = [r'$T \ / $K', r'$P \ / $Pa']
domain_limit = {'X': (0, 180), 'Y': (0, 90)}
xlims = domain_limit['X']
ylims = domain_limit['Y']
colour_schemes = ['RdPu', 'RdBu_r']
temperature_contour = np.arange(220, 320, 10)
pressure_contour = np.arange(955, 1025, 5 )
contours = [temperature_contour, pressure_contour]

# Things that are the same for all subplots
time_idxs = [0, 16, -1]
contour_method = 'tricontour'

# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grids = {'lat': coords_lon_1d, 'lon': coords_lat_1d}
# Level for horizontal slices

xlims = domain_limit['X']
ylims = domain_limit['Y']
# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #

data_file = Dataset(results_file_name, 'r')
for time_idx in time_idxs:
    if time_idx == 0:
        fig, axarray = plt.subplots(1, 2, figsize=(16, 16))
        # Loop through subplots
        for i, (ax, field_name, field_label, colour_scheme, slice_along, slice_at, title) in \
            enumerate(zip(axarray.flatten(), initial_field_names, initial_field_labels, colour_schemes,
                        initial_slices,  initial_slice_ats, initial_titles)):
            # ------------------------------------------------------------------------ #
            # Data extraction
            # ------------------------------------------------------------------------ #
            orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
                extract_gusto_vertical_slice(data_file, field_name, time_idx,
                                            slice_along=slice_along, slice_at=slice_at)

            # Slices need regridding as points don't cleanly live along lon or lat = 0.0
            field_data, coords_hori, coords_Z = regrid_vertical_slice(plotting_grids[slice_along],
                                                                    slice_along, slice_at,
                                                                    orig_coords_X, orig_coords_Y,
                                                                    orig_coords_Z, orig_field_data)
            time = data_file['time'][time_idx]
            time_in_days = time / (24*60*60)
            # ------------------------------------------------------------------------ #
            # Plot data
            # ------------------------------------------------------------------------ #
            auto_contours = tomplot_contours(field_data)
            cmap, lines = tomplot_cmap(auto_contours, colour_scheme)
            cf, _ = plot_contoured_field(ax, coords_hori, coords_Z, field_data,
                                        contour_method, auto_contours, cmap=cmap,
                                        line_contours=lines)
            add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
            # Don't add ylabels unless left-most subplots
            ylabel = True if i % 3 == 0 else None
            ylabelpad = -30 if i > 2 else -10

            tomplot_field_title(ax, title, minmax=True, field_data=field_data)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle(f'Baroclinic Wave initial fields')
        # ---------------------------------------------------------------------------- #
        # Save figure
        # ---------------------------------------------------------------------------- #
        plot_name = f'{plot_stem}_intial_fields.png'
        print(f'Saving figure to {plot_name}')
        fig.savefig(plot_name, bbox_inches='tight')
        plt.close()


    else:    
        fig, axarray = plt.subplots(2, 1, figsize=(16, 8), sharey='row')
        # Loop through subplots
        for i, (ax, field_name, field_label, colour_scheme, slice_along, title, level, contour) in \
            enumerate(zip(axarray.flatten(), field_names, field_labels, colour_schemes,
                          slices, titles, levels, contours)):
            # ------------------------------------------------------------------------ #
            # Data extraction
            # ------------------------------------------------------------------------ #
            # Extraction
            field_full = extract_gusto_field(data_file, field_name, time_idx)
            coords_X_full, coords_Y_full, coords_Z_full = \
                extract_gusto_coords(data_file, field_name)

            # Reshape
            field_full, coords_X_full, coords_Y_full, _ = \
                reshape_gusto_data(field_full, coords_X_full,
                                coords_Y_full, coords_Z_full)

            # Domain restriction
            field_data, coords_hori, coords_Z = \
                area_restriction(field_full[:, level], coords_X_full[:, level],
                                coords_Y_full[:, level], domain_limit)

            time = data_file['time'][time_idx]
            time_in_days = time / (24*60*60)
            # ------------------------------------------------------------------------ #
            # Plot data
            # ------------------------------------------------------------------------ #

            cmap, lines = tomplot_cmap(contour, colour_scheme)
            cf, _ = plot_contoured_field(ax, coords_hori, coords_Z, field_data,
                                        contour_method, contour, cmap=cmap,
                                        line_contours=lines)
            add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10)
            # Don't add ylabels unless left-most subplots
            ylabel = True if i % 3 == 0 else None
            ylabelpad = -30 if i > 2 else -10

            tomplot_field_title(ax, title, minmax=True, field_data=field_data)
            ax.set_xlim(xlims)
            ax.set_xticks(xlims)
            ax.set_xticklabels(xlims)
            ax.set_ylim(ylims)
            ax.set_yticks(ylims)
            ax.set_yticklabels(ylims)

        # These subplots tend to be quite clustered together, so move them apart a bit
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle(f'Baroclinic Wave at {time_in_days} days')
        # ---------------------------------------------------------------------------- #
        # Save figure
        # ---------------------------------------------------------------------------- #
        plot_name = f'{plot_stem}_{time_in_days}_days.png'
        print(f'Saving figure to {plot_name}')
        fig.savefig(plot_name, bbox_inches='tight')
        plt.close()
