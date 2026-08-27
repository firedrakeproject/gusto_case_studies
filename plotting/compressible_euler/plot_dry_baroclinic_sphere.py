"""
Plots the (dry, spherical) baroclinic wave test case.

Plots include:
- Initial fields: lon/lat slices at surface of temperature and zonal wind,
                  and lat/z slices at lon=0 of temperature and zonal wind
- Final fields: lon/lat slices at surface of temperature and pressure
                at 8 and 10 days
"""

from os.path import abspath, dirname
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from tomplot import (
    set_tomplot_style, tomplot_cmap, tomplot_contours, plot_contoured_field,
    add_colorbar_ax, extract_gusto_coords, extract_gusto_field,
    extract_gusto_vertical_slice, reshape_gusto_data, area_restriction,
    tomplot_field_title, regrid_vertical_slice
)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
test = 'dry_baroclinic_sphere'
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

# ---------------------------------------------------------------------------- #
# Initial plot details
# ---------------------------------------------------------------------------- #

initial_field_names = ['Temperature', 'u_zonal']
initial_titles = ['Temperature', 'Zonal wind']
initial_field_labels = [r'$T$ (K)', r'$u$ (m/s)']
initial_colour_schemes = ['OrRd', 'RdBu_r']
# Height (m) at which to take the top-row lon/lat slice for each field
initial_surface_heights = [0.0, 10000.0]
initial_slice_at = 0.0
initial_xlims = (-90, 90)
initial_ylims_km = (0, 30)

# 1D grids for vertical regridding
coords_lon_1d = np.linspace(-180, 180, 50)
coords_lat_1d = np.linspace(-90, 90, 50)
# Dictionary to hold plotting grids -- keys are "slice_along" values
plotting_grids = {'lat': coords_lon_1d, 'lon': coords_lat_1d}

# ---------------------------------------------------------------------------- #
# Final plot details
# ---------------------------------------------------------------------------- #
# Things that are the same for all subplots
time_idxs = [0, 8, 10]
contour_method = 'tricontour'

field_names = ['Pressure_Vt', 'Temperature']
titles = ['Pressure', 'Temperature']
slices = ['z', 'z']
levels = [0, 0]
slice_along = 'z'
colour_schemes = ['PiYG_r', 'RdBu_r']
centre_titles = ['Surface pressure', 'Temperature']
field_labels = [r'$P$ (hPa)', r'$T$ (K)']

contours = [
    np.linspace(92000, 103000, 23),
    np.linspace(230, 330, 11)
]
line_contours = [
    np.linspace(92000, 103000, 12),
    np.linspace(230, 330, 11)
]
remove_lines = [100000.0, None]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
domain_limit = {'X': (-180, 180), 'Y': (-90, 90)}
xlims = domain_limit['X']
ylims = domain_limit['Y']
time_idx = -1
# Things that are likely the same for all plots --------------------------------
set_tomplot_style(fontsize=24)
level = 0

yticks = [-90, 90]
ytick_labels = ['90S', '90N']
xticks = [-180, 180]
xtick_labels = ['180W', '180E']

pressure_ticks = [92000, 94000, 96000, 98000, 100000, 102000]
pressure_labels = ['920', '940', '960', '980', '1000', '1020']
temp_ticks = [240, 280, 320]
temp_labels = ['240', '280', '320']

cbar_ticks = [pressure_ticks, temp_ticks]
cbar_ticklabels = [pressure_labels, temp_labels]

cmap_rescale_types = ['top', None]
cmap_rescalings = [0.72, None]

# ---------------------------------------------------------------------------- #
# Things that are likely the same for all plots
# ---------------------------------------------------------------------------- #

data_file = Dataset(results_file_name, 'r')
for time_idx in time_idxs:
    if time_idx == 0:
        fig, axarray = plt.subplots(2, 2, figsize=(16, 12))

        time = data_file['time'][time_idx]
        time_in_days = time / (24*60*60)

        # -------------------------------------------------------------------- #
        # Top row: lon/lat slices at the surface
        # -------------------------------------------------------------------- #
        for i, (ax, field_name, field_label, colour_scheme, title, slice_height) in \
            enumerate(zip(axarray[0, :], initial_field_names, initial_field_labels,
                          initial_colour_schemes, initial_titles, initial_surface_heights)):
            # ---------------------------------------------------------------- #
            # Data extraction
            # ---------------------------------------------------------------- #
            field_full = extract_gusto_field(data_file, field_name, time_idx)
            coords_X_full, coords_Y_full, coords_Z_full = \
                extract_gusto_coords(data_file, field_name)

            field_full, coords_X_full, coords_Y_full, coords_Z_full = \
                reshape_gusto_data(field_full, coords_X_full, coords_Y_full, coords_Z_full)

            # Find the model level whose height is closest to the requested slice height
            level_idx = np.argmin(np.abs(coords_Z_full[0, :] - slice_height))

            field_data, coords_hori_X, coords_hori_Y = area_restriction(
                field_full[:, level_idx], coords_X_full[:, level_idx],
                coords_Y_full[:, level_idx], domain_limit
            )
            # ---------------------------------------------------------------- #
            # Plot data
            # ---------------------------------------------------------------- #
            auto_contours = tomplot_contours(field_data)
            cmap, lines = tomplot_cmap(auto_contours, colour_scheme)
            cf, _ = plot_contoured_field(ax, coords_hori_X, coords_hori_Y, field_data,
                                         contour_method, auto_contours, cmap=cmap,
                                         line_contours=lines)
            add_colorbar_ax(ax, cf, field_label, location='bottom', pad=0.15,
                            cbar_labelpad=-5, cbar_format='.0f')

            height_label = 'surface' if slice_height == 0.0 else f'z = {slice_height/1000:.0f} km'
            tomplot_field_title(ax, f'{title} ({height_label})', fontsize='17.0',
                                minmax=True, field_data=field_data)
            ax.set_xlim(xlims)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, fontsize='15.0')
            ax.set_xlabel('Longitude', labelpad=-10, fontsize='15.0')
            ax.set_ylim(ylims)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, fontsize='15.0')
            ax.set_ylabel('Latitude', labelpad=-20, fontsize='15.0')

        # -------------------------------------------------------------------- #
        # Bottom row: lat/z slices at lon = 0
        # -------------------------------------------------------------------- #
        for i, (ax, field_name, field_label, colour_scheme, title) in \
            enumerate(zip(axarray[1, :], initial_field_names, initial_field_labels,
                          initial_colour_schemes, initial_titles)):
            # ---------------------------------------------------------------- #
            # Data extraction
            # ---------------------------------------------------------------- #
            orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
                extract_gusto_vertical_slice(data_file, field_name, time_idx,
                                             slice_along='lon', slice_at=initial_slice_at)

            # Slice needs regridding as points don't cleanly live along lon = 0.0
            field_data, coords_hori, coords_Z = regrid_vertical_slice(
                plotting_grids['lon'], 'lon', initial_slice_at,
                orig_coords_X, orig_coords_Y, orig_coords_Z, orig_field_data
            )
            # Convert height coordinate from m to km
            coords_Z = coords_Z / 1000.0
            # ---------------------------------------------------------------- #
            # Plot data
            # ---------------------------------------------------------------- #
            auto_contours = tomplot_contours(field_data)
            cmap, lines = tomplot_cmap(auto_contours, colour_scheme)
            cf, _ = plot_contoured_field(ax, coords_hori, coords_Z, field_data,
                                         contour_method, auto_contours, cmap=cmap,
                                         line_contours=lines)
            add_colorbar_ax(ax, cf, field_label, location='bottom', pad=0.15,
                            cbar_labelpad=-5, cbar_format='.0f')

            tomplot_field_title(ax, f'{title} (lon = 0)', fontsize='17.0',
                                minmax=True, field_data=field_data)
            ax.set_xlim(initial_xlims)
            ax.set_xticks(yticks)
            ax.set_xticklabels(ytick_labels, fontsize='15.0')
            ax.set_xlabel('Latitude', labelpad=-10, fontsize='15.0')
            ax.set_ylim(initial_ylims_km)
            ax.set_yticks(initial_ylims_km)
            ax.set_yticklabels(initial_ylims_km, fontsize='15.0')
            ax.set_ylabel('Height (km)', labelpad=-20, fontsize='15.0')

        fig.suptitle('Baroclinic wave: Initial fields', y=0.98, fontsize=24)
        # -------------------------------------------------------------------- #
        # Save figure
        # -------------------------------------------------------------------- #
        plot_name = f'{plot_stem}_initial.png'
        print(f'Saving figure to {plot_name}')
        fig.savefig(plot_name, bbox_inches='tight')
        plt.close()

    else:
        fig, axarray = plt.subplots(1, 2, figsize=(15, 6), sharex='col', sharey='row')

        for i, (ax, field_name, colour_scheme, field_label, contour, line_contour,
                centre_title, cbar_tick, cbar_ticklabel, remove_contour, cmap_rescale_type,
                cmap_rescaling) in enumerate(zip(
                axarray, field_names, colour_schemes, field_labels, contours, line_contours,
                centre_titles, cbar_ticks, cbar_ticklabels, remove_lines,
                cmap_rescale_types, cmap_rescalings)):

            # Data extraction --------------------------------------------------
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

            cmap, _ = tomplot_cmap(
                contour, colour_scheme, cmap_rescale_type=cmap_rescale_type,
                cmap_rescaling=cmap_rescaling
            )
            _, lines = tomplot_cmap(
                line_contour, colour_scheme, remove_contour=remove_contour
            )

            # Plot data --------------------------------------------------------
            cf, _ = plot_contoured_field(
                ax, coords_hori, coords_Z, field_data, contour_method, contour,
                cmap=cmap, line_contours=lines
            )

            cb = add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_ticks=cbar_tick, pad=0.07)
            cb.ax.set_xticklabels(cbar_ticklabel)

            # Labels -----------------------------------------------------------
            ax.set_title(centre_title, fontsize='17.0')

            # Setting Y / X ticks
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, fontsize='15.0')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, fontsize='15.0')

        time = data_file['time'][time_idx]
        time_in_days = time / (24*60*60)

        # -------------------------------------------------------------------- #
        # Save figure
        # -------------------------------------------------------------------- #
        fig.suptitle(rf'Baroclinic wave, $t=$ {int(time_in_days):02d} days', y=0.98, fontsize=24)
        plot_name = f'{plot_stem}_day{int(time_in_days):02d}.png'
        print(f'Saving figure to {plot_name}')
        fig.savefig(plot_name, bbox_inches='tight', dpi=300)
        plt.close()
