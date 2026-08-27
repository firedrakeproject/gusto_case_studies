"""
Plots the solid body rotation test case.

Plots include:
- Initial fields: lon/lat slices at surface of temperature and zonal wind,
                  and lat/z slices at lon=0 of temperature and zonal wind
- Final fields: a 3x3 grid of zonal, meridional and radial wind components,
                each plotted on a lon/lat slice (z=10km), a lon/z slice
                (lat=0) and a lat/z slice (lon=0)
"""

from os.path import abspath, dirname
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, extract_gusto_coords, extract_gusto_field,
    extract_gusto_vertical_slice, reshape_gusto_data, area_restriction,
    tomplot_field_title, add_colorbar_fig, regrid_vertical_slice
)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
test = 'solid_body_rotation'
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

# ---------------------------------------------------------------------------- #
# Initial plot details
# ---------------------------------------------------------------------------- #

initial_field_names = ['Temperature', 'u_zonal']
initial_titles = ['Temperature', 'Zonal wind']
initial_field_labels = [r'$T$ (K)', r'$u$ (m/s)']
initial_colour_schemes = ['OrRd', 'YlGn']
# Placeholder contour values -- to be updated once output is available
initial_contours = [
    np.linspace(276, 281, 11),
    np.linspace(0, 45, 19)
]
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
time_idxs = [0, 30]
contour_method = 'tricontour'

# Columns: wind components. Rows: lon/lat (z=10km), lon/z (lat=0), lat/z (lon=0)
wind_field_names = ['u_zonal', 'u_meridional', 'u_radial']
wind_col_titles = [r'$u$', r'$v$', r'$w$']
wind_field_labels = [r'$u$ (m/s)', r'$v$ (m/s)', r'$w$ (m/s)']
wind_colour_schemes = ['YlGn', 'PiYG', 'RdBu_r']

# Placeholder contour values -- to be updated once output is available
wind_contours = [
    np.linspace(0, 45, 19),
    np.linspace(-0.5, 0.5, 21),
    np.linspace(-1e-3, 1e-3, 21)
]
remove_contours = [None, 0.0, 0.0]

row_types = ['lonlat', 'lonz', 'latz']
row_titles = ['Lon-Lat (z=10km)', 'Lon-z (lat=0)', 'Lat-z (lon=0)']
final_slice_height = 10000.0
final_slice_at = 0.0

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
domain_limit = {'X': (-180, 180), 'Y': (-90, 90)}
xlims = domain_limit['X']
ylims = domain_limit['Y']
time_idx = -1
# Things that are likely the same for all plots --------------------------------
set_tomplot_style(fontsize=18)
level = 0

yticks = [-90, 90]
ytick_labels = ['90S', '90N']
xticks = [-180, 180]
xtick_labels = ['180W', '180E']

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
        for i, (ax, field_name, field_label, colour_scheme, title, slice_height, contours) in \
            enumerate(zip(axarray[0, :], initial_field_names, initial_field_labels,
                          initial_colour_schemes, initial_titles, initial_surface_heights,
                          initial_contours)):
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
            cmap, lines = tomplot_cmap(contours, colour_scheme)
            cf, _ = plot_contoured_field(
                ax, coords_hori_X, coords_hori_Y, field_data,
                contour_method, contours, cmap=cmap, line_contours=lines
            )
            add_colorbar_ax(ax, cf, field_label, location='bottom', pad=0.15,
                            cbar_labelpad=-5, cbar_format='.0f')

            height_label = 'surface' if slice_height == 0.0 else f'z = {slice_height/1000:.0f} km'
            tomplot_field_title(
                ax, f'{title} ({height_label})', fontsize='17.0',
                minmax=True, field_data=field_data
            )
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
        for i, (ax, field_name, field_label, colour_scheme, title, contours) in \
            enumerate(zip(axarray[1, :], initial_field_names, initial_field_labels,
                          initial_colour_schemes, initial_titles, initial_contours)):
            # ---------------------------------------------------------------- #
            # Data extraction
            # ---------------------------------------------------------------- #
            orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
                extract_gusto_vertical_slice(
                    data_file, field_name, time_idx,
                    slice_along='lon', slice_at=initial_slice_at
                )

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
            cmap, lines = tomplot_cmap(contours, colour_scheme)
            cf, _ = plot_contoured_field(
                ax, coords_hori, coords_Z, field_data, contour_method,
                contours, cmap=cmap, line_contours=lines
            )
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

        fig.suptitle('Solid Body Rotation: Initial fields', y=0.96, fontsize=24)
        # -------------------------------------------------------------------- #
        # Save figure
        # -------------------------------------------------------------------- #
        plot_name = f'{plot_stem}_initial.png'
        print(f'Saving figure to {plot_name}')
        fig.savefig(plot_name, bbox_inches='tight')
        plt.close()

    else:
        fig, axarray = plt.subplots(3, 3, figsize=(20, 15))
        fig.subplots_adjust(wspace=0.25, hspace=0.25)

        time = data_file['time'][time_idx]
        time_in_days = time / (24*60*60)

        for col_idx, (field_name, col_title, field_label, colour_scheme, contours, remove_contour) in \
            enumerate(zip(wind_field_names, wind_col_titles, wind_field_labels,
                          wind_colour_schemes, wind_contours, remove_contours)):
            for row_idx, row_type in enumerate(row_types):
                ax = axarray[row_idx, col_idx]
                # ---------------------------------------------------------- #
                # Data extraction
                # ---------------------------------------------------------- #
                if row_type == 'lonlat':
                    field_full = extract_gusto_field(data_file, field_name, time_idx)
                    coords_X_full, coords_Y_full, coords_Z_full = \
                        extract_gusto_coords(data_file, field_name)

                    field_full, coords_X_full, coords_Y_full, coords_Z_full = \
                        reshape_gusto_data(field_full, coords_X_full, coords_Y_full, coords_Z_full)

                    level_idx = np.argmin(np.abs(coords_Z_full[0, :] - final_slice_height))

                    field_data, coords_hori, coords_vert = area_restriction(
                        field_full[:, level_idx], coords_X_full[:, level_idx],
                        coords_Y_full[:, level_idx], domain_limit
                    )
                else:
                    slice_along = 'lat' if row_type == 'lonz' else 'lon'

                    orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
                        extract_gusto_vertical_slice(
                            data_file, field_name, time_idx,
                            slice_along=slice_along, slice_at=final_slice_at
                        )

                    # Slice needs regridding as points don't cleanly live along the slice
                    field_data, coords_hori, coords_vert = regrid_vertical_slice(
                        plotting_grids[slice_along], slice_along, final_slice_at,
                        orig_coords_X, orig_coords_Y, orig_coords_Z, orig_field_data
                    )
                    # Convert height coordinate from m to km
                    coords_vert = coords_vert / 1000.0
                # ---------------------------------------------------------- #
                # Plot data
                # ---------------------------------------------------------- #
                cmap, lines = tomplot_cmap(
                    contours, colour_scheme, remove_contour=remove_contour
                )
                cf, _ = plot_contoured_field(
                    ax, coords_hori, coords_vert, field_data, contour_method,
                    contours, cmap=cmap, line_contours=lines
                )

                tomplot_field_title(
                    ax, f'{col_title}: {row_titles[row_idx]}', fontsize='15.0',
                    minmax=True, field_data=field_data, titlepad=10
                )

                if row_type == 'lonlat':
                    ax.set_xlim(xlims)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels, fontsize='15.0')
                    ax.set_xlabel('Longitude', labelpad=-10, fontsize='15.0')
                    ax.set_ylim(ylims)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(ytick_labels, fontsize='15.0')
                    ax.set_ylabel('Latitude', labelpad=-20, fontsize='15.0')
                elif row_type == 'lonz':
                    ax.set_xlim(xlims)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels, fontsize='15.0')
                    ax.set_xlabel('Longitude', labelpad=-10, fontsize='15.0')
                    ax.set_ylim(initial_ylims_km)
                    ax.set_yticks(initial_ylims_km)
                    ax.set_yticklabels(initial_ylims_km, fontsize='15.0')
                    ax.set_ylabel('Height (km)', labelpad=-10, fontsize='15.0')
                else:
                    ax.set_xlim(initial_xlims)
                    ax.set_xticks(yticks)
                    ax.set_xticklabels(ytick_labels, fontsize='15.0')
                    ax.set_xlabel('Latitude', labelpad=-10, fontsize='15.0')
                    ax.set_ylim(initial_ylims_km)
                    ax.set_yticks(initial_ylims_km)
                    ax.set_yticklabels(initial_ylims_km, fontsize='15.0')
                    ax.set_ylabel('Height (km)', labelpad=-10, fontsize='15.0')

            # ---------------------------------------------------------------- #
            # Add a single colorbar for this column, below the bottom axis
            # ---------------------------------------------------------------- #
            cbar_format = '.0f' if field_name == 'u_zonal' else '.1e'
            ax_idxs = [row*3 + col_idx for row in range(len(row_types))]
            add_colorbar_fig(
                fig, cf, field_label, location='bottom', ax_idxs=ax_idxs,
                cbar_format=cbar_format, cbar_padding=-0.02
            )

        # Re-apply spacing between axes, as add_colorbar_fig resets hspace
        fig.subplots_adjust(hspace=0.45)

        # -------------------------------------------------------------------- #
        # Save figure
        # -------------------------------------------------------------------- #
        fig.suptitle(rf'Solid Body Rotation, $t=$ {int(time_in_days):02d} days', y=0.95, fontsize=24)
        plot_name = f'{plot_stem}_day{int(time_in_days):02d}.png'
        print(f'Saving figure to {plot_name}')
        fig.savefig(plot_name, bbox_inches='tight', dpi=300)
        plt.close()
