"""
Plots the Held-Suarez test case.

Plots include:
- Initial fields: lon/lat slices at surface of temperature and zonal wind,
                  and lat/z slices at lon=0 of temperature and zonal wind
- Instantaneous fields: lon/z slices of temperature and zonal wind after 100
                        and 1200 days
- Zonal averages: zonal averages of temperature and zonal wind
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
test = 'held_suarez'
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

all_contours = [
    np.linspace(150, 320, 18),
    np.linspace(-30, 30, 25)
]

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
# Instantaneous field plot details
# ---------------------------------------------------------------------------- #
# lon/z slices (at lat=0) of temperature and zonal wind, after 100 and 1200 days
instant_field_names = ['Temperature', 'u_zonal']
instant_titles = ['Temperature', 'Zonal wind']
instant_field_labels = [r'$T$ (K)', r'$u$ (m/s)']
instant_colour_schemes = ['OrRd', 'RdBu_r']
instant_slice_at = 0.0
instant_time_idxs = [1]

# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
contour_method = 'tricontour'
domain_limit = {'X' : (-180, 180), 'Y' : (-90, 90)}
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

# ---------------------------------------------------------------------------- #
# 1. Initial fields
# ---------------------------------------------------------------------------- #
time_idx = 0
fig, axarray = plt.subplots(2, 2, figsize=(16, 12))

time = data_file['time'][time_idx]
time_in_days = time / (24*60*60)

# Top row: lon/lat slices at the surface
for i, (ax, field_name, field_label, colour_scheme, title, slice_height, contours) in \
    enumerate(zip(axarray[0, :], initial_field_names, initial_field_labels,
                  initial_colour_schemes, initial_titles, initial_surface_heights,
                  all_contours)):
    # ------------------------------------------------------------------------ #
    # Data extraction
    # ------------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------------ #
    # Plot data
    # ------------------------------------------------------------------------ #
    cmap, lines = tomplot_cmap(contours, colour_scheme)
    cf, _ = plot_contoured_field(ax, coords_hori_X, coords_hori_Y, field_data,
                                 contour_method, contours, cmap=cmap,
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

# Bottom row: lat/z slices at lon = 0
for i, (ax, field_name, field_label, colour_scheme, title, contours) in \
    enumerate(zip(axarray[1, :], initial_field_names, initial_field_labels,
                  initial_colour_schemes, initial_titles, all_contours)):
    # ------------------------------------------------------------------------ #
    # Data extraction
    # ------------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------------ #
    # Plot data
    # ------------------------------------------------------------------------ #
    cmap, lines = tomplot_cmap(contours, colour_scheme)
    cf, _ = plot_contoured_field(ax, coords_hori, coords_Z, field_data,
                                 contour_method, contours, cmap=cmap,
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
    ax.set_ylabel('Height (km)', labelpad=-10, fontsize='15.0')

fig.suptitle('Held-Suarez: Initial fields', y=0.98, fontsize=24)
# ------------------------------------------------------------------------------ #
# Save figure
# ------------------------------------------------------------------------------ #
plot_name = f'{plot_stem}_initial.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------- #
# 2. Instantaneous fields after 100 and 1200 days
# ---------------------------------------------------------------------------- #

for time_idx in instant_time_idxs:
    fig, axarray = plt.subplots(1, 2, figsize=(16, 8))

    time = data_file['time'][time_idx]
    time_in_days = time / (24*60*60)

    for col_idx, (field_name, title, field_label, colour_scheme, contours) in \
        enumerate(zip(instant_field_names, instant_titles, instant_field_labels,
                    instant_colour_schemes, all_contours)):
        ax = axarray[col_idx]
        # -------------------------------------------------------------------- #
        # Data extraction: lon/z slice at lat = 0
        # -------------------------------------------------------------------- #
        orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
            extract_gusto_vertical_slice(
                data_file, field_name, time_idx,
                slice_along='lat', slice_at=instant_slice_at
            )

        # Slice needs regridding as points don't cleanly live along lat = 0.0
        field_data, coords_hori, coords_Z = regrid_vertical_slice(
            plotting_grids['lat'], 'lat', instant_slice_at,
            orig_coords_X, orig_coords_Y, orig_coords_Z, orig_field_data
        )
        # Convert height coordinate from m to km
        coords_Z = coords_Z / 1000.0
        # -------------------------------------------------------------------- #
        # Plot data
        # -------------------------------------------------------------------- #
        cmap, lines = tomplot_cmap(contours, colour_scheme)
        cf, _ = plot_contoured_field(
            ax, coords_hori, coords_Z, field_data, contour_method,
            contours, cmap=cmap, line_contours=lines
        )
        add_colorbar_ax(ax, cf, field_label, location='bottom', pad=0.1,
                        cbar_labelpad=-5, cbar_format='.0f')

        tomplot_field_title(
            ax, f'{title} (lon=0)', fontsize='17.0',
            minmax=True, field_data=field_data
        )
        ax.set_xlim(xlims)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize='15.0')
        ax.set_xlabel('Longitude', labelpad=-10, fontsize='15.0')
        ax.set_ylim(initial_ylims_km)
        ax.set_yticks(initial_ylims_km)
        ax.set_yticklabels(initial_ylims_km, fontsize='15.0')
        ax.set_ylabel('Height (km)', labelpad=-10, fontsize='15.0')

    fig.suptitle(rf'Held-Suarez: Instantaneous fields, $t=$ {int(time_in_days)} days', y=0.98, fontsize=24)
    # ------------------------------------------------------------------------ #
    # Save figure
    # ------------------------------------------------------------------------ #
    plot_name = f'{plot_stem}_instantaneous_day{int(time_in_days):04d}.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight', dpi=300)
    plt.close()

# ---------------------------------------------------------------------------- #
# 3. Zonal averages
# ---------------------------------------------------------------------------- #
# TODO: tomplot does not yet have a routine to compute zonal averages of
# fields extracted from unstructured Gusto output. Once such a routine
# exists, use it here to plot the zonal-mean temperature and zonal wind
# (as functions of latitude and height) for Held-Suarez.
print(
    'Zonal average plot not yet implemented -- awaiting a zonal-averaging '
    'routine in tomplot'
)

