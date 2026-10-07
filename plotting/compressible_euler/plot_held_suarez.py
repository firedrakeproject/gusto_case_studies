"""
Plots the Held-Suarez test case.

Plots include:
- Initial fields: lon/lat slices at surface of temperature and zonal wind,
                  and lat/z slices at lon=0 of temperature and zonal wind
- Instantaneous fields: lon/z slices of temperature and zonal wind after 100
                        and 1200 days
- Instantaneous horizontal slices: lon/lat slices of temperature and pressure
                        at the surface, and temperature and zonal wind at
                        15 km height
- Zonal averages: zonal averages of temperature and zonal wind
"""

from os.path import abspath, dirname
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, extract_gusto_coords, extract_gusto_field,
    extract_gusto_vertical_slice, reshape_gusto_data, area_restriction,
    tomplot_field_title, regrid_vertical_slice, zonal_average
)

# ---------------------------------------------------------------------------- #
# Directory for results and plots
# ---------------------------------------------------------------------------- #
test = 'held_suarez'
plot_instantaneous = False
results_file_name = f'{abspath(dirname(__file__))}/../../results/{test}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/../../figures/compressible_euler/{test}'

all_contours = [
    np.linspace(150, 320, 18),
    np.linspace(-60, 60, 31)
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
if plot_instantaneous:
    instant_time_idxs = range(1, 13)
else:
    instant_time_idxs = []

# ---------------------------------------------------------------------------- #
# Instantaneous horizontal slice plot details
# ---------------------------------------------------------------------------- #
# lon/lat slices of temperature and pressure at the surface, and temperature
# and zonal wind at a height of 15 km
horiz_field_names = ['Temperature', 'Pressure_Vt', 'Temperature', 'u_zonal']
horiz_titles = ['Temperature', 'Surface Pressure', 'Temperature', 'Zonal wind']
horiz_field_labels = [r'$T$ (K)', r'$p$ (hPa)', r'$T$ (K)', r'$u$ (m/s)']
horiz_colour_schemes = ['OrRd', 'PiYG_r', 'YlOrBr', 'RdBu_r']
horiz_slice_heights = [0.0, 0.0, 15000.0, 15000.0]
horiz_scale_factors = [1.0, 0.01, 1.0, 1.0]
horiz_contours = [
    np.linspace(250, 320, 29), np.linspace(940, 1030, 19),
    np.linspace(190, 220, 13), all_contours[1]
]

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
# ---------------------------------------------------------------------------- #
# Save figure
# ---------------------------------------------------------------------------- #
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
        # Data extraction: lon/z slice at lon = 0
        # -------------------------------------------------------------------- #
        orig_field_data, orig_coords_X, orig_coords_Y, orig_coords_Z = \
            extract_gusto_vertical_slice(
                data_file, field_name, time_idx,
                slice_along='lon', slice_at=instant_slice_at
            )

        # Slice needs regridding as points don't cleanly live along lon = 0.0
        field_data, coords_hori, coords_Z = regrid_vertical_slice(
            plotting_grids['lon'], 'lon', instant_slice_at,
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
        ax.set_xlim(initial_xlims)
        ax.set_xticks(yticks)
        ax.set_xticklabels(ytick_labels, fontsize='15.0')
        ax.set_xlabel('Latitude', labelpad=-10, fontsize='15.0')
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
# 3. Instantaneous horizontal slices
# ---------------------------------------------------------------------------- #

for time_idx in instant_time_idxs:
    fig, axarray = plt.subplots(2, 2, figsize=(16, 12))

    time = data_file['time'][time_idx]
    time_in_days = time / (24*60*60)

    for ax, field_name, title, field_label, colour_scheme, slice_height, scale_factor, contours in \
        zip(axarray.flatten(), horiz_field_names, horiz_titles, horiz_field_labels,
            horiz_colour_schemes, horiz_slice_heights, horiz_scale_factors, horiz_contours):
        # -------------------------------------------------------------------- #
        # Data extraction
        # -------------------------------------------------------------------- #
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
        field_data = field_data * scale_factor
        # -------------------------------------------------------------------- #
        # Plot data
        # -------------------------------------------------------------------- #
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

    fig.suptitle(rf'Held-Suarez: Instantaneous horizontal slices, $t=$ {int(time_in_days)} days', y=0.98, fontsize=24)
    # ------------------------------------------------------------------------ #
    # Save figure
    # ------------------------------------------------------------------------ #
    plot_name = f'{plot_stem}_instantaneous_horizontal_day{int(time_in_days):04d}.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight', dpi=300)
    plt.close()

# ---------------------------------------------------------------------------- #
# 4. Zonal averages
# ---------------------------------------------------------------------------- #
zonal_field_names = ['Temperature_average', 'u_zonal_average']
time_idxs = range(2, 13)

fig, axarray = plt.subplots(1, 2, figsize=(16, 8))

time = data_file['time'][time_idxs[-1]]
time_in_days = time / (24*60*60)

for col_idx, (field_name, title, field_label, colour_scheme, contours) in \
    enumerate(zip(zonal_field_names, instant_titles, instant_field_labels,
                  instant_colour_schemes, all_contours)):
    ax = axarray[col_idx]
    # ------------------------------------------------------------------------ #
    # Data extraction and zonal averaging
    # ------------------------------------------------------------------------ #

    # Time average fields
    field_full = extract_gusto_field(data_file, field_name, time_idxs[0])
    for time_idx in time_idxs[1:]:
        field_full += extract_gusto_field(data_file, field_name, time_idx)
    field_full /= len(time_idxs)

    coords_X_full, coords_Y_full, coords_Z_full = \
        extract_gusto_coords(data_file, field_name)

    # Bins with no data points (e.g. near the poles, for fields whose mesh
    # points don't extend all the way to the poles) are omitted by
    # zonal_average, rather than being left as NaN
    zonal_mean, lat_bin_centres, level_heights = zonal_average(
        field_full, coords_X_full, coords_Y_full, coords_Z_full, num_bins=60
    )
    # Convert height coordinate from m to km
    level_heights_km = level_heights / 1000.0
    # Zonal-mean data is structured (lat x height), so build a meshgrid and
    # use the "contour" method rather than "tricontour"
    coords_lat_2d, coords_height_2d = np.meshgrid(
        lat_bin_centres, level_heights_km
    )
    # ------------------------------------------------------------------------ #
    # Plot data
    # ------------------------------------------------------------------------ #
    cmap, lines = tomplot_cmap(contours, colour_scheme)
    cf, _ = plot_contoured_field(
        ax, coords_lat_2d, coords_height_2d, zonal_mean.T, 'contour',
        contours, cmap=cmap, line_contours=lines
    )
    add_colorbar_ax(ax, cf, field_label, location='bottom', pad=0.1,
                    cbar_labelpad=-5, cbar_format='.0f')

    tomplot_field_title(
        ax, f'Zonal-mean {title.lower()}', fontsize='17.0',
        minmax=True, field_data=zonal_mean
    )
    ax.set_xlim(initial_xlims)
    ax.set_xticks(yticks)
    ax.set_xticklabels(ytick_labels, fontsize='15.0')
    ax.set_xlabel('Latitude', labelpad=-10, fontsize='15.0')
    ax.set_ylim(initial_ylims_km)
    ax.set_yticks(initial_ylims_km)
    ax.set_yticklabels(initial_ylims_km, fontsize='15.0')
    ax.set_ylabel('Height (km)', labelpad=-10, fontsize='15.0')

fig.suptitle(rf'Held-Suarez: Zonal averages, $t=$ {int(time_in_days)} days', y=0.98, fontsize=24)
# ---------------------------------------------------------------------------- #
# Save figure
# ---------------------------------------------------------------------------- #
plot_name = f'{plot_stem}_zonal_average.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight', dpi=300)
plt.close()
