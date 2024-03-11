from firedrake import CheckpointFile, Function, FunctionSpace, errornorm, norm
from icecream import ic
import numpy as np
from tomplot import (set_tomplot_style, plot_convergence, tomplot_legend_fig,
                     only_minmax_ticklabels, tomplot_legend_ax, check_directory)
import matplotlib.pyplot as plt

path_to_data = '/home/d-witt/firedrake/src/gusto/plotting/MyTomplots/GW_tomconfig'
plot_dir = f'{path_to_data}/plots'
check_directory(plot_dir)


data_field = 'theta'
plot_name = f'{plot_dir}/Convergence_{data_field}.png'
dxs = [400, 500, 600, 700]
# copied out values as the algorithm takes awhile to run

def VOM_error_norm(path_to_checkpoints, dx, order, data_field):
    # Load high data
    high_res_chkpoint_path = f'{path_to_checkpoints}/GW_TomPaper_dt=1={order[0]}_{order[1]}_dx=100/chkpt.h5'
    with CheckpointFile(high_res_chkpoint_path, 'r') as high_chk:
        high_mesh = high_chk.load_mesh('firedrake_default_extruded')
        high_data_field = high_chk.load_function(high_mesh, data_field)

    # load low resolution data
    low_res_chkpoint_path = f'{path_to_checkpoints}/GW_TomPaper_dt=1={order[0]}_{order[1]}_dx={dx}/chkpt.h5'
    with CheckpointFile(low_res_chkpoint_path, 'r') as low_chk:
        low_mesh = low_chk.load_mesh('firedrake_default_extruded')
        low_data_field = low_chk.load_function(low_mesh, data_field)
    
    # interpolate high resolution data onto a low resolution mesh
    interpolated_field = Function(low_data_field.function_space()).interpolate(high_data_field)
    error_norm = errornorm(interpolated_field, low_data_field)
    norm_field = norm(low_data_field)
    normalised_error = error_norm / norm_field
    print(f'erorr for order {order[0]}_{order[1]}, dx={dx} is: {normalised_error}')
    return (normalised_error)

orders = [(0,0), (1,0), (0,1), (1,1)]
data_recorded = False
if not data_recorded:
    error_data = np.zeros((len(orders), len(dxs)))
    for j, order in enumerate(orders):
        print(f'calculating order for order: {order[0]}_{order[1]}')
        for i, dx in enumerate(dxs):
            print(f'i={i}, j={j}')
            error_data[j, i] = VOM_error_norm(path_to_data, dx, order, data_field)

    print(error_data)


# Plot options
colours = ['red', 'blue', 'purple', 'green']
markers = ['s', 'o', '^', 'X']
base_labels = ['order: (0,0)', 'order: (1,0)', 'order: (0,1)', 'order: (1,1)']
labels = [base_label+', gradient =' for base_label in base_labels]

log_by = 'data'
xlabel = r"$\log(\Delta x)$"
ylabel = r"$\log(\Delta q)$"
legend_loc = 'upper center'


set_tomplot_style()
fig, ax = plt.subplots(1,1, figsize = (8,8))
for j, colour, marker, label in enumerate(zip( colours, markers, labels)):
    plot_convergence(ax, dxs, error_data[j, :], color=colour, label=label,
                     marker = marker, log_by=log_by)

only_minmax_ticklabels(ax)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
tomplot_legend_fig(fig)
plt.legend(loc=legend_loc)
tomplot_legend_ax(ax, location='bottom')
plt.grid()

print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')