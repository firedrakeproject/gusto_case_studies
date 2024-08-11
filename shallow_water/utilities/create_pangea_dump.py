"""
Creates the checkpoint with the topography field for the shallow water pangea
test case. Can be used with cubed sphere or icosahedral sphere meshes.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from gusto import GeneralCubedSphereMesh, GeneralIcosahedralSphereMesh
from firedrake import (FunctionSpace, Function, CheckpointFile,
                       SpatialCoordinate, atan2, asin)
from os.path import join, abspath, dirname

filename = join(abspath(dirname(__file__)), "pangea_modified.png")

create_pangea_defaults = {
    'mesh_type': 'cubed_sphere',  # "cubed_sphere" or "icosahedron"
    'sphere_degree': '2'
}


def create_pangea_dump(
        ncells_1d,
        mesh_type=create_pangea_defaults['mesh_type'],
        sphere_degree=create_pangea_defaults['sphere_degree']
):

    # Mesh details
    radius = 6371220.
    sphere_degree = 2

    if mesh_type == 'cubed_sphere':
        mesh = GeneralCubedSphereMesh(radius, ncells_1d, degree=sphere_degree)
    elif mesh_type == 'icosahedron':
        mesh = GeneralIcosahedralSphereMesh(radius, ncells_1d, degree=sphere_degree)
    else:
        raise ValueError(f'mesh_type {mesh_type} is invalid. Permitted options'
                         + 'are "cubed_sphere" and "icosahedron"')

    # Function space to import to
    V = FunctionSpace(mesh, "DG", 1)
    b_field = Function(V, name='topography')

    # Coordinates
    mesh_x, mesh_y, mesh_z = SpatialCoordinate(mesh)
    lon_DG = Function(V).interpolate(180.*atan2(mesh_y, mesh_x)/np.pi)
    lat_DG = Function(V).interpolate(180.*asin(mesh_z/radius)/np.pi)

    # Import image and convert to array of data
    from PIL import Image
    img = Image.open(filename).convert('L')
    pix = np.array(img).astype(int)
    img.close()

    # Create corresponding longitude-latitude coordinates
    num_lat, num_lon = np.shape(pix)
    lat_1d = np.linspace(-90., 90., num_lat)
    lon_1d = np.linspace(-180., 180., num_lon)

    # Scale heights
    from scipy import stats
    max_height = 5000.0
    min_land_height = 100.0
    sea_col = stats.mode(pix.flatten())[0]  # Assume sea colour is most common
    min_col = np.min(pix[pix != sea_col])   # Minimum land value
    pix[pix == sea_col] = 0
    max_col = np.max(pix)
    heights = (
        (max_height - min_land_height)
        * (pix - min_col) / (max_col - min_col)
        + min_land_height
    )
    heights[heights < 0.0] = 0.0

    # Interpolate
    from scipy.interpolate import RectBivariateSpline
    interpolator = RectBivariateSpline(lon_1d, lat_1d, heights.T)
    b_field.dat.data[:] = interpolator(lon_DG.dat.data[:], lat_DG.dat.data[:], grid=False)

    # Checkpoint!
    chkfile = join(abspath(dirname(__file__)), f"pangea_C{ncells_1d}_chkpt.h5")
    chkpt = CheckpointFile(chkfile, 'w')
    chkpt.save_mesh(mesh)
    chkpt.save_function(b_field)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncells_1d',
        help="The number of cells per edge of cubed sphere or icosahedron panel",
        type=int
    )
    parser.add_argument(
        '--mesh_type',
        help="The mesh type: cubed_sphere or icosahedron",
        type=str,
        default=create_pangea_defaults['mesh_type']
    )
    parser.add_argument(
        '--sphere_degree',
        help="The degree of polynomial approximation of the sphere",
        type=int,
        default=create_pangea_defaults['sphere_degree']
    )
    args, unknown = parser.parse_known_args()

    create_pangea_dump(**vars(args))
