import numpy as np
from gusto import GeneralCubedSphereMesh
from firedrake import (FunctionSpace, Function, CheckpointFile,
                       SpatialCoordinate, atan_2, asin)
from os.path import join, abspath, dirname

filename = join(abspath(dirname(__file__)), "pangea_modified.png")

# Mesh details
radius = 6371220.
ncells_1d = 48
sphere_degree = 2
mesh = GeneralCubedSphereMesh(radius, ncells_1d, degree=sphere_degree)

# Function space to import to
V = FunctionSpace(mesh, "DG", 1)
b_field = Function(V, name='topography')

# Coordinates
mesh_x, mesh_y, mesh_z = SpatialCoordinate(mesh)
lon_DG = Function(V).interpolate(180.*atan_2(mesh_y,mesh_x)/np.pi)
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
lat_2d, lon_2d = np.meshgrid(lon_1d, lat_1d)

# Scale heights
from scipy import stats
max_height = 5000.0
min_land_height = 100.0
sea_col = stats.mode(pix.flatten())[0][0] # Assume sea colour is most common
min_col = np.min(pix[pix != sea_col])     # Minimum land value
pix[pix == sea_col] = 0
max_col = np.max(pix)
heights = (max_height - min_land_height) * (pix - min_col) / (max_col - min_col) + min_land_height
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
