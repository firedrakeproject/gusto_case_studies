from dry_baroclinic_sphere import dry_baroclinic_sphere
from moist_bryan_fritsch import moist_bryan_fritsch

def test_dry_baroclinic_sphere():
    dry_baroclinic_sphere(
        ncell_per_edge=4,
        nlayers=3,
        dt=900,
        tmax=1800,
        dumpfreq=2
    )

def test_moist_bryan_fritsch():
    moist_bryan_fritsch(
        ncolumns=5,
        nlayers=5,
        dt=2.0,
        tmax=10.0,
        dumpfreq=5
    )