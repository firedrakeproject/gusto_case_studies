from dry_baroclinic_sphere import dry_baroclinic_sphere

def test_dry_baroclinic_sphere():
    dry_baroclinic_sphere(
        ncell_per_edge=4,
        nlayers=3,
        dt=900,
        tmax=1800,
        dumpfreq=2
    )