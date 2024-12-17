from dry_baroclinic_sphere import dry_baroclinic_sphere
from moist_baroclinic_channel import moist_baroclinic_channel
from moist_bryan_fritsch import moist_bryan_fritsch
from mountain_nonhydrostatic import mountain_nonhydrostatic
from robert_bubble import robert_bubble
from solid_body_rotation import solid_body_sphere


def test_dry_baroclinic_sphere():
    dry_baroclinic_sphere(
        ncell_per_edge=4,
        nlayers=3,
        dt=900,
        tmax=1800,
        dumpfreq=2,
        dirname='pytest_dry_baroclinic_sphere'
    )


def test_moist_baroclinic_channel():
    moist_baroclinic_channel(
        nx=10,
        ny=5,
        nlayers=5,
        dt=300.0,
        tmax=600.0,
        dumpfreq=2,
        dirname='pytest_moist_baroclinic_channel'
    )


def test_moist_bryan_fritsch():
    moist_bryan_fritsch(
        ncolumns=5,
        nlayers=5,
        dt=2.0,
        tmax=10.0,
        dumpfreq=5,
        dirname='pytest_moist_bryan_fritsch'
    )


def test_mountain_nonhydrostatic():
    mountain_nonhydrostatic(
        ncolumns=20,
        nlayers=10,
        dt=5.0,
        tmax=10.0,
        dumpfreq=2,
        dirname='pytest_mountain_nonhydrostatic'
    )


def test_robert_bubble():
    robert_bubble(
        ncolumns=5,
        nlayers=5,
        dt=1.0,
        tmax=2.0,
        dumpfreq=2,
        dirname='pytest_robert_bubble'
    )


def test_solid_body_sphere():
    solid_body_sphere(
        ncell_per_edge=4,
        nlayers=3,
        dt=900,
        tmax=1800,
        dumpfreq=2,
        dirname='pytest_solid_body_sphere'
    )