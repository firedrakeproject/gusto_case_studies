from dry_baroclinic_sphere import dry_baroclinic_sphere
from moist_baroclinic_channel import moist_baroclinic_channel
from moist_bryan_fritsch import moist_bryan_fritsch
from moist_skamarock_klemp import moist_skamarock_klemp
from mountain_hydrostatic import mountain_hydrostatic
from mountain_nonhydrostatic import mountain_nonhydrostatic
from robert_bubble import robert_bubble
from skamarock_klemp_hydrostatic import skamarock_klemp_hydrostatic
import pytest


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


def test_moist_skamarock_klemp():
    moist_skamarock_klemp(
        ncolumns=30,
        nlayers=5,
        dt=6.0,
        tmax=60.0,
        dumpfreq=10,
        dirname='pytest_moist_skamarock_klemp'
    )


def test_mountain_hydrostatic():
    mountain_hydrostatic(
        ncolumns=20,
        nlayers=10,
        dt=5.0,
        tmax=50.0,
        dumpfreq=10,
        dirname='pytest_mountain_hydrostatic',
        hydrostatic=False
    )


# Hydrostatic switch not currently working
@pytest.mark.xfail
def test_hyd_switch_mountain_hydrostatic():
    mountain_hydrostatic(
        ncolumns=20,
        nlayers=10,
        dt=5.0,
        tmax=50.0,
        dumpfreq=10,
        dirname='pytest_hyd_switch_mountain_hydrostatic',
        hydrostatic=True
    )


def test_mountain_nonhydrostatic():
    mountain_nonhydrostatic(
        ncolumns=20,
        nlayers=10,
        dt=5.0,
        tmax=10.0,
        dumpfreq=2,
        dirname='pytest_mountain_nonhydrostatic',
        hydrostatic=False
    )


# Hydrostatic switch not currently working
@pytest.mark.xfail
def test_hyd_switch_mountain_nonhydrostatic():
    mountain_nonhydrostatic(
        ncolumns=20,
        nlayers=10,
        dt=5.0,
        tmax=10.0,
        dumpfreq=2,
        dirname='pytest_hyd_switch_mountain_nonhydrostatic',
        hydrostatic=True
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


def test_skamarock_klemp_hydrostatic():
    skamarock_klemp_hydrostatic(
        ncolumns=20,
        nlayers=4,
        dt=6.0,
        tmax=60.0,
        dumpfreq=10,
        dirname='pytest_skamarock_klemp_hydrostatic',
        hydrostatic=False
    )


# Hydrostatic equations not currently working
@pytest.mark.xfail
def test_hyd_switch_skamarock_klemp_hydrostatic():
    skamarock_klemp_hydrostatic(
        ncolumns=20,
        nlayers=4,
        dt=6.0,
        tmax=60.0,
        dumpfreq=10,
        dirname='pytest_hyd_switch_skamarock_klemp_hydrostatic',
        hydrostatic=True
    )
