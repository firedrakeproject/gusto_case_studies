from galewsky_jet import galewsky_jet
from shallow_water_pangea import shallow_water_pangea
from utilities.create_pangea_dump import create_pangea_dump


def test_galewsky_jet():
    galewsky_jet(
        ncells_per_edge=4,
        dt=900,
        tmax=1800,
        dumpfreq=2,
        dirname='pytest_galewsky_jet'
    )


def test_shallow_water_pangea():
    shallow_water_pangea(
        ncells_per_edge=4,
        dt=900,
        tmax=1800,
        dumpfreq=2,
        dirname='pytest_shallow_water_pangea'
    )


def test_create_pangea_dump():
    create_pangea_dump(
        ncells_1d=3
    )