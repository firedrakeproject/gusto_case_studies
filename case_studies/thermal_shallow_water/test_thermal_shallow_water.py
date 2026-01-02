from thermal_galewsky_jet import thermal_galewsky
from thermal_williamson_5 import thermal_williamson_5
from moist_thermal_gravity_wave import moist_thermal_gw


def test_thermal_galewsky_jet():
    thermal_galewsky(
        ncells_per_edge=4,
        dt=900,
        tmax=1800,
        dumpfreq=2,
        dirname='pytest_thermal_galewsky_jet'
    )


def test_thermal_williamson_5():
    thermal_williamson_5(
        ncells_per_edge=4,
        dt=900,
        tmax=1800,
        dumpfreq=2,
        dirname='pytest_thermal_williamson_5'
    )


def test_moist_thermal_gravity_wave():
    moist_thermal_gw(
        ncells_per_edge=4,
        dt=900,
        tmax=1800,
        dumpfreq=2,
        dirname='pytest_moist_thermal_gravity_wave'
    )
