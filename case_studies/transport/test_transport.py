from four_part_sbr import four_part_sbr
from nair_lauritzen_divergent import nair_lauritzen_divergent
from nair_lauritzen_non_divergent import nair_lauritzen_non_divergent
from terminator_toy import terminator_toy
from vertical_slice_nair_lauritzen import vertical_slice_nair_lauritzen
from volcanic_ash import volcanic_ash


def test_four_part_sbr():
    four_part_sbr(
        ncells_per_edge=3,
        dt=0.5,
        tmax=1.0,
        dumpfreq=2,
        dirname='pytest_four_part_sbr'
    )


def test_nair_lauritzen_divergent_cylinder():
    nair_lauritzen_divergent(
        initial_conditions='slotted_cylinder',
        ncells_per_edge=3,
        dt=900.0,
        tmax=1800.0,
        dumpfreq=2,
        dirname='pytest_nair_lauritzen_div_cylinder'
    )


def test_nair_lauritzen_divergent_cosine():
    nair_lauritzen_divergent(
        initial_conditions='cosine_bells',
        no_background_flow=True,
        ncells_per_edge=3,
        dt=900.0,
        tmax=1800.0,
        dumpfreq=2,
        dirname='pytest_nair_lauritzen_div_cosine'
    )


def test_nair_lauritzen_divergent_gaussian():
    nair_lauritzen_divergent(
        initial_conditions='gaussian',
        ncells_per_edge=3,
        dt=900.0,
        tmax=1800.0,
        dumpfreq=2,
        dirname='pytest_nair_lauritzen_div_gaussian'
    )


def test_nair_lauritzen_non_divergent_cylinder():
    nair_lauritzen_non_divergent(
        initial_conditions='slotted_cylinder',
        ncells_per_edge=3,
        dt=900.0,
        tmax=1800.0,
        dumpfreq=2,
        dirname='pytest_nair_lauritzen_divfree_cylinder'
    )


def test_nair_lauritzen_non_divergent_cosine():
    nair_lauritzen_non_divergent(
        initial_conditions='cosine_bells',
        ncells_per_edge=3,
        dt=900.0,
        tmax=1800.0,
        dumpfreq=2,
        dirname='pytest_nair_lauritzen_divfree_cosine'
    )


def test_nair_lauritzen_non_divergent_gaussian():
    nair_lauritzen_non_divergent(
        initial_conditions='gaussian',
        ncells_per_edge=3,
        dt=900.0,
        tmax=1800.0,
        dumpfreq=2,
        dirname='pytest_nair_lauritzen_divfree_gaussian',
    )


def test_terminator_toy():
    terminator_toy(
        ncells_per_edge=3,
        dt=900.0,
        tmax=1800.0,
        dumpfreq=2,
        dirname='pytest_terminator'
    )


def test_vertical_slice_nair_lauritzen_consistency():
    vertical_slice_nair_lauritzen(
        configuration='consistency',
        ncells_1d=5,
        dt=1.0,
        tmax=2.0,
        dumpfreq=2,
        dirname='pytest_vertical_slice_nair_lauritzen_consistency'
    )


def test_vertical_slice_nair_lauritzen_convergence():
    vertical_slice_nair_lauritzen(
        configuration='convergence',
        ncells_1d=5,
        dt=1.0,
        tmax=2.0,
        dumpfreq=2,
        dirname='pytest_vertical_slice_nair_lauritzen_convergence'
    )


def test_volcanic_ash():
    volcanic_ash(
        ncells_1d=5,
        dt=1.0,
        tmax=2.0,
        dumpfreq=2,
        dirname='pytest_volcanic_ash'
    )
