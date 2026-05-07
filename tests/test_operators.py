import pytest
import numpy as np

from diatomic.operators import (
    HalfInt,
    T2_C,
    electric_gradient,
    generate_vecs,
    proj_iter,
    proj_iter_bounded,
    mn_crop_indices,
    num_proj_with_below,
    num_proj,
    sph_iter,
    tensor_nuclear,
    uncoupled_basis_pos,
    uncoupled_basis_iter,
    unit_ac_aniso,
    unit_ac_aniso_ellip,
    unit_ac_iso,
    unit_dipole_operator,
)


def test_constructor():
    hi = HalfInt(of=3)
    assert hi._double == 3

    hi2 = HalfInt(of=4)
    assert hi2 == 2

    with pytest.raises(TypeError):
        HalfInt(of="not an integer")

    with pytest.raises(TypeError):
        HalfInt(of=3.0)


def test_str():
    hi = HalfInt(of=3)
    assert str(hi) == "(3/2)"


def test_repr():
    hi = HalfInt(of=3)
    assert repr(hi) == "((3/2) : HalfInt)"


def test_eq():
    hi1 = HalfInt(of=3)
    hi2 = HalfInt(of=3)

    assert hi1 == hi2
    assert hi1 != HalfInt(of=2)
    assert hi1 != HalfInt(of=5)
    assert hi1 == 1.5
    assert hi1 != 2
    assert hi1 != 2.0


def test_ordering():
    hi1 = HalfInt(of=3)
    hi2 = HalfInt(of=4)

    assert hi1 < hi2
    assert hi1 <= hi2
    assert hi2 > hi1
    assert hi2 >= hi1

    assert hi1 < 2
    assert hi1 <= 2
    assert hi2 > 1
    assert hi2 >= 1


def test_numeric_conversion():
    assert float(HalfInt(of=-3)) == -1.5
    assert float(HalfInt(of=5)) == 2.5
    assert float(HalfInt(of=6)) == 3.0
    assert int(HalfInt(of=-3)) == int(-3 / 2)
    assert int(HalfInt(of=3)) == int(3 / 2)
    assert int(HalfInt(of=4)) == 2
    assert abs(HalfInt(of=3)) == HalfInt(of=3)
    assert abs(HalfInt(of=-3)) == HalfInt(of=3)


def test_add():
    assert HalfInt(of=3) + HalfInt(of=4) == HalfInt(of=7)
    assert HalfInt(of=3) + HalfInt(of=5) == 4
    assert HalfInt(of=3) + 1 == HalfInt(of=5)
    assert HalfInt(of=3) + 1.5 == 3.0


def test_sub():
    assert HalfInt(of=3) - HalfInt(of=4) == HalfInt(of=-1)
    assert HalfInt(of=3) - HalfInt(of=3) == 0
    assert HalfInt(of=3) - 1 == HalfInt(of=1)
    assert HalfInt(of=3) - 0.5 == 1.0


def test_neg():
    assert -HalfInt(of=3) == HalfInt(of=-3)


def test_mul():
    assert HalfInt(of=3) * HalfInt(of=4) == 3.0
    assert HalfInt(of=3) * 2 == 3
    assert HalfInt(of=3) * 3 == HalfInt(of=9)
    assert HalfInt(of=3) * 4 == 6
    assert HalfInt(of=3) * 1.5 == 2.25


def test_num_proj():
    assert num_proj(3) == 7
    assert num_proj(3.0) == 7
    assert num_proj(HalfInt(of=6)) == 7

    assert num_proj(3.5) == 8
    assert num_proj(HalfInt(of=7)) == 8


def test_num_proj_with_below():
    assert num_proj_with_below(0) == 1
    assert num_proj_with_below(1) == 4
    assert num_proj_with_below(2) == 9

    assert num_proj_with_below(0.0) == 1
    assert num_proj_with_below(1.0) == 4
    assert num_proj_with_below(2.0) == 9

    assert num_proj_with_below(2, Nmin=1) == 8


def test_proj_iter():
    assert list(proj_iter(0)) == [0]
    assert list(proj_iter(2)) == [2, 1, 0, -1, -2]

    assert list(proj_iter(0.0)) == [0]
    assert list(proj_iter(2.0)) == [2, 1, 0, -1, -2]

    assert list(proj_iter(HalfInt(of=3))) == [
        HalfInt(of=3),
        HalfInt(of=1),
        HalfInt(of=-1),
        HalfInt(of=-3),
    ]


def test_proj_iter_bounded_snaps_to_physical_projection_ladder():
    assert list(proj_iter_bounded(2, mmax=0.5)) == [0, -1, -2]
    assert list(proj_iter_bounded(2, mmin=0.5)) == [2, 1]

    assert list(proj_iter_bounded(HalfInt(of=3), mmax=1)) == [
        HalfInt(of=1),
        HalfInt(of=-1),
        HalfInt(of=-3),
    ]
    assert list(proj_iter_bounded(HalfInt(of=3), mmin=-1)) == [
        HalfInt(of=3),
        HalfInt(of=1),
        HalfInt(of=-1),
    ]


def test_mn_crop_indices_with_half_integer_bounds_do_not_duplicate_indices():
    assert mn_crop_indices(2, 0, 0, MNmax=0.5).tolist() == [0, 2, 3, 6, 7, 8]


def test_sph_iter():
    assert list(sph_iter(1)) == [(0, 0), (1, 1), (1, 0), (1, -1)]
    assert list(sph_iter(2, Nmin=1)) == [
        (1, 1),
        (1, 0),
        (1, -1),
        (2, 2),
        (2, 1),
        (2, 0),
        (2, -1),
        (2, -2),
    ]


def test_uncoupled_basis_pos():
    I1, I2 = 3, HalfInt(of=7)
    N, MN, MI1, MI2 = 1, 0, 1, HalfInt(of=-1)
    pos = uncoupled_basis_pos(N, MN, MI1, MI2, I1, I2, Nmin=0)
    manual_pos = 0
    for lN, lMN in sph_iter(3):
        for lMI1 in proj_iter(I1):
            for lMI2 in proj_iter(I2):
                if (N, MN, MI1, MI2) == (lN, lMN, lMI1, lMI2):
                    assert manual_pos == pos
                    return
                manual_pos += 1


def test_old_operator_signatures_match_keyword_nmin_zero():
    I1, I2, Nmax = 0, 0, 1

    assert list(sph_iter(Nmax)) == list(sph_iter(Nmax, Nmin=0))
    assert list(uncoupled_basis_iter(Nmax, I1, I2)) == list(
        uncoupled_basis_iter(Nmax, I1, I2, Nmin=0)
    )
    assert uncoupled_basis_pos(1, 0, 0, 0, I1, I2) == uncoupled_basis_pos(
        1, 0, 0, 0, I1, I2, Nmin=0
    )
    assert num_proj_with_below(Nmax) == num_proj_with_below(Nmax, Nmin=0)

    old_vecs = generate_vecs(Nmax, I1, I2)
    new_vecs = generate_vecs(Nmax, I1, I2, Nmin=0)
    for old, new in zip(old_vecs, new_vecs):
        np.testing.assert_allclose(old, new)

    np.testing.assert_allclose(T2_C(Nmax, I1, I2), T2_C(Nmax, I1, I2, Nmin=0))
    np.testing.assert_allclose(
        unit_dipole_operator(Nmax, 0), unit_dipole_operator(Nmax, 0, Nmin=0)
    )
    np.testing.assert_allclose(electric_gradient(Nmax), electric_gradient(Nmax, Nmin=0))
    np.testing.assert_allclose(
        unit_ac_iso(Nmax, I1, I2), unit_ac_iso(Nmax, I1, I2, Nmin=0)
    )
    np.testing.assert_allclose(
        unit_ac_aniso(Nmax, I1, I2, 0), unit_ac_aniso(Nmax, I1, I2, 0, Nmin=0)
    )
    np.testing.assert_allclose(
        unit_ac_aniso_ellip(Nmax, I1, I2, 0, 0, 0),
        unit_ac_aniso_ellip(Nmax, I1, I2, 0, 0, 0, Nmin=0),
    )
    np.testing.assert_allclose(
        tensor_nuclear(1, old_vecs[1], old_vecs[2], I1, I2, Nmax),
        tensor_nuclear(1, old_vecs[1], old_vecs[2], I1, I2, Nmax, Nmin=0),
    )
