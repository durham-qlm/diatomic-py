import pytest

from diatomic.operators import (
    HalfInt,
    proj_iter,
    num_proj_with_below,
    num_proj,
    sph_iter,
    uncoupled_basis_pos,
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


def test_sph_iter():
    assert list(sph_iter(1)) == [(0, 0), (1, 1), (1, 0), (1, -1)]


def test_uncoupled_basis_pos():
    I1, I2 = 3, HalfInt(of=7)
    N, MN, MI1, MI2 = 1, 0, 1, HalfInt(of=-1)
    pos = uncoupled_basis_pos(N, MN, MI1, MI2, I1, I2)
    manual_pos = 0
    for lN, lMN in sph_iter(3):
        for lMI1 in proj_iter(I1):
            for lMI2 in proj_iter(I2):
                if (N, MN, MI1, MI2) == (lN, lMN, lMI1, lMI2):
                    assert manual_pos == pos
                    return
                manual_pos += 1
