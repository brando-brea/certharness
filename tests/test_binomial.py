import pytest

from certharness.stats.binomial import binom_cdf, clopper_pearson_upper, n_required


def test_binom_cdf():
    assert binom_cdf(0, 3, 0.5) == pytest.approx(0.125)
    assert binom_cdf(1, 3, 0.5) == pytest.approx(0.5)
    assert binom_cdf(3, 3, 0.5) == 1.0
    assert binom_cdf(-1, 3, 0.5) == 0.0


def test_clopper_pearson_known_values():
    # k=0 closed form: 1 - alpha**(1/n)
    assert clopper_pearson_upper(0, 149) == pytest.approx(1 - 0.05 ** (1 / 149))
    assert clopper_pearson_upper(0, 149) <= 0.02 < clopper_pearson_upper(0, 148)
    # 1 event in 10, one-sided 95%: Beta(0.95; 2, 9) = 0.3942
    assert clopper_pearson_upper(1, 10) == pytest.approx(0.3942, abs=1e-4)
    assert clopper_pearson_upper(5, 5) == 1.0
    with pytest.raises(ValueError):
        clopper_pearson_upper(3, 2)


def test_upper_bound_is_conservative():
    for k, n in [(1, 50), (3, 200), (10, 100)]:
        assert clopper_pearson_upper(k, n) > k / n


def test_n_required():
    assert n_required(0.02) == 149  # rule of three: ~3 / 0.02
    n1 = n_required(0.02, misses=1)
    assert clopper_pearson_upper(1, n1) <= 0.02 < clopper_pearson_upper(1, n1 - 1)
