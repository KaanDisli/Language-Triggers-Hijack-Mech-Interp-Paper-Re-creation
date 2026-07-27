from fractions import Fraction
import math

import pytest

from trigger_heads.metrics import (
    cosine_similarity,
    expected_jaccard,
    expected_jaccard_fraction,
    hypergeometric_upper_tail,
    hypergeometric_upper_tail_fraction,
    jaccard,
    jaccard_p_value,
    pairwise_jaccard_matrix,
    rank_top_heads,
)


def test_signed_top_head_ranking_and_tie_breaking():
    scores = [[-100.0, 0.5], [0.5, 0.2]]
    assert rank_top_heads(scores, 3) == [(0, 1), (1, 0), (1, 1)]
    assert rank_top_heads({(3, 2): 2.0, (1, 4): 2.0}, 2) == [(1, 4), (3, 2)]


def test_jaccard_and_pairwise_cross_matrix():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert jaccard([], []) == 1.0
    matrix = pairwise_jaccard_matrix(
        {"fr": {1, 2}, "de": {2, 3}}, {"trigger": {2}}
    )
    assert matrix == [[0.5], [0.5]]


def test_expected_jaccard_is_exact_hypergeometric_sum():
    # For two size-2 subsets of a four-element universe, X has masses 1/6,4/6,1/6.
    expected = Fraction(4, 6) * Fraction(1, 3) + Fraction(1, 6)
    assert expected_jaccard_fraction(4, 2) == expected
    assert expected_jaccard(4, 2) == pytest.approx(float(expected))


def test_hypergeometric_upper_tail_exact_and_jaccard_conversion():
    # P(X >= 1) = 5/6 for two size-2 subsets from four elements.
    assert hypergeometric_upper_tail_fraction(4, 2, 1) == Fraction(5, 6)
    assert hypergeometric_upper_tail(4, 2, 1) == pytest.approx(5 / 6)
    assert jaccard_p_value(4, 2, 1 / 3) == pytest.approx(5 / 6)


def test_paper_scale_baseline_and_significance():
    assert expected_jaccard(512, 10) == pytest.approx(0.0103708, rel=1e-5)
    assert expected_jaccard(1024, 10) < 0.01
    assert expected_jaccard(1280, 10) == pytest.approx(0.00412635, rel=1e-5)
    assert hypergeometric_upper_tail(512, 10, 5) == pytest.approx(2.119e-7, rel=1e-3)
    assert hypergeometric_upper_tail(1024, 10, 5) == pytest.approx(6.696e-9, rel=1e-3)


def test_cosine_similarity_and_input_errors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_similarity([0, 0], [1, 0])
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1], [1, 2])


def test_ranking_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        rank_top_heads([[math.nan]], 1)
