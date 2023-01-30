import pathlib
from typing import List, Optional

import numpy as np
import pytest

from sortviz import sorters as m_sorters


@pytest.mark.parametrize(
    "attribute, expected_value",
    [
        ("arr", None),
        ("num_rows", None),
        ("num_cols", None),
        ("is_row_completed", None),
        ("passes", []),
    ],
)
def test_bubble_sort_init(attribute: str, expected_value: Optional[List]):
    """Test that `BubbleSorter` will initialize properly.

    Assert that the attributes of the instantiated `BubbleSorter` matches as expected.

    :param attribute: The attribute to test the value of.
    :type attribute: str
    :param expected_value: The expected value associated with the `attribute`
    :type expected_value: Optional[List]
    """

    sorter = m_sorters.BubbleSorter()
    assert hasattr(sorter, attribute)
    assert getattr(sorter, attribute) == expected_value


@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (10, 10),
        (150, 100),
        (1, 15),
    ],
)
def test_bubble_sorter_sort(gen_random_data_fixture, n_rows: int, n_cols: int):
    """Tests that the results from the `BubbleSorter` class work as expected.

    :param gen_random_data_fixture: A fixture for generating random data
    :param n_rows: The number of rows for the test data.
    :type n_rows: int
    :param n_cols: The number of columns for the test data.
    :type n_cols: int
    """

    data = gen_random_data_fixture(n_rows, n_cols)

    sorter = m_sorters.BubbleSorter()
    result = sorter.sort(data.copy())

    for idx_row in np.arange(data.shape[0]):
        expected_result = np.sort(data[idx_row])
        assert np.array_equal(result[idx_row], expected_result)


@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (10, 10),
        (150, 100),
        (1, 15),
    ],
)
def test_bubble_sorter__bubble_sort(gen_random_data_fixture, n_rows: int, n_cols: int):
    """Tests that the results from the `BubbleSorter` class work as expected (using `_bubble_sort()`).

    :param gen_random_data_fixture: A fixture for generating random data
    :param n_rows: The number of rows for the test data.
    :type n_rows: int
    :param n_cols: The number of columns for the test data.
    :type n_cols: int
    """

    data = gen_random_data_fixture(n_rows, n_cols)

    sorter = m_sorters.BubbleSorter()
    sorter._init_params(data)
    result = sorter._bubble_sort()

    for idx_row in np.arange(data.shape[0]):
        expected_result = np.sort(data[idx_row])
        assert np.array_equal(result[idx_row], expected_result)


def test_bubble_sorter__bubble_sort_not_init():
    """Tests that the results from the `BubbleSorter` class fails when `_bubble_sort()` is called
    without using `sort()`.

    :param gen_random_data_fixture: A fixture for generating random data
    :param n_rows: The number of rows for the test data.
    :type n_rows: int
    :param n_cols: The number of columns for the test data.
    :type n_cols: int
    """

    sorter = m_sorters.BubbleSorter()
    with pytest.raises(ValueError):
        sorter._bubble_sort()


@pytest.mark.parametrize(
    "data, expected_result",
    [
        (np.array([[5, 1, 4, 2, 8]]), np.array([[1, 4, 2, 5, 8]])),
        (np.array([[1, 4, 2, 5, 8]]), np.array([[1, 2, 4, 5, 8]])),
        (np.array([[1, 2, 4, 5, 8]]), np.array([[1, 2, 4, 5, 8]])),
    ],
)
def test_bubble_sorter_sort_pass(data: np.ndarray, expected_result: np.ndarray):
    """Tests that the `sort_pass()` method will return an individual pass for the Bubble Sort
    method.

    References: https://en.wikipedia.org/wiki/Bubble_sort#Step-by-step_example

    :param gen_random_data_fixture: A fixture for generating random data
    :param n_rows: The number of rows for the test data.
    :type n_rows: int
    :param n_cols: The number of columns for the test data.
    :type n_cols: int
    """
    sorter = m_sorters.BubbleSorter()

    # initialize
    sorter._init_params(data)

    # sort passes
    result = sorter.sort_pass(data)
    assert np.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (10, 10),
        (150, 100),
        (1, 15),
    ],
)
def test_bubble_sorter_sort_pass_value_error(gen_random_data_fixture, n_rows: int, n_cols: int):
    """Tests that a ValueError will be raised if `sort()` is not called before `sort_pass()`.

    :param gen_random_data_fixture: A fixture for generating random data
    :param n_rows: The number of rows for the test data.
    :type n_rows: int
    :param n_cols: The number of columns for the test data.
    :type n_cols: int
    """

    data = gen_random_data_fixture(n_rows, n_cols)
    sorter = m_sorters.BubbleSorter()
    with pytest.raises(ValueError):
        sorter.sort_pass(data)


def test_bubble_sorter_save_sort_value_error(tmp_path: pathlib.PosixPath):
    """Tests that a ValueError will be raised if `sort()` is not called before `save_sort()`.

    :param tmp_path: _description_
    :type tmp_path: pathlib.PosixPath
    """

    test_filename = tmp_path / "sorted_data.json"
    sorter = m_sorters.BubbleSorter()
    with pytest.raises(ValueError):
        sorter.save_sort(str(test_filename))
