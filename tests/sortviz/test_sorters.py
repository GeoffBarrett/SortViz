import pathlib
from collections import defaultdict
from typing import Dict, List, Optional

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
    """Tests that a ValueError will be raised if `BubbleSorter.sort()` is not called before
    `save_sort()`.

    :param tmp_path: The temporary directory to test with.
    :type tmp_path: pathlib.PosixPath
    """

    test_filename = tmp_path / "sorted_data.json"
    sorter = m_sorters.BubbleSorter()
    with pytest.raises(ValueError):
        sorter.save_sort(str(test_filename))


@pytest.mark.parametrize(
    "attribute, expected_value",
    [
        ("arr", None),
        ("num_rows", None),
        ("num_cols", None),
        ("is_row_completed", None),
        ("passes", defaultdict(list)),
    ],
)
def test_merge_sort_init(attribute: str, expected_value: Optional[List]):
    """Test that `MergeSorter` will initialize properly.

    Assert that the attributes of the instantiated `MergeSorter` matches as expected.

    :param attribute: The attribute to test the value of.
    :type attribute: str
    :param expected_value: The expected value associated with the `attribute`
    :type expected_value: Optional[List]
    """

    sorter = m_sorters.MergeSorter()
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
def test_merge_sorter_sort(gen_random_data_fixture, n_rows: int, n_cols: int):
    """Tests that the results from the `MergeSorter` class work as expected.

    :param gen_random_data_fixture: A fixture for generating random data
    :param n_rows: The number of rows for the test data.
    :type n_rows: int
    :param n_cols: The number of columns for the test data.
    :type n_cols: int
    """

    data = gen_random_data_fixture(n_rows, n_cols)

    sorter = m_sorters.MergeSorter()
    result = sorter.sort(data.copy())

    for idx_row in np.arange(data.shape[0]):
        expected_result = np.sort(data[idx_row])
        assert np.array_equal(result[idx_row], expected_result)


@pytest.mark.parametrize(
    "num_divide, arr, left_index, right_index",
    [
        (10, None, 0, 3),
        (0, np.array([1, 2, 3, 4]), None, 3),
        (2, np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), 1, None),
        (3, None, None, None),
    ],
)
def test_merge_sort__mergesort_value_error(
    num_divide: int,
    arr: Optional[np.ndarray],
    left_index: Optional[int],
    right_index: Optional[int],
):
    """Tests that the `_mergesort()` method will fail if any of the inputs are `None` (aside from
    `num_divide`).

    :param num_divide: The numbers of divisions made.
    :type num_divide: int
    :param arr: An array (or sub-division of the array) being sorted.
    :type arr: Optional[np.ndarray]
    :param left_index: The left index of the array to include in the Merge Sort.
    :type left_index: Optional[int]
    :param right_index: The right index of the array to include in the Merge Sort.
    :type right_index: Optional[int]
    :return: The sorted array.
    :rtype: np.ndarray
    """

    sorter = m_sorters.MergeSorter()
    with pytest.raises(ValueError):
        sorter._mergesort(num_divide, arr, left_index, right_index)


def test_merge_sorter_save_sort_value_error(tmp_path: pathlib.PosixPath):
    """Tests that a ValueError will be raised if `MergeSorter.sort()` is not called before
    `save_sort()`.

    :param tmp_path: The temporary directory to test with.
    :type tmp_path: pathlib.PosixPath
    """

    test_filename = tmp_path / "sorted_data.json"
    sorter = m_sorters.MergeSorter()
    with pytest.raises(ValueError):
        sorter.save_sort(str(test_filename))


@pytest.mark.parametrize(
    "attribute, expected_value",
    [
        ("arr", None),
        ("num_rows", None),
        ("num_cols", None),
        ("is_row_completed", None),
        ("passes", defaultdict(list)),
    ],
)
def test_quick_sort_init(attribute: str, expected_value: Optional[List]):
    """Test that `QuickSorter` will initialize properly.

    Assert that the attributes of the instantiated `QuickSorter` matches as expected.

    :param attribute: The attribute to test the value of.
    :type attribute: str
    :param expected_value: The expected value associated with the `attribute`
    :type expected_value: Optional[List]
    """

    sorter = m_sorters.QuickSorter()
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
def test_quick_sorter_sort(gen_random_data_fixture, n_rows: int, n_cols: int):
    """Tests that the results from the `QuickSorter` class work as expected.

    :param gen_random_data_fixture: A fixture for generating random data
    :param n_rows: The number of rows for the test data.
    :type n_rows: int
    :param n_cols: The number of columns for the test data.
    :type n_cols: int
    """

    data = gen_random_data_fixture(n_rows, n_cols)

    sorter = m_sorters.QuickSorter()
    result = sorter.sort(data.copy())

    for idx_row in np.arange(data.shape[0]):
        expected_result = np.sort(data[idx_row])
        assert np.array_equal(result[idx_row], expected_result)


@pytest.mark.parametrize(
    "num_divide, idx_row, arr, left_index, right_index",
    [
        (10, 1, None, 0, 3),
        (0, 2, np.array([1, 2, 3, 4]), None, 3),
        (2, 3, np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), 1, None),
        (3, 4, None, None, None),
    ],
)
def test_quick_sort__quicksort_value_error(
    num_divide: int,
    idx_row: int,
    arr: Optional[np.ndarray],
    left_index: Optional[int],
    right_index: Optional[int],
):
    """Tests that the `_quicksort()` method will fail if any of the inputs are `None` (aside from
    `num_divide`).

    :param num_divide: The numbers of divisions made.
    :type num_divide: int
    :param arr: An array (or sub-division of the array) being sorted.
    :type arr: Optional[np.ndarray]
    :param left_index: The left index of the array to include in the Merge Sort.
    :type left_index: Optional[int]
    :param right_index: The right index of the array to include in the Merge Sort.
    :type right_index: Optional[int]
    :return: The sorted array.
    :rtype: np.ndarray
    """

    sorter = m_sorters.QuickSorter()
    with pytest.raises(ValueError):
        sorter._quicksort(num_divide, idx_row, arr, left_index, right_index)


def test_quick_sorter_save_sort_value_error(tmp_path: pathlib.PosixPath):
    """Tests that a ValueError will be raised if `QuickSorter.sort()` is not called before
    `save_sort()`.

    :param tmp_path: The temporary directory to test with.
    :type tmp_path: pathlib.PosixPath
    """

    test_filename = tmp_path / "sorted_data.json"
    sorter = m_sorters.QuickSorter()
    with pytest.raises(ValueError):
        sorter.save_sort(str(test_filename))


@pytest.mark.parametrize(
    "data, expected_passes",
    [
        (
            np.array([[5, 1, 4, 2, 8]]),
            {1: [[2, 1, 4, 5, 8]], 2: [[1, 2, 4], [5, 8]], 3: [[2, 4]]},
        )
    ],
)
def test_quick_sort_passes(data: np.ndarray, expected_passes: Dict[int, List[List[int]]]):
    """Look at the divide and conquer specific data held in the `QuickSorter.passes`.

    Note: `expected_passes` was manually created following:
    https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme

    :param data: The data being sorted
    :type data: np.ndarray
    :param expected_passes: The expected divide and conquer results.
    :type expected_passes: Dict[int, List[List[int]]]
    """
    sorter = m_sorters.QuickSorter()
    sorter.sort(data.copy())

    assert sorter.passes is not None

    for num_divide, pass_values in expected_passes.items():
        sort_passes = [_pass_values[0].astype(int) for _pass_values in sorter.passes[num_divide]]
        assert len(sort_passes) == len(pass_values)
        for values in pass_values:
            values = np.array(values, dtype=int)
            assert any([np.array_equal(values, _pass_values) for _pass_values in sort_passes])
