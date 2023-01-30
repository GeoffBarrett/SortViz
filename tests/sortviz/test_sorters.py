from typing import List, Optional

import numpy as np
import pytest
import pytest_mock

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
