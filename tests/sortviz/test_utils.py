import numpy as np
import pytest

import sortviz.utils as m_utils


@pytest.mark.parametrize("num_rows, num_cols", [(100, 100), (5, 10), (13, 5)])
def test_generate_data(num_rows: int, num_cols: int):
    """Assert that the data is genrated with the correct size and contains the correct values.

    :param num_rows: The number of rows for the generated data.
    :type num_rows: int
    :param num_cols: The number of columns for the generated data.
    :type num_cols: int
    """

    results = m_utils.generate_data(num_rows, num_cols)

    assert results.shape == (num_rows, num_cols)

    values = np.arange(num_cols)
    for result_row in results:
        assert np.array_equal(np.sort(result_row), values)
