import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def random_state_fixture() -> None:
    """A session test fixture that will set the random seed."""
    np.random.seed(42)


@pytest.fixture(scope="session")
def gen_random_data_fixture(random_state_fixture):
    def _gen_random_data(num_rows: int, num_cols: int) -> np.ndarray:
        """Generate random data row-wise.

        :param num_rows: The number of rows for the generated data.
        :type num_rows: int
        :param num_cols: The number of columns for the generated data.
        :type num_cols: int
        :return: The random array.
        :rtype: np.ndarray
        """
        values = np.arange(num_cols)
        data = np.zeros((num_rows, num_cols), dtype=np.int16)
        for idx_row in np.arange(num_rows):
            np.random.shuffle(values)  # in place shuffling of values to use
            data[idx_row] = values
        return data

    return _gen_random_data
