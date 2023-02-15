import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pytest_mock

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


@pytest.mark.parametrize(
    "data, filename",
    [
        (
            m_utils.SortDataModel(
                method="test_method",
                rows=2,
                cols=2,
                passes=[
                    [
                        m_utils.PassDataModel(num_pass=0, row=0, col=0, value=1),
                        m_utils.PassDataModel(num_pass=0, row=0, col=1, value=2),
                    ]
                ],
            ),
            "fun_file.txt",
        )
    ],
)
def test_plot_sort(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
    data: m_utils.SortDataModel,
    filename: str,
):
    sort_filename = tmp_path / filename
    parse_spy = mocker.spy(m_utils.SortDataModel, "parse_file")
    plot_mock = mocker.patch.object(plt.Axes, "imshow")

    # create temp data
    with sort_filename.open("w") as f:
        json.dump(data.dict(), f, ensure_ascii=True)

    # expected data
    expected_data = np.array([[1, 2], [0, 0]], dtype=int)

    m_utils.plot_sort(str(sort_filename))

    parse_spy.assert_called_once_with(str(sort_filename))
    plot_mock.assert_called_once()
    assert np.array_equal(plot_mock.call_args.args[0], expected_data)


def test_plot_sort_invalid_filename(mocker: pytest_mock.MockerFixture):
    parse_spy = mocker.spy(m_utils.SortDataModel, "parse_file")
    plot_mock = mocker.patch.object(plt.Axes, "imshow")

    filename = "this_file_does_not_exist"
    assert not os.path.exists(filename)

    with pytest.raises(FileNotFoundError):
        m_utils.plot_sort(filename)

    parse_spy.assert_not_called()
    plot_mock.assert_not_called()
