import json
import os
import pathlib
from typing import List

import matplotlib
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
                comparisons=m_utils.ComparisonModel(comparisons={0: 10, 1: 20}),
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
    matplotlib.use("Agg")
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
    matplotlib.use("Agg")
    parse_spy = mocker.spy(m_utils.SortDataModel, "parse_file")
    plot_mock = mocker.patch.object(plt.Axes, "imshow")

    filename = "this_file_does_not_exist"
    assert not os.path.exists(filename)

    with pytest.raises(FileNotFoundError):
        m_utils.plot_sort(filename)

    parse_spy.assert_not_called()
    plot_mock.assert_not_called()


@pytest.mark.parametrize(
    "data, filenames",
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
                comparisons=m_utils.ComparisonModel(comparisons={0: 10, 1: 20}),
            ),
            ["fun_file2.txt"],
        )
    ],
)
def test_plot_sort_comparison(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
    data: m_utils.SortDataModel,
    filenames: List[str],
) -> None:
    """Tests that the plot_sort_comparison will create a bar plot with the provided averages.

    :param mocker: a mocker test fixture.
    :type mocker: pytest_mock.MockerFixture
    :param tmp_path: a temp path to save data to.
    :type tmp_path: pathlib.Path
    :param data: the data to plot
    :type data: m_utils.SortDataModel
    :param filenames: the filenames containing data to compare.
    :type filenames: List[str]
    """
    matplotlib.use("Agg")
    sort_filenames = [str(tmp_path / _file) for _file in filenames]
    parse_spy = mocker.spy(m_utils.SortDataModel, "parse_file")
    plot_mock = mocker.patch.object(plt.Axes, "bar")
    num_sorters = len(filenames)
    # create temp data
    for sort_filename in sort_filenames:
        with open(sort_filename, "w") as f:
            json.dump(data.dict(), f, ensure_ascii=True)

    m_utils.plot_sort_comparison(sort_filenames)

    parse_spy.assert_called_once_with(sort_filenames[0])
    plot_mock.assert_called_once()

    assert np.array_equal(plot_mock.call_args.args[0], np.arange(num_sorters))
    assert np.array_equal(plot_mock.call_args.args[1], [np.mean([10, 20])])  # averages
