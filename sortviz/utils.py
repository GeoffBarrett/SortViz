import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.figure as m_figure
import numpy as np
import pydantic.class_validators
import pydantic.main


class ComparisonModel(pydantic.main.BaseModel):
    """A data-class containing the number of comparisons for the sorters."""

    comparisons: Dict[int, int]
    average: Optional[float] = None
    standard_dev: Optional[float] = None
    standard_error: Optional[float] = None

    @pydantic.class_validators.root_validator(pre=True)
    def validate_statistics(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Adds the statistics to the model.

        :param values: the model values, likely will only contain the number of comparisons.
        :type values: Dict[str, Any]
        :return: the updated comparison model (including the statistics).
        :rtype: Dict[str, Any]
        """
        comparisons = list(values["comparisons"].values())

        if values.get("average", None) is None:
            values["average"] = np.mean(comparisons)

        if values.get("standard_dev", None) is None:
            values["standard_dev"] = np.std(comparisons)

        if values.get("standard_error", None) is None:
            values["standard_error"] = values["standard_dev"] / np.sqrt(len(comparisons))

        return values


class PassDataModel(pydantic.main.BaseModel):
    """A model for containing data from the sorting passes."""

    num_pass: int
    row: int
    col: int
    value: int


class SortDataModel(pydantic.main.BaseModel):
    """A model for containing sorting data."""

    method: str
    rows: int
    cols: int
    passes: List[List[PassDataModel]]
    comparisons: ComparisonModel


def generate_data(num_rows: int, num_cols: int) -> np.ndarray:
    """Generates random data with `num_rows` rows and `num_cols` cols.

    :param num_rows: The number of rows for the generated data.
    :type num_rows: int
    :param num_cols: The number of cols for the generated data.
    :type num_cols: int
    :return: The randomly generated data.
    :rtype: np.ndarray
    """
    values = np.arange(num_cols)
    data = np.zeros((num_rows, num_cols), dtype=np.int16)
    for idx_row in np.arange(num_rows):
        np.random.shuffle(values)  # in place shuffling of values to use
        data[idx_row] = values
    return data


def plot_sort(filename: str) -> List[m_figure.Figure]:
    """A method for plotting the sorted data by iteration/pass

    :param filename: The name of the .json file that contains the sort data.
    :type filename: str
    :raises FileNotFoundError: raised when the filename does not exist.
    :return figures: the plot figures.
    :rtype: List[m_figure.Figure]
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Unable to plot the data, file does not exist: {filename}")

    sort_data = SortDataModel.parse_file(filename)
    rows = sort_data.rows
    cols = sort_data.cols
    passes = sort_data.passes
    plot_data = np.zeros((len(passes), rows, cols), dtype=int)

    for sort_pass in passes:
        for data in sort_pass:
            plot_data[data.num_pass, data.row, data.col] = data.value

    figures = []
    for sort_pass in plot_data:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(sort_pass)
        figures.append(fig)

    return figures


def plot_sort_comparison(
    filenames: List[str], figsize=(10, 10), colors=("r", "g", "b")
) -> m_figure.Figure:
    """A method for comparing the number of comparisons made per sorter.

    :param filenames: The name of the .json files that contains the sort data.
    :type filename: List[str]
    :raises FileNotFoundError: raised when a filename does not exist.
    :return fig: the plot figure.
    :rtype: m_figure.Figure
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    num_sorters = len(filenames)
    sorters = []
    averages = np.zeros(num_sorters)
    std_devs = np.zeros(num_sorters)
    std_errs = np.zeros(num_sorters)

    for idx_sorter, filename in enumerate(filenames):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Unable to plot the data, file does not exist: {filename}")

        sort_data = SortDataModel.parse_file(filename)
        comparisons = sort_data.comparisons

        sorters.append(sort_data.method)
        averages[idx_sorter] = comparisons.average
        std_devs[idx_sorter] = comparisons.standard_dev
        std_errs[idx_sorter] = comparisons.standard_error

    ax.bar(np.arange(num_sorters), averages, color=colors, yerr=std_devs)
    ax.set_xticks(np.arange(num_sorters))
    ax.set_xticklabels(sorters)
    ax.set_ylabel("Num Comparisons (#)")
    ax.set_xlabel("Sorters")
    ax.set_title("Sort Method Number of Comparisons")
    return fig
