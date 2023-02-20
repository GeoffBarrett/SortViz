import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pydantic.main


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


def plot_sort(filename: str) -> None:
    """A method for plotting the sorted data by iteration/pass

    :param filename: The name of the .json file that contains the sort data.
    :type filename: str
    :raises FileNotFoundError: raised when the filename does not exist.
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

    for sort_pass in plot_data:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(sort_pass)
