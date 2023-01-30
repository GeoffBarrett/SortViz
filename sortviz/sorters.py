import abc
import json
from typing import Dict, List, Optional, Union

import numpy as np

from sortviz import utils as m_utils


class BaseSorter(abc.ABC):
    """An Abstract Base Class that all of the sorters will inherit from."""

    @abc.abstractmethod
    def sort(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The 'sort()' method is not implemented.")

    @abc.abstractmethod
    def save_sort(self, filename: str) -> None:
        raise NotImplementedError("The 'save_sort()' method is not implemented.")


class BubbleSorter(BaseSorter):
    """A class that will perform the Bubble Sort method."""

    def __init__(self):
        """Initializes the parameters that will contain the sorting information for the class."""
        self.arr: Optional[np.ndarray] = None
        self.num_rows: Optional[int] = None
        self.num_cols: Optional[int] = None
        self.is_row_completed: Optional[np.ndarray] = None
        self.passes: List[np.ndarray] = []
        self._init_params(None)

    def _init_params(self, arr: Optional[np.ndarray]) -> None:
        """Initializes the sorting parameters based on the provided `arr`.

        :param arr: The optional array to use when initializing sorting parameters.
        :type arr: Optional[np.ndarray]
        """
        self.arr = arr
        if arr is None:
            self.num_rows = None
            self.num_cols = None
            self.is_row_completed = None
        else:
            self.num_rows, self.num_cols = arr.shape
            self.is_row_completed = np.zeros(self.num_rows)
        self.passes = []

    def sort(self, arr: np.ndarray) -> np.ndarray:
        """Performs the bubble sort algorithm on the provided `arr` array.

        This result from each pass will be cached in `self.passes` for visualization.

        :param arr: The matrix to sort (row-wise).
        :type arr: np.ndarray
        :return: The sorted array.
        :rtype: np.ndarray
        """
        self._init_params(arr)
        return self._bubble_sort()

    def _bubble_sort(self) -> np.ndarray:
        """Performs the bubble sort algorithm on `self._arr`.

        :raises ValueError: raised when this method was not called through `self.sort()`.
        :return: The sorted array.
        :rtype: np.ndarray
        """

        if self.arr is None:
            raise ValueError("A numpy array is not set, please call the `sort()` method.")

        sorted_arr = self.arr.copy()
        self.passes.append(sorted_arr.copy())
        while np.any(self.is_row_completed == 0):
            sorted_arr = self.sort_pass(sorted_arr)
            self.passes.append(sorted_arr.copy())
        return sorted_arr

    def sort_pass(self, arr: np.ndarray) -> np.ndarray:
        """Runs one pass through the value of each row and performs a bubble sort.

        This sorting method will iterate through each value and swap with adjanct
        indices in the case where the values are out of order.

        :param arr: The array to sort every row.
        :type arr: np.ndarray
        :raises ValueError: raised when this method was not called through `self.sort()`.
        :return: A sorted pass through the `arr` array. Note: this output will not be fully sorted
            until the final pass has been run.
        :rtype: np.ndarray
        """

        if self.arr is None or self.is_row_completed is None or self.num_cols is None:
            raise ValueError("A numpy array is not set, please call the `sort()` method.")

        # iterate over each row and sort
        for idx_row, row_data in enumerate(arr):
            if self.is_row_completed[idx_row] == 1:
                # already completed sort
                continue

            self.is_row_completed[idx_row] = 1
            # sort the row
            for idx_col in range(1, self.num_cols):
                if row_data[idx_col] > row_data[idx_col - 1]:
                    # no need to sort
                    continue
                self.is_row_completed[idx_row] = 0
                row_data[[idx_col - 1, idx_col]] = row_data[[idx_col, idx_col - 1]]
        return arr

    def save_sort(self, filename: str) -> None:
        """Saves the sorted data into a .json file that makes it easy to plot using d3.

        :param filename: The .json filename to save the data to.
        :type filename: str
        :raises ValueError: raised when `sort()` has not been called yet!
        """

        if self.arr is None:
            raise ValueError("Unable to save sort, call the `sort()` before attempting to save.")

        passes: List[List[m_utils.PassDataModel]] = []
        rows, cols = self.arr.shape
        for (idx_pass, sort_pass) in enumerate(self.passes):
            current_pass = [
                m_utils.PassDataModel(
                    num_pass=idx_pass, row=row, col=col, value=int(sort_pass[row][col])
                )
                for row in np.arange(rows)
                for col in np.arange(cols)
            ]
            passes.append(current_pass)

        data = m_utils.SortDataModel(method="Bubble Sort", rows=rows, cols=cols, passes=passes)

        with open(filename, "w") as f:
            json.dump(
                data.dict(),
                f,
            )
