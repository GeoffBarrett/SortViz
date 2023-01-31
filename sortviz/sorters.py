import abc
import json
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Union

import numpy as np

from sortviz import utils as m_utils


class BaseSorter(abc.ABC):
    """An Abstract Base Class that all of the sorters will inherit from."""

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

    @abc.abstractmethod
    def sort(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The 'sort()' method is not implemented.")

    @abc.abstractmethod
    def save_sort(self, filename: str) -> None:
        raise NotImplementedError("The 'save_sort()' method is not implemented.")


class BubbleSorter(BaseSorter):
    """A class that will perform the Bubble Sort method."""

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


class MergeSorter(BaseSorter):
    def __init__(self):
        super().__init__()
        self.passes: DefaultDict = defaultdict(list)

    def _init_params(self, arr: Optional[np.ndarray]) -> None:
        super()._init_params(arr)
        self.passes: DefaultDict = defaultdict(list)

    def sort(self, arr: np.ndarray) -> np.ndarray:
        """Sorts the provided `arr` using Merge Sort.

        :param arr: The array to sort.
        :type arr: np.ndarray
        :return: The sorted array.
        :rtype: np.ndarray
        """

        self._init_params(arr)
        return self._mergesort(0, arr.copy(), 0, self.num_cols)

    def _merge(self, left_arr: np.ndarray, right_arr: np.ndarray) -> np.ndarray:
        """Merges the two halves (`left_arr` and `right_arr`) in a sorted manner.

        :param left_arr: The left array to merge.
        :type left_arr: np.ndarray
        :param right_arr: The right array to merge.
        :type right_arr: np.ndarray
        :return: The merged (and sorted) array.
        :rtype: np.ndarray
        """
        cols = left_arr.shape[1] + right_arr.shape[1]
        rows = left_arr.shape[0]
        tmp_arr = np.zeros((rows, cols))

        # sort each row separately
        for idx_row in np.arange(rows):
            idx_left = 0
            idx_right = 0
            idx_temp = 0
            while idx_temp < cols:
                left_value = left_arr[idx_row, idx_left] if idx_left < left_arr.shape[1] else None
                right_value = (
                    right_arr[idx_row, idx_right] if idx_right < right_arr.shape[1] else None
                )

                if left_value is None:
                    tmp_arr[idx_row, idx_temp] = right_value
                    idx_right += 1
                elif right_value is None:
                    tmp_arr[idx_row, idx_temp] = left_arr[idx_row, idx_left]
                    idx_left += 1
                elif left_arr[idx_row, idx_left] < right_arr[idx_row, idx_right]:
                    tmp_arr[idx_row, idx_temp] = left_arr[idx_row, idx_left]
                    idx_left += 1
                else:
                    tmp_arr[idx_row, idx_temp] = right_arr[idx_row, idx_right]
                    idx_right += 1
                idx_temp += 1
        return tmp_arr

    def _mergesort(
        self,
        num_divide: int,
        arr: Optional[np.ndarray],
        left_index: Optional[int],
        right_index: Optional[int],
    ) -> np.ndarray:
        """Recursively divides the array until the size becomes one.

        :param num_divide: The numbers of divisions made.
        :type num_divide: int
        :param arr: An array (or sub-division of the array) being sorted.
        :type arr: Optional[np.ndarray]
        :param left_index: The left index of the array to include in the Merge Sort.
        :type left_index: Optional[int]
        :param right_index: The right index of the array to include in the Merge Sort.
        :type right_index: Optional[int]
        :raises ValueError: Raised when `sort()` was not called beforehand.
        :return: The sorted array.
        :rtype: np.ndarray
        """

        if arr is None or left_index is None or right_index is None:
            raise ValueError("A numpy array is not set, please call the `sort()` method.")

        if arr.shape[1] <= 1:
            self.passes[num_divide].append([arr, left_index, right_index])
            return arr

        mid = arr.shape[1] // 2
        pointer = (left_index + right_index) // 2
        left_arr = arr[:, :mid].copy()  # divide and conquer the left side of the array.
        right_arr = arr[:, mid:].copy()  # divide and conquer the right side of the array.

        left_arr = self._mergesort(num_divide + 1, left_arr, left_index, pointer)
        right_arr = self._mergesort(num_divide + 1, right_arr, pointer, right_index)
        sorted_arr = self._merge(left_arr, right_arr)
        self.passes[num_divide].append([sorted_arr, left_index, right_index])
        return sorted_arr

    def save_sort(self, filename: str) -> None:
        """Saves the sorted data into a .json file that makes it easy to plot using d3.

        :param filename: The .json filename to save the data to.
        :type filename: str
        :raises ValueError: Raised when the data has not been sorted.
        """

        if self.arr is None:
            raise ValueError("Unable to save sort, call the `sort()` before attempting to save.")

        rows, cols = self.arr.shape

        # initialize with the starting array as num_pass = 0
        passes = [
            [
                m_utils.PassDataModel(num_pass=0, row=row, col=col, value=int(self.arr[row][col]))
                for row in range(rows)
                for col in range(cols)
            ]
        ]
        sorted_arr = self.arr.copy()

        for idx_divide in self.passes:
            num_pass = len(self.passes) - idx_divide  # lower num pass equates to early in the sort
            for (data, left_index, right_index) in self.passes[idx_divide]:
                sorted_arr[:, left_index:right_index] = data
            current_pass = [
                m_utils.PassDataModel(
                    num_pass=num_pass, row=row, col=col, value=int(sorted_arr[row][col])
                )
                for row in range(rows)
                for col in range(cols)
            ]
            passes.append(current_pass)

        data = m_utils.SortDataModel(method="Merge Sort", rows=rows, cols=cols, passes=passes)

        with open(filename, "w") as f:
            json.dump(
                data.dict(),
                f,
            )
