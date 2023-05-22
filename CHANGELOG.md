# Sortviz Changelog

- [0.3.0](https://github.com/GeoffBarrett/SortViz/pull/3) - 2023-05-21
  - *Added*
    - `num_comparisons` to the sorters to quantify performance differences in the sorters.
    - `ComparisonModel`: a data-model for containing the number of comparisons metrics.
  - *Modified*
    - `Dockerfile`: now has multi-stage build.
    - `README`: now contains gifs of the sort performance.

- [0.2.0](https://github.com/GeoffBarrett/SortViz/pull/1) - 2023-02-19
  - *Modified*
    - `CHANGELOG.md`: updated to contain the proper Pull Request links.
    - `pyproject.toml`: added more package meta-info.

- [0.1.0](https://github.com/GeoffBarrett/SortViz/pull/2) - 2023-02-19
  - *Added*
    - Initialized SortViz project
    - `BaseSorter`: a base class for all the sorters.
    - `BubbleSorter`: a sorter class that will perform the `Bubble Sort` method.
    - `MergeSorter`: a sorter class that will perform the `Merge Sort` method.
    - `QuickSorter`: a sorter class that will perform the `Quick Sort` method.
