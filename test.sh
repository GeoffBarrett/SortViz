#!/bin/bash
poetry install
rm .coverage
poetry run black --check sortviz tests
poetry run isort --check sortviz tests
poetry run pytype .
poetry run pytest
poetry run coverage-badge -o svg/coverage.svg -f
