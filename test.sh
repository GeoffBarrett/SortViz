#!/bin/bash
poetry install
rm .coverage
poetry run black --check sortviz tests
poetry run isort --check sortviz tests
poetry run pytype sortviz tests
poetry run pytest --cov=sortviz --cov-report term-missing
poetry run coverage-badge -o svg/coverage.svg -f
