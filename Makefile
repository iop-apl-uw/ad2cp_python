# For source checking, testing

all: rufffmt rufflint mypy test

rufflint:
	-ruff check .

rufffmt:
	-ruff check --select I --fix *py tests/*py BasePlotting/*py
	-ruff format *py tests/*py BasePlotting/*py

mypy:
	-mypy

test:
	-pytest --cov --cov-report term-missing tests/

testhtml:
	-pytest --cov --cov-report html tests/

# Requires act tool to be installed
# For MacOS
# brew install act
# Runs github workflow locally
act:
	-act -j check --container-daemon-socket -  --container-architecture linux/aarch64 push
