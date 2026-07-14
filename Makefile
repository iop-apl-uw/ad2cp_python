# For source checking, testing

all: rufffmt rufflint typecheck test

rufflint:
	-uv run ruff check .

rufffmt:
	-uv run ruff check --select I --fix *py tests/*py
	-uv run ruff format *py tests/*py

typecheck:
	-uv run --extra ci ty check --output-format github 2>/dev/null
# BaseADCP.py, BaseADCPMission.py, DiveOcean.py and MissionOcean.py import BaseLog
# unconditionally and only make sense when this directory is checked out as a subdirectory
# of a basestation3 installation, so they're skipped unless that parent checkout is present
# (e.g. in the standalone CI checkout).
	-@if [ -f ../BaseLog.py ]; then \
		uv run --extra ci ty check --output-format github BaseADCP.py BaseADCPMission.py DiveOcean.py MissionOcean.py; \
	fi

test:
	-uv run pytest --cov --cov-report term-missing tests/

testpdb:
	-uv run pytest --pdb --cov --cov-report term-missing tests/

testhtml:
	-uv run pytest --cov --cov-report html tests/

# Requires act tool to be installed
# For MacOS
# brew install act
# Runs github workflow locally
act:
	-act -j check --container-daemon-socket -  --container-architecture linux/aarch64 push
