# For source checking, testing

all: rufffmt rufflint typecheck test

rufflint:
	-uv run ruff check .

rufffmt:
	-uv run ruff check --select I --fix *py tests/*py
	-uv run ruff format *py tests/*py

typecheck:
	-uv run --extra ci ty check

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
