# NMRDfromMD - Developer section

## How to contribute

If you intend to make modifications to the code, please raise an
issue or send an email first. Then fork the repository, apply
your changes, and submit a pull request for review.

## Development install

NMRDfromMD requires Python 3.12 or newer. For local development, install
the package in editable mode together with the development tools
(the `--group` option requires pip 25.1 or newer):

```bash
pip install -e . --group test --group lint
```

The `test` and `lint` dependency groups are defined in `pyproject.toml`
and are never installed for end users.

## Running the tests

From the repository root:

```bash
pytest
```

With a coverage report (written to the terminal and to `htmlcov/`):

```bash
pytest --cov --cov-report=term-missing --cov-report=html
```

The editable install matters here: with a regular install, the tests
import the copy in `site-packages` and coverage of `nmrdfrommd/` reads 0%.

## Linting and type checking

```bash
ruff check .
ruff format --check .
ty check
```

These checks run in CI on every push and pull request, but are not
blocking yet: findings are reported without failing the workflow.

## Continuous integration

The GitHub Actions workflow (`.github/workflows/python-app.yml`) runs:

- the test suite on Python 3.12 and on the newest stable Python, and
  uploads the coverage reports (`coverage.xml` and `htmlcov/`) as
  workflow artifacts named `coverage-py<version>`;
- ruff and ty, as a non-blocking job.

## Publish new pip version (1)

Publish a new pip version by following those
[instructions](https://gist.github.com/arsho/fc651bfadd8a0f42be72156fd21bd8a9).

1 - if necessary, update *docs/source/conf.py*, *CITATION.cff*, and *setup.py*

2 - Create source distribution using

``` bash
    python3 setup.py sdist
```

3 - Create a new release on GitHub using the generated tar.gz file located in dist/

4 - Update the link terminating with *tar.gz* in setup.py

5 - Create wheel using:

``` bash
    python3 setup.py bdist_wheel
```

6 - Upload to pypi using (with the appropriate number):

``` bash
    twine upload dist/nmrdfrommd-0.2.0*
```

## Publish new pip version (2)

This instruction are from [this page](https://gist.github.com/arsho/fc651bfadd8a0f42be72156fd21bd8a9).

Install last version using 

```bash
    pip install -e .
```

1 - Update version in conf.py

2 - Update CITATION.cff

3 - Update setup.py

4 - Update CHANGELOG

5 - Create source distribution using python3 setup.py sdist

6 - Create a new release on Github using the generated tar.gz file

7 - Update the link in setup.py

8 - Create wheel using python3 setup.py bdist_wheel

9 - Upload to pypi using twine upload dist/nmrdfrommd-0.1.0* \


