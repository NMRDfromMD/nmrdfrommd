# NMRDfromMD - Developer section

## How to

If you intend to make modification to the code, please raise an issue or send me an email
first. Then, fork the repository, apply your changes, then make a pull request
that will be reviewed.

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
    twine upload dist/nmrdfrommd-0.1.0*
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


