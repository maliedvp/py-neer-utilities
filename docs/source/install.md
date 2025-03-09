

# Installation

## PiPy

To install `py-neer-utilities` from PyPi, run:

``` bash
pip install neer-match
pip install neer-match-utilities
```

## From Source

Clone the repository, build the package, and install it:

``` bash
git clone https://github.com/maliedvp/py-neer-utilities
python -m build
python -m pip install dist/$(basename `ls -Art dist | tail -n 1` -py3-none-any.whl .tar.gz
```
