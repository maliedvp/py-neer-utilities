Rebuild the package to include the version information:

	python -m build

Install the package locally for testing:

	pip install dist/neer_match_utilities-0.1.0a0-py3-none-any.whl

Verify the version:

	python -c "import neer_match_utilities; print(neer_match_utilities.__version__)"


If no new build wanted:

1. change python script
2. run

	pip install -e .



# Testing

	pytest tests/test_prepare.py -v
	pytest tests/test_training.py -v


# Documentation

from `docs/` directory, to generate the `source/` folder and `make.bat` as well as `Makefile`:

	sphinx-quickstart
	sphinx-apidoc -f -o source/ ../src   

and to create the `build/` folder in `docs/`

	make html      

