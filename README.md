Rebuild the package to include the version information:

	python -m build

Install the package locally for testing:

	pip install dist/neer_match_utilities-0.1.2a0-py3-none-any.whl

Verify the version:

	python -c "import neer_match_utilities; print(neer_match_utilities.__version__)"


If no new build wanted:

1. change python script
2. run

	pip install -e .



# Testing

	pytest tests/test_prepare.py -v
	pytest tests/test_training.py -v
	pytest tests/test_split.py -v -s
	pytest tests/test_split_duplicates.py -v -s


# Documentation

from `docs/` directory, to generate the `source/` folder and `make.bat` as well as `Makefile`:

	sphinx-quickstart
	sphinx-apidoc -f -o source/ ../src   

and to create the `build/` folder in `docs/`

	make html      



# panel_setup.py


	## Example 1 (simple -> no panel_id & no overlap)
	df_1 = pd.DataFrame(
		{
		'id': [1,2,3,4,5,6,7,8],
		'col': ['a','b','c','d','a','b','c','d']
		}
	)

	matches_1 = [
		(1,5),
		(2,6),
		(3,7),
		(4,8),
	]

	df_left, df_right, df_matches = SetupData(
		matches =matches_1
	).data_preparation(
			df_panel = df_1,
			unique_id = 'id'
		)

	# print(f'df_left:\n{df_left}','\n')
	# print(f'df_right:\n{df_right}','\n')
	print(f'df_matches:\n{df_matches}','\n')


	## Example 2 (medium -> no panel_id & but overlap)
	df_2 = pd.DataFrame(
		{
		'id': [1,2,3,7,5,6,7,8],
		'col': ['a','b','c','c','a','b','c','c']
		}
	)

	matches_2 = [
		(1,5),
		(2,6),
		(3,7),
		(7,8),
	]

	df_left, df_right, df_matches = SetupData(
		matches =matches_2
	).data_preparation(
			df_panel = df_2,
			unique_id = 'id'
		)

	# print(f'df_left:\n{df_left}','\n')
	# print(f'df_right:\n{df_right}','\n')
	print(f'df_matches:\n{df_matches}','\n')


	## Example 3 (medium hard -> panel_id but no overlap)
	df_3 = pd.DataFrame(
		{
		'pid' : [10,10,20,20,30,30,40,40],
		'id': [1,2,3,4,5,6,7,8],
		'col': ['a','a','b','b','c','c','d','d']
		}
	)

	df_left, df_right, df_matches = SetupData().data_preparation(
		df_panel = df_3,
		unique_id = 'id',
		panel_id = 'pid'
	)

	# print(f'df_left:\n{df_left}','\n')
	# print(f'df_right:\n{df_right}','\n')
	print(f'df_matches:\n{df_matches}','\n')


	## Example 4 (extreme hard -> panel_id & overlap)
	df_4 = pd.DataFrame(
		{
		'pid' : [10,10,20,20,30,30,40,40],
		'id': [1,2,3,4,5,6,7,8],
		'col': ['a','a','b','b','c','c','a*','a*']
		}
	)

	matches_4 = [
		(1,8),
	]

	df_left, df_right, df_matches = SetupData(
		matches =matches_4
	).data_preparation(
			df_panel = df_4,
			unique_id = 'id',
			panel_id = 'pid'
		)

	# print(f'df_left:\n{df_left}','\n')
	# print(f'df_right:\n{df_right}','\n')
	print(f'df_matches:\n{df_matches}','\n')

