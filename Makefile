install:
	pip install .[core,profiling,lint,unittest,dev]

fix_lint:
	isort src/ tests/
	black src/ tests/ --line-length 120
	flake8 src/ --max-line-length 120
	toml-sort pyproject.toml -i

check_lint:
	black --check src/ tests/ --line-length 120
	flake8 src/ --max-line-length 120
	toml-sort --check pyproject.toml
	mypy src --ignore-missing-imports

unit_test:
	pytest tests

data_checks:
	pytest data_checks

# Train command with parameters
train:
	python src/topic_transition/scripts/train_on_multiple_datasets.py configs/training/multi/$(dataset)/confusion_$(model)_l$(L).yml

# Evaluate command with parameters
eval:
	python src/topic_transition/scripts/evaluate_model.py configs/evaluation/$(dataset)/confusion_$(model)_l$(L).yml

# Special case for evaluate_tvd
eval_tvd:
	python src/topic_transition/scripts/evaluate_tvd.py configs/evaluation/$(dataset)/tvd$(L).yml

summarize:
	python src/topic_transition/scripts/summarize.py configs/summary/$(dataset).yml

generate_artificial_split:
	python src/topic_transition/scripts/generate_guardian_with_split.py configs/artificial_split_data_generation/$(config).yml

profile:
	python src/topic_transition/scripts/profile_data.py dvc/datasets/$(dataset)/$(time_interval)/$(section).pkl configs/profiling/config_$(dataset).yml dvc/profilings/$(dataset)

comparison_plots:
	python src/topic_transition/scripts/generate_comparison_plots.py configs/training/multi/$(dataset)/confusion_$(model)_l$(L).yml configs/evaluation/$(dataset)/tvd$(L).yml
