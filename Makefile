install:
	python -m pip install -r requirements.txt

test:
	pytest -q

visuals:
	python scripts/make_visuals.py

report:
	python scripts/generate_report.py

smoke:
	python scripts/run_experiment.py --dataset ieee_cis --max-rows 10000 --epochs 5 --graph-mode auto --output-dir outputs/smoke_test

experiment:
	python scripts/run_experiment.py --dataset ieee_cis --max-rows 100000 --epochs 30 --history-k 3 --tune --graph-mode auto --output-dir outputs/benchmark_100k

ml-study:
	python scripts/run_ml_study.py --dataset ieee_cis --max-rows 50000 --epochs 20 --suite full --seeds 42 123 456 --output-dir outputs/ml_study_full

walk-forward:
	python scripts/run_temporal_cv.py --dataset ieee_cis --max-rows 50000 --epochs 10 --output-dir outputs/walk_forward
