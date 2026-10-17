.PHONY: install install-full test pipeline pipeline-light app docker docker-run notebooks mlflow-ui clean

install:            ## runtime deps only (dashboard)
	pip install -r requirements.txt

install-full:       ## everything: Prophet, TensorFlow, MLflow, Optuna, notebooks
	pip install -r requirements-full.txt

test:
	python -m pytest tests -q

pipeline:           ## full pipeline (Prophet + LSTM + MLflow tracking)
	python scripts/run_pipeline.py --mlflow

pipeline-light:     ## fast pipeline (GBM only)
	python scripts/run_pipeline.py --light

app:
	streamlit run dashboard/app.py

docker:
	docker build -t retailpulse:latest .

docker-run: docker
	docker run --rm -p 8501:8501 retailpulse:latest

notebooks:
	./rerun_all.sh

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache mlruns notebooks/mlruns mlflow/mlflow.db
