.PHONY: help install format lint test train clean

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make format      - Format code with black and isort"
	@echo "  make lint        - Run linters (flake8, pylint)"
	@echo "  make test        - Run tests"
	@echo "  make train       - Train model"
	@echo "  make clean       - Remove build artifacts and cache"
	@echo "  make jupyter     - Start Jupyter Lab"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100
	pylint src/ --disable=C0111,C0103

test:
	pytest tests/ -v

train:
	python src/train.py --config config/config.yaml

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

jupyter:
	jupyter lab notebooks/
