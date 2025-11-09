.PHONY: reproduce install test lint clean docker-build docker-up docker-down

# Reproduce entire pipeline from scratch
reproduce:
	@echo "🔄 Reproducing full analysis pipeline..."
	dvc repro

# Install dependencies
install:
	pip install -r requirements.txt
	pre-commit install
	dvc pull

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

# Lint code
lint:
	black src/ tests/
	ruff check src/ tests/ --fix

# Clean generated files
clean:
	rm -rf results/*.csv
	rm -rf data/cache/*
	rm -rf .pytest_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

# Build Docker image
docker-build:
	docker-compose build

# Run Docker stack
docker-up:
	docker-compose up -d

# Stop Docker stack
docker-down:
	docker-compose down

# Check Docker image size
docker-size:
	@docker images | grep acct445 | awk '{print $$1 ":" $$2 " - " $$7 " " $$8}'
