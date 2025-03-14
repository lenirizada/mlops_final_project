# MLOps Final Project

By Arvie Asis, Samn Mercado, and Leni Rizada as a requirement for Machine Learning Operations class.

## Project Overview
This project aims to create an end-to-end machine learning pipeline using Docker, Dagster, MLflow, FastAPI, and JupyterLab. The pipeline includes data fetching, preprocessing, model training, evaluation, and deployment.

## Directory Structure
- `.github/workflows` - GitHub Actions CI/CD pipeline configurations
- `dockerfiles` - Dockerfiles for various services
- `src` - Source code directory
  - `api` - FastAPI source code for model predictions
  - `dagster` - Dagster pipelines for data processing and model training
  - `notebooks` - Jupyter notebooks for testing
- `tests` - Test directory

## Deploying Docker Locally
- Deploying:
  ```sh
  docker-compose --env-file sample_env up --build -d

- Cleanup:
docker-compose down

Accessing Services
Dagster: http://localhost:3000
MLflow: http://localhost:5000
MinIO: http://localhost:9001
JupyterLab: http://localhost:8888
FastAPI: http://localhost:8000/docs


## Work in Progress
This project is still a work in progress.