# mlops_final_project

## Important links
- dagster [http://localhost:3000](http://localhost:3000)
- mlflow [http://localhost:5000](http://localhost:5000)
- jupyterlab [http://localhost:8888](http://localhost:8888)
- fastapi
  - [http://localhost:8000/docs](http://localhost:8000/docs)
  - to get predictions via cURL (or use notebook in jupyterlab)
  ```commandline
    curl --location --request GET 'https://localhost:8000/predict' \
    --header 'Content-Type: application/json' \
    --data '{
        "data": [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.1, 4.4, 1.4],
            [6.7, 3.1, 4.4, 1.4]
        ]
    }'
    ```
## Github Actions
- pre-commit
  - See `.github/workflows/pre-commit.yml` and `.pre-commit-config.yaml`
- pytest