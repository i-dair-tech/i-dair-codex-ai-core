# ri_codex_ai_core
## Install requirements
- **Step1:** Install it from the requirement file:
   ```
    pip install -r requirement.txt
  ```
- **Step2:**  Add automatic instrumentation
  ```
    opentelemetry-bootstrap --action=install
  ```
## Create the environment variables

- **Step1:** Open the Django project folder
- **Step2:** Create a file and name it ".env"
- **Step3:** Add the environment variables to the ".env" file :

```
APP_DB_NAME=app database name
APP_DB_USER =app database username
APP_DB_PASSWORD=app database password
APP_DB_HOST=Your app database host

CELERY_DB_NAME=your Database To Store Celery Results
CELERY_DB_USER=your celery database username
CELERY_DB_PASSWORD= your celery database password
CELERY_DB_HOST=your celery database host

RABBITMQ_USER= your RabbitMQ username
RABBITMQ_PASSWORD=your RabbitMQ Password
RABBITMQ_HOST=your RabbitMQ host

MEMCACHED_LOCATION= your Memcached location (E.g. "memcached: port")

BASE_URL_MLFLOW_REST_API=http://127.0.0.1:5000/api/2.0/mlflow
MLFLOW_TRACKING_URI=mysql://YourDbUser:YourDbPassword@localhost:3306/your_mlfow_schema_name
IS_LOCAL= True or False
SECRET_KEY= Django secret key
OTLP_ENDPOINT= by default it's http://localhost:4318/v1/traces
```
## How to Generate a Secret Key in Django
- **Step1:** Access the Python Interactive Shell:
  ```
    python manage.py shell
  ```
  - **Step2:**  Import the get_random_secret_key():
  ```
    from django.core.management.utils import get_random_secret_key
  ```
  - **Step3:** Generate the Secret Key in the Terminal
  ```
    print(get_random_secret_key())
  ```

## How To run the Django project

- **Step1:** Open the Django project folder (ri_codex_ai_core)
- **Step2:**
  ```
    python manage.py makemigrations
  ```
  ```
    python manage.py migrate
  ```
  ```
    python  manage.py loaddata models.json
  ```
  ```
    python manage.py migrate django_celery_results
  ```
- **Step3:** Run the Django runserver command:` python manage.py runserver`
  
## How To run the Celery:
```
celery -A ri_codex_ai_core worker --loglevel=info
```
## How To run the MLflow:
- **Step1:** Create your mlruns a folder
- **Step2:** Create your Mlflow schema
- **Step3:** Run Mlflow command
```
mlflow server --backend-store-uri "mysql://YourDbUser:YourDbPassword@localhost:3306/your_mlfow_schema_name" --default-artifact-root your_mlruns_folder_path
```

