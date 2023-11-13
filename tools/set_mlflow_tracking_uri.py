
from dotenv import load_dotenv
load_dotenv()
import os
import mlflow
def set_tracking_uri():
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)