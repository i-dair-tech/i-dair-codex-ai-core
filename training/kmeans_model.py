import mlflow.sklearn
from sklearn.cluster import KMeans
from skopt import BayesSearchCV
import sklearn
from datetime import datetime
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.utils import resample
import copy 
import os 
import sys
import glob
import joblib
import shutil
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

tools_path = os.path.join(dataset_path_prefix,'tools')
sys.path.insert(0, tools_path)

from db_tools import add_trained_model, epochs_tracker 
from custom_exceptions import CustomException
from set_mlflow_tracking_uri import set_tracking_uri
import logging

logger = logging.getLogger("main")

epoch=1
def on_step (optim_result):
    global id_model 
    global id_dataset
    global epoch
    global session_id
    epochs_tracker(session_id,id_model,id_dataset,epoch)
    print(f'Epoch {epoch}')  # print the current epoch
    epoch+=1
    if epoch > iterations:
        epoch=1
    
id_model=None
id_dataset=None
iterations=None 
session_id=None

def kmeans_model(user_id,session,model_id,dataset_id,x,n_iter,optimizer="BayesianOptimization",param_space=None):
    global id_model 
    global id_dataset
    global iterations
    global epoch
    global session_id
    session_id=int(session)
    iterations=copy.deepcopy(n_iter)
    id_model= copy.deepcopy(model_id)
    id_dataset=copy.deepcopy(dataset_id)
    n_samples = 100
    # Define the model and the hyperparameter search space
    model = KMeans()

    folds_number=param_space.get("folds_number",None)
    if folds_number:
        del param_space["folds_number"]
    
    if folds_number:
        cross_validation_method="StratifiedKFold"
    else:
        cross_validation_method="3-fold cross validation"
    joblib.dump(param_space,f"{session}/{model_id}/param_space.joblib")
    
    timestamp = datetime.now()
    # set Tracking Uri
    set_tracking_uri()
    # Create the experiment
    experiment_name = 'Kmeans Experiment '+str(timestamp)
    experiment_id = mlflow.create_experiment(experiment_name)
     
    # define the silhouette_score function before you use it
    def silhouette_score(estimator,x):
        pred=estimator.predict(x)
        return sklearn.metrics.silhouette_score(x, pred, metric="euclidean")
    if optimizer=="RandomSearch":
        # Create the random search object
        search = RandomizedSearchCV(model, param_space, n_iter=n_iter, n_jobs=-1, cv=folds_number)
    elif optimizer=="BayesianOptimization" :   
        # Create the Bayesian optimization search object
        search = BayesSearchCV(model, param_space, n_iter=n_iter, n_jobs=-1, cv=folds_number,scoring=silhouette_score)
        
    
    
    # Start an mlflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        try:
            run_id = run.info.run_id
            logger.info(f"train launched run_id:{run_id} in task/session: {session_id}",extra={'user_id': user_id})
            add_trained_model(run_id,id_model,id_dataset,0,session)
            # Fit the model to the training data
            search.fit(x,callback=on_step)
        except Exception as e:
            # Log the error message
            mlflow.log_param('error_message', str(e))
            raise CustomException(f"{model_id}-Possible issue with Hyperparameter combination or search: {e}")
        else:
            epochs_tracker(id_model,id_dataset,epoch,"completed")
            
            # Get the best model from BayesSearchCV
            best_model = search.best_estimator_

            # Get the cluster labels
            cluster_labels = best_model.labels_
            
            # Evaluate the model's performance
            silhouette_score = sklearn.metrics.silhouette_score(x, cluster_labels, metric="euclidean")
            
            # Initialize an empty list to store the silhouette scores
            scores = []
            
            # Perform bootstrapping
            for i in range(n_samples):
                xbootstrap = resample(x, random_state=i)
                
                scores.append(sklearn.metrics.silhouette_score(xbootstrap, best_model.predict(xbootstrap),metric="euclidean"))
            
            # Print the mean and standard deviation of the silhouette scores
            print("Mean silhouette score: {:.3f} +/- {:.3f}".format(np.mean(scores), np.std(scores)))
            print(np.array(scores).shape)
            
            # Define the number of clusters
            bootstrap_mean_silhouette_score=np.mean(scores)
            bootstrap_std_silhouette_score=np.std(scores)
            mlflow.log_metric('bootstrap_mean_silhouette_score', bootstrap_mean_silhouette_score)
            mlflow.log_metric('bootstrap_std_silhouette_score', bootstrap_std_silhouette_score) 
            ch_score = sklearn.metrics.calinski_harabasz_score(x, cluster_labels)
            
            # Log the evaluation metrics
            mlflow.log_metric('silhouette_score', silhouette_score)
            mlflow.log_metric('calinski_harabasz_score', ch_score)
            
            #log optimizer
            mlflow.log_param("optimizer",optimizer)            
            
            #log cross validation method
            mlflow.log_param("cross_validation_method",cross_validation_method)
            
            #log the number of folds
            mlflow.log_param("folds_number",folds_number)
            
            # Log iterations
            mlflow.log_param("iterations",n_iter)
            
            mlflow.log_params(search.best_params_)
            registered_model_name="Kmeans_model_"+str(timestamp)

            data_preprocessing_files = glob.glob(f"{session}/*.joblib")
            for file in data_preprocessing_files:
                if file.find("feature_engineering_tracking.joblib")==-1: 
                    mlflow.log_artifact(file)
                    
            feature_engineering_files = glob.glob(f"{session}/{model_id}/*.joblib")
            for file in feature_engineering_files:
                mlflow.log_artifact(file)             
            # Log the model
            mlflow.sklearn.log_model(search.best_estimator_, 'model',registered_model_name=registered_model_name) 
            shutil.rmtree(f"{session}/{model_id}")
            epochs_tracker(session_id,id_model,id_dataset,n_iter,"completed")
            epoch=1
