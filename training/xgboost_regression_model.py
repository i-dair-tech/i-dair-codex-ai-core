import mlflow.xgboost
from skopt import BayesSearchCV
from set_decimals import set_decimals
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import copy
import os 
import sys
import glob
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import json
import  shutil
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"
tools_path = os.path.join(dataset_path_prefix,'tools')
sys.path.insert(0, tools_path)

from db_tools import add_trained_model, epochs_tracker 
from bootsrapfuns import bootstrap

from custom_exceptions import CustomException
import logging
from custom_permutation_importance import codex_PI,generate_permutation_summary, plot_PI
from set_mlflow_tracking_uri import set_tracking_uri
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
        epoch=0
    
def rmse_scoring(estimator, x, y):
    pred = estimator.predict(x)
    return mean_absolute_error(y, pred)

id_model=None
id_dataset=None 
iterations=None
session_id=None


def xgb_regression(user_id,session,model_id,dataset_id,x_train,y_train,x_test,y_test,n_iter,optimizer="BayesianOptimization",param_space=None):
    
    # Define the number of bootstrap samples to generate
    nperms = 100
    
    global id_model 
    global id_dataset
    global iterations
    global session_id
    session_id=int(session)
    iterations=copy.deepcopy(n_iter)
    id_model= copy.deepcopy(model_id)
    id_dataset=copy.deepcopy(dataset_id)
    folds_number=param_space.get("folds_number",None)
    scores= {"mse":mean_squared_error, "rmse":mean_squared_error,"r2":r2_score}
    if folds_number:
        del param_space["folds_number"]
    
    if folds_number:
        cross_validation_method="StratifiedKFold"
    else:
        cross_validation_method="3-fold cross validation"
    # Define the model and the hyperparameter search space
    model = XGBRegressor(booster='gbtree',objective='reg:squarederror',importance_type='gain')

    joblib.dump(param_space,f"{session}/{model_id}/param_space.joblib")
    
    # set Tracking Uri
    set_tracking_uri()
    
    # Create the timestamp
    timestamp = datetime.now()
    
    # Create the experiment
    experiment_name = 'XGBoost Regression Experiment '+ str(timestamp)
    experiment_id = mlflow.create_experiment(experiment_name)
    
    if optimizer=="RandomSearch":
        # Create the random search object
        search = RandomizedSearchCV(model, param_space, n_iter=n_iter, n_jobs=-1, cv=folds_number)
    elif optimizer=="BayesianOptimization" :
        # Create the Bayesian optimization search object
        search = BayesSearchCV(model, param_space, n_iter=n_iter, cv=folds_number, scoring=rmse_scoring,random_state=42)
        
    
    # Start an mlflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        try:
            run_id = run.info.run_id
            logger.info(f"train launched run_id:{run_id} in task/session: {session_id}",extra={'user_id': user_id})
            add_trained_model(run_id, id_model, id_dataset, 0, session)
            # Fit the model to the training data
            search.fit(x_train, y_train, callback=on_step)
        except Exception as e:
            # Log the error message
            mlflow.log_param('error_message', str(e))
            raise CustomException(f"{model_id}-Possible issue with Hyperparameter combination or search: {e}")
        else:
            # Get the best model from BayesSearchCV
            best_model = search.best_estimator_
            
            # Make predictions on the test data
            predictions = best_model.predict(x_test)

            # Calculate the permutation feature importance
            importances=codex_PI(session,model=best_model, data=[x_test,y_test], criterion=mean_squared_error, nreps=100)
            importances = importances.reset_index(drop=True)
            
            importances_summary=generate_permutation_summary(importances,type="variables")
            
            PI_mean_variables = plot_PI(importances_summary, df_type='mean', type="variables")
            
            pio.write_json(PI_mean_variables, f'{session}/{model_id}/importance_fig.json')
            
            importances.to_json(f"{session}/{model_id}/importance_scores.json")
            


            # Evaluate the model's performance
            mse = mean_squared_error(y_test, predictions)
            rmse = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Evaluate the model's performance
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)


            mlflow.log_metric('mse', set_decimals(mse))
            mlflow.log_metric('rmse', set_decimals(rmse))
            mlflow.log_metric('r2', set_decimals(r2))
            
            # Bootsrap model callibration
            bootstrap_estimates = bootstrap(model=best_model, X=x_test, y=y_test, metrics=scores, model_name="XGBoost", nreps=nperms)
            bootstrap_estimates.to_csv(f"{session}/{model_id}/bootstrap_estimates.csv",index=False)
            mlflow.log_artifact(f'{session}/{model_id}/bootstrap_estimates.csv')

            data_preprocessing_files = glob.glob(f"{session}/*.joblib")
            for file in data_preprocessing_files:
                if file.find("feature_engineering_tracking.joblib")==-1: 
                    mlflow.log_artifact(file)
                    
            feature_engineering_files = glob.glob(f"{session}/{model_id}/*.joblib")
            for file in feature_engineering_files:
                mlflow.log_artifact(file)                
                
            mlflow.log_artifact(f"{session}/{model_id}/importance_fig.json")   
            mlflow.log_artifact(f"{session}/{model_id}/importance_scores.json")
            
            #log optimizer
            mlflow.log_param("optimizer",optimizer)
            
            # log cross validation method
            mlflow.log_param("cross_validation_method",cross_validation_method)
            
            #log the number of folds
            mlflow.log_param("folds_number",folds_number)
            # Log iterations
            mlflow.log_param("iterations", n_iter)
            # Log the best hyperparameters
            mlflow.log_params(search.best_params_)
            registered_model_name = "XGBoost_regression_model_" + str(timestamp)
            # Log the model
            mlflow.sklearn.log_model(search.best_estimator_, 'model', registered_model_name=registered_model_name)
            shutil.rmtree(f"{session}/{model_id}")
            epochs_tracker(session_id,id_model, id_dataset, n_iter, "completed")
            global epoch
            epoch=1            